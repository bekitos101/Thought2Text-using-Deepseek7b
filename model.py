from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from config import EEGEncoderConfig, EEGModelForCausalLMConfig
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from typing import Optional, Tuple
import torch.nn.functional as F
import torch
import logging
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import os
from model_utils import get_lora_target_modules

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


class EEGModelForCausalLM(PreTrainedModel):
    config_class = EEGModelForCausalLMConfig
    base_model_prefix = "eegllm"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        eeg_encoder: Optional[PreTrainedModel] = None,
        llm: Optional[PreTrainedModel] = None,
        use_lora=False,
        token_inject=False,
    ):
        if config is None and (eeg_encoder is None or llm is None):
            raise ValueError(
                "Either a configuration or an eeg_encoder and an LLM model has to be provided."
            )
        if config is None:
            config = EEGModelForCausalLMConfig.from_separate_configs(
                eeg_encoder.config, llm.config
            )

        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"Config: {config} has to be of type {self.config_class}"
                )
        super().__init__(config)
        if eeg_encoder is None:
            eeg_encoder = ChannelNetModel(config=config.eeg_encoder)

        if llm is None:
            llm = AutoModelForCausalLM.from_config(
                config.llm,
                attn_implementation=config._attn_implementation,
            )

        self.eeg_encoder = eeg_encoder
        self.llm = llm
        self.padding_token_id = self.llm.config.eos_token_id
        self.bos_token_id = self.llm.config.bos_token_id
        self.use_lora = use_lora
        self.token_inject = token_inject

        if self.eeg_encoder.config.to_dict() != self.config.eeg_encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.eeg_encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.eeg_encoder}"
            )
        if self.llm.config.to_dict() != self.config.llm.to_dict():
            logger.warning(
                f"Config of the decoder: {self.llm.__class__} is overwritten by shared decoder config:"
                f" {self.config.llm}"
            )

        self.eeg_encoder.config = self.config.eeg_encoder
        self.llm.config = self.config.llm

        if token_inject:
            # per-token projection: each of the 10 temporal tokens (dim=out_channels)
            # is projected independently to the LLM hidden size.
            # nn.Linear broadcasts over the sequence dim, so (B,10,50) -> (B,10,hidden).
            self.mm_proj = nn.Linear(
                self.eeg_encoder.config.out_channels,
                self.llm.config.hidden_size,
            )
        elif self.eeg_encoder.config.embedding_size != self.llm.config.hidden_size:
            # original single-token path
            self.mm_proj = nn.Linear(
                self.eeg_encoder.config.embedding_size,
                self.llm.config.hidden_size,
            )
        else:
            self.mm_proj = nn.Linear(
                self.eeg_encoder.config.embedding_size,
                self.eeg_encoder.config.embedding_size,
            )

    def get_eeg_encoder(self):
        return self.eeg_encoder

    def get_llm(self):
        return self.llm

    def _get_layers(self):
        """Return the transformer block list for any supported model family."""
        try:
            lm = self.llm.model.model if self.use_lora else self.llm.model
            return lm.layers
        except AttributeError:
            dec = self.llm.model.model.decoder if self.use_lora else self.llm.model.decoder
            return dec.layers

    def _get_backbone(self):
        """Return the inner backbone (LlamaModel or equivalent)."""
        try:
            return self.llm.model.model if self.use_lora else self.llm.model
        except AttributeError:
            return self.llm.model.model.decoder if self.use_lora else self.llm.model.decoder

    # ── True DeepInsert (EACL 2026) ──────────────────────────────────────────
    # Stage 1: text-only [pad|prompt|response] through layers 0..L-1 (no EEG).
    # At layer L: EEG tokens physically inserted into the hidden-state tensor.
    # Stage 2: expanded sequence through layers L..N-1.

    def _deepinsert_forward(self, input_ids1, input_ids2, mm_embeds, injection_layer):
        """Training forward with true two-stage bypass."""
        backbone = self._get_backbone()
        layers   = backbone.layers
        pad_id   = self.padding_token_id
        B        = input_ids1.shape[0]
        E        = mm_embeds.shape[1]
        H        = backbone.config.hidden_size
        device   = input_ids1.device
        dtype    = backbone.embed_tokens.weight.dtype

        eff1     = (input_ids1 != pad_id).sum(1).long()
        eff2     = (input_ids2 != pad_id).sum(1).long()
        max_eff1 = eff1.max().item()
        max_eff2 = eff2.max().item()
        T        = max_eff1 + max_eff2
        mm_embeds = mm_embeds.to(dtype)   # align with embed_tokens dtype

        # ── Build text-only [pad | prompt | response] ───────────────────────
        text_embeds = torch.zeros(B, T, H, device=device, dtype=dtype)
        text_labels = torch.full((B, T), IGNORE_INDEX, device=device, dtype=torch.long)
        for i in range(B):
            e1, e2 = eff1[i].item(), eff2[i].item()
            p = T - e1 - e2
            text_embeds[i, p:p+e1] = backbone.embed_tokens(input_ids1[i, -e1:])
            text_embeds[i, p+e1:T] = backbone.embed_tokens(input_ids2[i, -e2:])
            text_labels[i, p:p+e1] = input_ids1[i, -e1:]
            text_labels[i, p+e1:T] = input_ids2[i, -e2:]

        pos_ids_1 = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # 4-D causal+padding mask: prevents real tokens from attending to
        # zero-padded positions, matching the original prepare_inputs behaviour.
        # Shape: (B, 1, seq_len, seq_len) — added to raw attention scores.
        NEG_INF = torch.finfo(dtype).min
        def _causal_pad_mask(seq_len, pad_starts):
            # pad_starts: (B,) int tensor — index of the first real token per element
            causal   = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            mask     = torch.full((B, 1, seq_len, seq_len), NEG_INF, device=device, dtype=dtype)
            for i in range(B):
                valid_kv  = torch.arange(seq_len, device=device) >= pad_starts[i]  # (seq_len,)
                mask[i, 0] = torch.where(causal & valid_kv.unsqueeze(0), 0.0, NEG_INF)
            return mask  # (B, 1, seq_len, seq_len)

        pad_starts = (T - eff1 - eff2).clamp(min=0)   # (B,) — 0 when no padding
        attn1 = _causal_pad_mask(T, pad_starts)

        # ── Stage 1: text-only through layers 0..L-1 ────────────────────────
        with torch.no_grad():
            hidden = text_embeds
            for layer in layers[:injection_layer]:
                hidden = layer(hidden, attention_mask=attn1,
                               position_ids=pos_ids_1, use_cache=False)[0]

        # ── Insert EEG between prompt and response hidden states ─────────────
        eeg_ign = torch.full((E,), IGNORE_INDEX, device=device, dtype=torch.long)
        parts_h, parts_l = [], []
        for i in range(B):
            ins = T - eff2[i].item()          # split point: end of [pad|prompt]
            parts_h.append(torch.cat([hidden[i, :ins], mm_embeds[i], hidden[i, ins:]], 0))
            parts_l.append(torch.cat([text_labels[i, :ins], eeg_ign, text_labels[i, ins:]], 0))

        hidden      = torch.stack(parts_h, 0)   # (B, T+E, H)
        full_labels = torch.stack(parts_l, 0)   # (B, T+E)
        T_full      = T + E
        pos_ids_2   = torch.arange(T_full, device=device).unsqueeze(0).expand(B, -1)

        # Stage-2 padding region is unchanged: EEG tokens are inserted after
        # the prompt (after the pad prefix), so pad_starts is the same.
        attn2 = _causal_pad_mask(T_full, pad_starts)

        # ── Stage 2: full sequence through layers L..N-1 ────────────────────
        for layer in layers[injection_layer:]:
            hidden = layer(hidden, attention_mask=attn2,
                           position_ids=pos_ids_2, use_cache=False)[0]

        target_dtype = self.llm.lm_head.weight.dtype
        logits = self.llm.lm_head(backbone.norm(hidden.to(target_dtype)))

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = full_labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits), full_labels

    def _deepinsert_generate(self, input_ids1, input_ids2, mm_embeds, injection_layer,
                              max_new_tokens=100, repetition_penalty=1.0, **kwargs):
        """Inference with true two-stage bypass and KV-cache.

        Mirrors _deepinsert_forward exactly:
          Stage 1: [pad|prompt|suffix] through layers 0..L-1
          EEG inserted between prompt_hidden and suffix_hidden
          Stage 2: [pad|prompt_h|EEG|suffix_h] through layers L..N-1
          Generate from after suffix → outputs caption only
        """
        from transformers.cache_utils import DynamicCache

        backbone  = self._get_backbone()
        layers    = backbone.layers
        pad_id    = self.padding_token_id
        eos_id    = self.llm.config.eos_token_id
        B         = input_ids1.shape[0]
        E         = mm_embeds.shape[1]
        H         = backbone.config.hidden_size
        device    = input_ids1.device
        dtype     = backbone.embed_tokens.weight.dtype

        eff1      = (input_ids1 != pad_id).sum(1).long()
        eff2      = (input_ids2 != pad_id).sum(1).long()
        max_eff1  = eff1.max().item()
        max_eff2  = eff2.max().item()
        T         = max_eff1 + max_eff2   # prompt + suffix (no EEG) in stage-1
        T_full    = T + E                 # prompt + EEG + suffix in stage-2
        mm_embeds = mm_embeds.to(dtype)

        # ── Build [pad | prompt | suffix] embeddings (mirrors training) ──────
        text_embeds = torch.zeros(B, T, H, device=device, dtype=dtype)
        for i in range(B):
            e1, e2 = eff1[i].item(), eff2[i].item()
            p = T - e1 - e2   # pad length (0 when no padding)
            text_embeds[i, p:p+e1]  = backbone.embed_tokens(input_ids1[i, -e1:])
            text_embeds[i, p+e1:T]  = backbone.embed_tokens(input_ids2[i, -e2:])

        cache  = DynamicCache()
        pos_1  = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        cpos_1 = torch.arange(T, device=device)

        with torch.no_grad():
            # ── Stage 1 prefill: [pad|prompt|suffix] through layers 0..L-1 ──
            hidden = text_embeds
            for layer in layers[:injection_layer]:
                hidden = layer(hidden, attention_mask=None, position_ids=pos_1,
                               past_key_value=cache, use_cache=True,
                               cache_position=cpos_1)[0]

            # ── Insert EEG between prompt_hidden and suffix_hidden ───────────
            parts_h = []
            for i in range(B):
                ins = T - eff2[i].item()  # end of [pad|prompt], start of suffix
                parts_h.append(
                    torch.cat([hidden[i, :ins], mm_embeds[i], hidden[i, ins:]], 0)
                )
            stage2_in = torch.stack(parts_h, 0)   # (B, T_full, H)

            # ── Stage 2 prefill: [pad|prompt_h|EEG|suffix_h] through layers L..N-1
            pos_2  = torch.arange(T_full, device=device).unsqueeze(0).expand(B, -1)
            cpos_2 = torch.arange(T_full, device=device)
            hidden = stage2_in
            for layer in layers[injection_layer:]:
                hidden = layer(hidden, attention_mask=None, position_ids=pos_2,
                               past_key_value=cache, use_cache=True,
                               cache_position=cpos_2)[0]

            # First generated token from last suffix position
            _td        = self.llm.lm_head.weight.dtype
            logit      = self.llm.lm_head(backbone.norm(hidden[:, -1:, :].to(_td)))[:, 0, :]
            next_token = logit.argmax(-1, keepdim=True)   # (B, 1)

        generated = [next_token]
        finished  = next_token.squeeze(1) == eos_id

        with torch.no_grad():
            for step in range(1, max_new_tokens):
                if finished.all():
                    break

                embed = backbone.embed_tokens(generated[-1])   # (B, 1, H)

                # Stage 1 decode: position T+step-1 (after full prompt+suffix)
                s1i   = T + step - 1
                s1pos = torch.full((B, 1), s1i, device=device, dtype=torch.long)
                s1cp  = torch.tensor([s1i], device=device, dtype=torch.long)
                hidden = embed
                for layer in layers[:injection_layer]:
                    hidden = layer(hidden, attention_mask=None, position_ids=s1pos,
                                   past_key_value=cache, use_cache=True,
                                   cache_position=s1cp)[0]

                # Stage 2 decode: position T_full+step-1 (after prompt+EEG+suffix)
                s2i   = T_full + step - 1
                s2pos = torch.full((B, 1), s2i, device=device, dtype=torch.long)
                s2cp  = torch.tensor([s2i], device=device, dtype=torch.long)
                for layer in layers[injection_layer:]:
                    hidden = layer(hidden, attention_mask=None, position_ids=s2pos,
                                   past_key_value=cache, use_cache=True,
                                   cache_position=s2cp)[0]

                logit = self.llm.lm_head(backbone.norm(hidden.to(_td)))[:, 0, :]

                if repetition_penalty != 1.0:
                    gen_ids = torch.cat(generated, dim=1)
                    for b in range(B):
                        for tok in gen_ids[b].tolist():
                            if logit[b, tok] < 0:
                                logit[b, tok] *= repetition_penalty
                            else:
                                logit[b, tok] /= repetition_penalty

                next_token = logit.argmax(-1, keepdim=True)
                generated.append(next_token)
                finished = finished | (next_token.squeeze(1) == eos_id)

        return torch.cat(generated, dim=1), None


    def save_pretrained(self, output_dir, *model_args, **kwargs):
        # we need to save all the models separately
        
        self.eeg_encoder.save_pretrained(
            os.path.join(output_dir, "eeg_encoder"), *model_args, **kwargs
        )
        # self.llm.save_pretrained(os.path.join(output_dir, "llm"), *model_args, **kwargs)
        torch.save(
            self.mm_proj.state_dict(),
            os.path.join(output_dir, "projector.pth"),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        use_lora=False,
        *model_args,
        **kwargs,
    ):
        # TODO: Implement Download From Hub Functionality
        eeg_encoder_path = os.path.join(pretrained_model_name_or_path, "eeg_encoder")
        projector_path = os.path.join(pretrained_model_name_or_path, "projector.pth")
        llm_path = os.path.join(pretrained_model_name_or_path, "llm")
        eeg_config_path = os.path.join(pretrained_model_name_or_path, "eeg_config.json")

        # --- original ---
        # model = cls.from_separate_pretrained(eeg_encoder_path, llm_path, ...)
        # model.mm_proj.load_state_dict(torch.load(projector_path))

        # auto-detect token_inject from projector weight shape:
        # token_inject=True  → mm_proj weight is (hidden, out_channels=50)
        # token_inject=False → mm_proj weight is (hidden, 512)
        proj_state = torch.load(projector_path, map_location="cpu")
        token_inject = (proj_state["weight"].shape[1] == 50)

        # load injection_layer from saved metadata if available
        injection_layer = 0
        if os.path.exists(eeg_config_path):
            import json
            with open(eeg_config_path) as f:
                eeg_cfg = json.load(f)
            injection_layer = eeg_cfg.get("injection_layer", 0)

        kwargs.setdefault("token_inject", token_inject)
        kwargs.setdefault("injection_layer", injection_layer)

        if use_lora:
            model = None
        else:
            model = cls.from_separate_pretrained(
                eeg_encoder_path=eeg_encoder_path,
                llm_path=llm_path,
                *model_args,
                **kwargs,
            )

        model.mm_proj.load_state_dict(proj_state)
        return model

    @classmethod
    def from_separate_pretrained(
        cls,
        eeg_encoder_path: str = None,
        llm_path: str = None,
        use_lora=False,
        token_inject=False,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:

        kwargs_eeg_encoder = {
            argument[len("eeg_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("eeg_encoder_")
        }

        kwargs_llm = {
            argument[len("llm_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("llm_")
        }
        for key in kwargs_eeg_encoder.keys():
            del kwargs["eeg_encoder_" + key]
        for key in kwargs_llm.keys():
            del kwargs["llm_" + key]

        eeg_encoder = kwargs_eeg_encoder.pop("model", None)

        if eeg_encoder is None:
            if eeg_encoder_path is None:
                raise ValueError(
                    "If `eeg_encoder_model` is not defined as an argument, a `eeg_encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_eeg_encoder:
                eeg_encoder_config, kwargs_eeg_encoder = (
                    EEGEncoderConfig.from_pretrained(
                        eeg_encoder_path,
                        **kwargs_eeg_encoder,
                        return_unused_kwargs=True,
                    )
                )

                kwargs_eeg_encoder["config"] = eeg_encoder_config

            eeg_encoder = ChannelNetModel.from_pretrained(
                eeg_encoder_path, *model_args, **kwargs_eeg_encoder
            )
        llm = kwargs_llm.pop("model", None)
        if llm is None:
            if llm_path is None:
                raise ValueError(
                    "If `llm` is not defined as an argument, a `llm_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_llm:
                llm_config, kwargs_llm = AutoConfig.from_pretrained(
                    llm_path,
                    **kwargs_llm,
                    return_unused_kwargs=True,
                )

                kwargs_llm["config"] = llm_config

            llm = AutoModelForCausalLM.from_pretrained(
                llm_path, device_map="auto", **kwargs_llm
            )

        if use_lora:
            # --- original hardcoded target modules (kept for reference) ---
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
            # --- new: resolved from the model family via model_utils ---
            lora_target_modules = get_lora_target_modules(llm_path or "")
            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
            )
            llm = get_peft_model(llm, peft_config)
            llm.print_trainable_parameters()
        config = EEGModelForCausalLMConfig.from_separate_configs(
            eeg_encoder_config=eeg_encoder.config, llm_config=llm.config, **kwargs
        )
        return cls(eeg_encoder=eeg_encoder, llm=llm, config=config, use_lora=use_lora,
                   token_inject=token_inject)

    def prepare_inputs(self, input_ids1, input_ids2, mm_emb, type="train"):
        batch_size, max_length = input_ids1.shape
        hidden_dim = mm_emb.shape[-1]
        mm_seq_len = mm_emb.shape[1]

        # Compute token embeddings
        if self.use_lora:
            try:
                input_embeds1 = self.llm.model.model.embed_tokens(input_ids1)
                input_embeds2 = self.llm.model.model.embed_tokens(input_ids2)
            except:
                # for decoder only models like OPT
                input_embeds1 = self.llm.model.model.decoder.embed_tokens(input_ids1)
                input_embeds2 = self.llm.model.model.decoder.embed_tokens(input_ids2)
        else:
            try:
                input_embeds1 = self.llm.model.embed_tokens(input_ids1)
                input_embeds2 = self.llm.model.embed_tokens(input_ids2)
            except:
                input_embeds1 = self.llm.model.decoder.embed_tokens(input_ids1)
                input_embeds2 = self.llm.model.decoder.embed_tokens(input_ids2)

        # Create attention masks (1 for non-padding tokens, 0 for padding tokens)
        attention_masks1 = (input_ids1 != self.padding_token_id).float()
        attention_masks2 = (input_ids2 != self.padding_token_id).float()

        # Compute the effective length for each input
        effective_lengths1 = attention_masks1.sum(dim=1).long()
        effective_lengths2 = attention_masks2.sum(dim=1).long()

        # Calculate the maximum effective lengths for positioning
        max_effective_length1 = max(effective_lengths1).item()
        max_effective_length2 = max(effective_lengths2).item()

        # Initialize final embeddings and labels
        final_max_length = mm_seq_len + max_effective_length1 + max_effective_length2

        final_input_embeds = torch.zeros(
            batch_size, final_max_length, hidden_dim, device=input_embeds1.device
        )
        attention_masks = torch.zeros(
            batch_size, final_max_length, device=input_embeds1.device
        )
        labels = torch.full(
            (batch_size, final_max_length),
            IGNORE_INDEX,
            device=input_ids1.device,
        )

        for i in range(batch_size):
            effective_length1 = effective_lengths1[i].item()
            effective_length2 = effective_lengths2[i].item()

            total_len = mm_seq_len + effective_length1 + effective_length2
            start_idx = final_max_length - total_len
            
            final_input_embeds[
                i,
                start_idx: start_idx + effective_length1,
                :,
            ] = input_embeds1[i, -effective_length1:, :]

            final_input_embeds[i, start_idx+ effective_length1 : start_idx + effective_length1+ mm_seq_len, :] = mm_emb[
                i, :, :
            ]

            final_input_embeds[
                i, start_idx + effective_length1 + mm_seq_len : final_max_length, :
            ] = input_embeds2[i, -effective_length2:, :]

            attention_masks[i, start_idx:] = 1
            labels[
                i, start_idx: start_idx + effective_length1
            ] = input_ids1[i, -effective_length1:]
            
            # labels[i, start_idx+ effective_length1 : start_idx + effective_length1+ mm_seq_len] = IGNORE_INDEX
            

            labels[i, start_idx + effective_length1 + mm_seq_len : final_max_length] = (
                input_ids2[i, -effective_length2:]
            )

        # Create position ids
        final_input_embeds = final_input_embeds.to(input_embeds1.dtype)

        # attention_masks = attention_masks.to(input_embeds1.dtype)
        # print(attention_masks)
        if type != "train":
            attention_masks = None

        return final_input_embeds, attention_masks, labels

    def forward(
        self,
        input_ids1,
        input_ids2,
        mm_embeds=None,
        injection_layer: int = 0,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        mm_embeds = self.mm_proj(mm_embeds)
        if len(mm_embeds.shape) == 2:
            mm_embeds = mm_embeds.unsqueeze(1)

        if injection_layer > 0:
            return self._deepinsert_forward(input_ids1, input_ids2, mm_embeds, injection_layer)

        # injection_layer == 0: original full-sequence path
        final_input_embeds, attention_masks, labels = self.prepare_inputs(
            input_ids1=input_ids1, input_ids2=input_ids2, mm_emb=mm_embeds)
        try:
            llm_outputs = self.llm(
                input_ids=None,
                attention_mask=attention_masks,
                inputs_embeds=final_input_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=return_dict,
                **kwargs,
            )
        except TypeError:
            llm_outputs = self.llm(
                input_ids=None,
                inputs_embeds=final_input_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=return_dict,
                **kwargs,
            )
        return llm_outputs, labels

    def generate(
        self,
        input_ids1,
        input_ids2,
        mm_embeds=None,
        injection_layer: int = 0,
        **kwargs,
    ):
        mm_embeds = self.mm_proj(mm_embeds)
        if len(mm_embeds.shape) == 2:
            mm_embeds = mm_embeds.unsqueeze(1)

        if injection_layer > 0:
            return self._deepinsert_generate(
                input_ids1, input_ids2, mm_embeds, injection_layer, **kwargs)

        # injection_layer == 0: original full-sequence path
        final_input_embeds, _, labels = self.prepare_inputs(
            input_ids1=input_ids1, input_ids2=input_ids2,
            mm_emb=mm_embeds, type="inference",
        )
        output_ids = self.llm.generate(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=final_input_embeds,
            **kwargs,
        )
        return output_ids, labels
