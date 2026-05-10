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
            # LLaMA / DeepSeek / Qwen / Mistral
            lm = self.llm.model.model if self.use_lora else self.llm.model
            return lm.layers
        except AttributeError:
            # OPT
            dec = self.llm.model.model.decoder if self.use_lora else self.llm.model.decoder
            return dec.layers

    def _compute_eeg_positions(self, input_ids1, input_ids2, mm_seq_len):
        """Return per-batch EEG token start indices in the final padded sequence.
        Must stay in sync with the layout built by prepare_inputs."""
        eff1 = (input_ids1 != self.padding_token_id).sum(dim=1).long()
        eff2 = (input_ids2 != self.padding_token_id).sum(dim=1).long()
        final_max_len = mm_seq_len + eff1.max().item() + eff2.max().item()
        total_lens = mm_seq_len + eff1 + eff2
        # layout: [pad | prompt(eff1) | eeg(mm_seq_len) | response(eff2)]
        return final_max_len - total_lens + eff1  # (B,)

    def _make_injection_hook(self, eeg_embeds, eeg_positions, mm_seq_len):
        """Pre-forward hook: overwrites EEG placeholder positions with actual embeddings.
        Fires on the input hidden states of the target layer before it runs.
        Skips injection on KV-cache steps where seq_len shrinks to 1 token."""
        # --- original: in-place assignment severs gradient graph on 8-bit hidden states ---
        # def hook(module, args):
        #     hidden = args[0].clone()
        #     for b in range(hidden.size(0)):
        #         start = eeg_positions[b].item()
        #         end   = start + mm_seq_len
        #         if end <= seq_len:
        #             hidden[b, start:end] = eeg_embeds[b]
        #     return (hidden,) + args[1:]

        # out-of-place torch.scatter preserves the gradient graph through both
        # hidden states (LoRA path) and eeg_embeds (mm_proj path).
        def hook(module, args):
            hidden = args[0]
            seq_len = hidden.size(1)
            B, _, H = hidden.shape

            # skip KV-cache auto-regressive steps
            if (eeg_positions + mm_seq_len).max().item() > seq_len:
                return args

            # index[b, t, h] = eeg_positions[b] + t  →  shape (B, mm_seq_len, H)
            idx = (eeg_positions.view(B, 1, 1).to(hidden.device) +
                   torch.arange(mm_seq_len, device=hidden.device).view(1, mm_seq_len, 1))
            idx = idx.expand(B, mm_seq_len, H).long()

            # returns a new tensor — does not modify hidden in-place
            new_hidden = torch.scatter(hidden, 1, idx, eeg_embeds.to(hidden.dtype))
            return (new_hidden,) + args[1:]
        return hook

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
            # We are working on pooled embeddings now, but in the future, patched embeddings are possible
            # prepare_inputs assumes a sequence of mm_embeds, hence a shape of B*S*N
            # Pooled embeddings : B *N -> B*S*N -> B*1*N for now
            mm_embeds = mm_embeds.unsqueeze(1)

        mm_seq_len = mm_embeds.shape[1]

        if injection_layer > 0:
            # mid-layer injection: EEG tokens are present from input level so that
            # gradient checkpointing sees requires_grad=True inputs (needed for backward
            # to work with 8-bit frozen base weights). The hook re-injects mm_embeds at
            # layer L via out-of-place scatter, which is the dominant gradient path.
            # --- original: zeros as placeholder broke gradient checkpointing ---
            # zeros = torch.zeros_like(mm_embeds)
            # final_input_embeds, attention_masks, labels = self.prepare_inputs(
            #     input_ids1=input_ids1, input_ids2=input_ids2, mm_emb=zeros)
            eeg_positions = self._compute_eeg_positions(input_ids1, input_ids2, mm_seq_len)
            final_input_embeds, attention_masks, labels = self.prepare_inputs(
                input_ids1=input_ids1, input_ids2=input_ids2, mm_emb=mm_embeds)
            hook_handle = self._get_layers()[injection_layer].register_forward_pre_hook(
                self._make_injection_hook(mm_embeds, eeg_positions, mm_seq_len))
        else:
            # original path: EEG is present from the input embedding layer
            final_input_embeds, attention_masks, labels = self.prepare_inputs(
                input_ids1=input_ids1, input_ids2=input_ids2, mm_emb=mm_embeds)
            hook_handle = None

        try:
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
                # decoder only models like OPT
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
        finally:
            if hook_handle is not None:
                hook_handle.remove()

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
            # We are working on pooled embeddings now, but in the future, patched embeddings are possible
            # prepare_inputs assumes a sequence of mm_embeds, hence a shape of B*S*N
            # Pooled embeddings : B *N -> B*S*N -> B*1*N for now
            mm_embeds = mm_embeds.unsqueeze(1)
        # Switch this on if you want to test without projector
        # mm_embeds = torch.zeros_like(mm_embeds).to(mm_embeds.device)

        mm_seq_len = mm_embeds.shape[1]

        if injection_layer > 0:
            # same hook strategy as forward(): zeros placeholder, inject at layer L
            eeg_positions = self._compute_eeg_positions(input_ids1, input_ids2, mm_seq_len)
            zeros = torch.zeros_like(mm_embeds)
            final_input_embeds, _, labels = self.prepare_inputs(
                input_ids1=input_ids1, input_ids2=input_ids2,
                mm_emb=zeros, type="inference",
            )
            hook_handle = self._get_layers()[injection_layer].register_forward_pre_hook(
                self._make_injection_hook(mm_embeds, eeg_positions, mm_seq_len))
        else:
            # original path
            final_input_embeds, _, labels = self.prepare_inputs(
                input_ids1=input_ids1, input_ids2=input_ids2,
                mm_emb=mm_embeds, type="inference",
            )
            hook_handle = None

        try:
            output_ids = self.llm.generate(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                inputs_embeds=final_input_embeds,
                **kwargs,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        return output_ids, labels
