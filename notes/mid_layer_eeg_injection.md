# Mid-Layer EEG Token Injection in Multimodal LLMs
## Research Notes — Thought2Text Improvement Direction

---

## 1. Motivation

The standard approach to multimodal LLM (MLM) construction concatenates encoded non-language tokens (visual, audio, EEG, etc.) with the language prompt at the **input embedding level** — i.e., before the first transformer layer. This is the design used in the current Thought2Text baseline, where a single pooled EEG embedding is projected and prepended to the token sequence before being passed through all LLM layers.

Two recent findings challenge the premise that this is optimal:

1. Most cross-modal token interactions in MLMs are deferred to **deeper transformer layers**, meaning early layers perform little meaningful EEG-language grounding.
2. Multimodal tokens that already carry rich semantic representations (from a pre-trained encoder) do not benefit from the early-layer linguistic processing that text tokens require.

These observations motivate a simple but underexplored intervention: **inject EEG tokens directly at an intermediate transformer layer**, bypassing the early layers entirely.

---

## 2. Theoretical Foundations

### 2.1 DeepInsert — Mid-Layer Multimodal Injection

**Reference**: *DeepInsert* (abstract provided; full citation pending)

**Core claim**: MLMs naturally defer most cross-modal token interactions to deeper transformer layers. The early layers are therefore wasted when processing multimodal tokens, since those tokens will not meaningfully interact with language context until later anyway. DeepInsert proposes injecting multimodal tokens at the middle of the transformer stack instead of the input, bypassing early layers entirely.

**Empirical validation**: Tested across diverse modalities and model sizes:
- Vision: LLaVA, BLIP
- Audio: LTU
- Molecular data: MoLCA
- Model scale: 350M → 13B parameters

**Results**: Reduced both training and inference cost while at least preserving, and in several cases surpassing, baseline performance.

**Relevance to Thought2Text**: EEG has never been tested under this framework. The DeepInsert approach maps directly onto this codebase — the injection point is the only architectural change required.

---

### 2.2 Fan et al. — Visual Token Sparsity and Mid-Layer Alignment

**Reference**: Fan et al. (2026). *What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models*. arXiv:2603.00510.

**Core findings**:

**(a) Semantic sparsity at the input level.** Visual tokens consistently partition into three categories:
- **Sink tokens**: receive disproportionate attention mass but carry little semantic content (attention sinks, known from Mistral / StreamingLLM literature).
- **Dead tokens**: carry no image-specific information; essentially noise.
- **Alive tokens**: carry genuine image-specific semantic meaning. Comprise approximately 60% of total input tokens.

**(b) Internal visual computation is largely redundant.** Alive tokens already encode fine-grained cues (objects, colors, OCR text) *before* entering the LLM. The internal visual processing performed by self-attention and FFN layers is redundant for most standard tasks.

**(c) Mid-layer alignment.** For the subset of highly vision-centric tasks that do benefit from internal processing, alive tokens naturally align with **intermediate LLM layers** rather than the initial embedding space. This indicates that shallow-layer processing is unnecessary and that direct mid-layer injection is sufficient.

**Tool introduced**: EmbedLens — a probing framework for analyzing semantic content of individual multimodal tokens at each layer of an MLM.

**Relevance to Thought2Text**: This paper provides the mechanistic justification for mid-layer injection in this project. The alive/dead/sink taxonomy has never been applied to EEG tokens. Given that EEG signals are inherently noisy (artifacts, subject variability), the proportion of alive tokens is expected to be lower than for visual inputs — making the analysis especially relevant.

---

### 2.3 Flamingo — Cross-Attention Injection (Background Reference)

**Reference**: Alayrac et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS 2022. arXiv:2204.14198.

**Architecture**: Flamingo introduces interleaved cross-attention layers between frozen LM transformer blocks. Each new cross-attention block uses the visual features as keys and values, and the LM hidden states as queries. This gives every transformer layer persistent access to visual context.

**Relevance**: This represents a more architecturally expressive alternative to token injection. Rather than inserting EEG tokens into the sequence at a single depth L, cross-attention allows every layer to attend to EEG features. Flamingo is the foundational reference for this design pattern.

**Note**: Flamingo-style injection is the more complex and parameter-intensive approach. It is the natural follow-up to the mid-layer injection experiments described here, not the starting point.

---

### 2.4 ChannelNet — EEG Encoder

**Reference**: Palazzo, S., Spampinato, C., Kavasidis, I., Giordano, D., Schmidt, J., Shah, M. (2020). *Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features*. IEEE Transactions on Pattern Analysis and Machine Intelligence. doi:10.1109/TPAMI.2020.2995909.

**Architecture overview** (as used in this project):
- Input: `(B, 1, 128, 440)` — 1 channel, 128 EEG electrodes (height), 440 time samples (width)
- TemporalBlock: 4 dilated 2D conv layers, stride (1,2) — captures temporal patterns at multiple scales
- SpatialBlock: 4 spatial conv layers, stride (2,1) — captures cross-electrode spatial patterns
- 4 Residual blocks with progressive downsampling (stride 2)
- final_conv: reduces to `(B, 50, 1, 10)` — 50 feature channels, 1 spatial dim (collapsed), **10 temporal positions**

**Critical observation**: the current codebase applies `view(B, -1)` immediately after `final_conv`, collapsing the `(B, 50, 1, 10)` feature map into a flat `(B, 500)` vector, then projecting to `(B, 512)`. This destroys temporal structure that the encoder has preserved up to this point.

---

## 3. The EEG-Specific Hypothesis

> **Hypothesis**: EEG temporal tokens, like visual tokens, exhibit semantic sparsity (a subset carry signal-specific meaning while others are dominated by noise or are attention sinks). The LLM's early transformer layers perform no meaningful cross-modal processing on these tokens. Injecting EEG temporal tokens at an intermediate layer L bypasses this idle computation, reducing training and inference cost while preserving or improving text generation quality.

**Why this is stronger for EEG than for vision:**

Visual tokens from CLIP or ViT are clean, high-quality patch representations. EEG signals from ChannelNet are inherently noisy due to:
- Electrode artifacts and movement noise
- Subject-to-subject variability
- Temporal segments that coincide with non-cognitive states (e.g., blinking, rest)

This means the alive/dead/sink distribution for EEG tokens is likely more skewed than for visual tokens — a larger proportion of dead/noise tokens is expected. This is a novel finding that has not been reported in prior BCI or EEG-to-text literature.

---

## 4. Current Architecture (Thought2Text Baseline)

```
EEG signal (B, 1, 128, 440)
        ↓
ChannelNetModel.encoder → FeaturesExtractor
        ↓  (B, 50, 1, 10) — temporal feature map
view(B, -1)               ← destroys temporal structure
        ↓  (B, 500)
Linear(500, 512)          ← projector
        ↓  (B, 512)       ← single pooled EEG embedding
mm_proj: Linear(512, hidden_size)
        ↓  (B, hidden_size)
unsqueeze(1)
        ↓  (B, 1, hidden_size) ← single EEG "token"

Sequence: [prompt_embeds | EEG_token | response_embeds]
        ↓
LLM Layer 0 → Layer 1 → ... → Layer 27  (all 28 layers)
```

**Injection depth**: Layer 0 (input embedding level).
**Token count**: 1.

---

## 5. Proposed Architecture

### 5.1 Step 1 — Temporal Tokenization (resolves the single-token problem)

Modify `ChannelNetModel.forward()` to optionally return the temporal token sequence instead of the pooled embedding:

```python
def forward(self, x, return_tokens=False):
    out = self.encoder(x)           # (B, 50, 1, 10)

    if return_tokens:
        out = out.squeeze(2)        # (B, 50, 10)
        out = out.permute(0, 2, 1)  # (B, 10, 50)
        return out                  # 10 temporal EEG tokens

    # original path — unchanged, backward-compatible
    out = out.view(x.size(0), -1)
    emb = self.projector(out)
    cls = self.classifier(emb)
    return emb, cls
```

The 10 tokens represent 10 temporal windows across the 440-sample EEG recording (~44ms per token at 500Hz sampling). These are analogous to spatial patches in vision transformers, but along the temporal axis.

Correspondingly, update `mm_proj` in `EEGModelForCausalLM`:

```python
# from: Linear(512, hidden_size)  →  1 token
# to:   Linear(50, hidden_size)   →  applied per-token, gives 10 tokens
self.mm_proj = nn.Linear(50, self.llm.config.hidden_size)
```

The `prepare_inputs` method in `model.py` already handles variable-length multimodal sequences via `mm_seq_len` — no changes needed there.

### 5.2 Step 2 — Mid-Layer Injection

Replace the monolithic `self.llm(inputs_embeds=...)` call with a layer-by-layer forward pass that injects EEG tokens at depth L:

```python
def forward_with_injection(self, lang_embeds, eeg_embeds, injection_layer, attention_mask, labels):
    layers   = self.llm.model.layers       # list of transformer blocks
    norm     = self.llm.model.norm
    head     = self.llm.lm_head

    hidden   = lang_embeds
    prompt_len = ...                       # position at which to insert EEG

    for i, layer in enumerate(layers):
        if i == injection_layer:
            hidden = torch.cat([
                hidden[:, :prompt_len],
                eeg_embeds,
                hidden[:, prompt_len:]
            ], dim=1)
            # update attention_mask and position_ids accordingly

        layer_out = layer(hidden, attention_mask=..., position_ids=...)
        hidden = layer_out[0]

    logits = head(norm(hidden))
    # compute loss against labels
```

`injection_layer` becomes a configurable hyperparameter in `EEGModelForCausalLMConfig`.

---

## 6. Experimental Design

### 6.1 Ablation 1 — Injection Depth

**Variable**: `injection_layer` ∈ {0, 7, 14, 21} for a 28-layer DeepSeek 7B (layers N/4 intervals).

**Fixed**: 10 EEG temporal tokens, full token sequence.

**Metrics**:
- BLEU-1, BLEU-4
- ROUGE-1, ROUGE-L
- Training time per epoch
- Inference latency per sample

**Expected result**: Performance peaks at some intermediate L, with layer 0 (current baseline) being suboptimal. The optimal L for EEG is the primary finding.

**Comparison point**: DeepInsert reports optimal injection at approximately N/2 for vision and audio. EEG may differ.

---

### 6.2 Ablation 2 — Token Count

**Variable**: number of EEG tokens injected ∈ {1, 5, 10} (10 is the natural count from the feature map; 1 is current baseline; 5 is temporal subsampling).

**Fixed**: injection_layer = optimal L from Ablation 1.

**Goal**: Establish whether multiple temporal tokens provide meaningful signal over a single pooled token at the same injection depth.

---

### 6.3 Analysis — EEG Token Sparsity (Fan et al. Framework)

After training with the optimal configuration, probe the EEG token representations:

**Protocol** (adapted from EmbedLens):
1. For each sample, extract the 10 EEG token representations at the input and at several intermediate layers.
2. Compute cosine similarity between each EEG token and the attending language hidden states.
3. Classify each token as:
   - **Alive**: high cosine similarity with semantically relevant language tokens; high attention weight from language query vectors.
   - **Dead**: near-zero attention from all language positions across all layers.
   - **Sink**: disproportionately high attention mass but low semantic content (flat similarity profile).
4. Report the alive/dead/sink ratio and compare to the ~60% alive rate reported for visual tokens in Fan et al.

**Why this matters**: If EEG tokens show a lower alive ratio (e.g., 40%), it empirically validates the hypothesis that EEG signals carry more noise than visual tokens. This motivates follow-on work on selective token pruning (inject only alive tokens) — a natural extension that connects to the efficiency motivation.

---

### 6.4 Extension — Alive-Token-Only Injection

Once the alive/dead/sink classification is established:
- Re-run mid-layer injection using only the alive EEG tokens (e.g., 4–6 tokens instead of 10).
- Compare performance vs. full 10-token injection at the same depth.
- Report the further efficiency gain from token pruning.

This mirrors the "selective token pruning" direction recommended by Fan et al. and constitutes a self-contained follow-up experiment.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Small dataset → high variance in BLEU/ROUGE | High | Report across multiple splits; use paired statistical tests |
| Optimal L varies by subject | Medium | Run per-subject ablations on ZuCo subject splits |
| 10 temporal tokens don't carry more information than 1 (already pooled by architecture) | Medium | Ablation 2 directly tests this; if true, document as finding |
| Manual layer loop breaks HuggingFace KV cache / generate() | Medium | Implement cache-compatible version; test generation separately |
| EEG tokens show all-alive or all-dead (no sparsity) | Low | If so, negative result is still interesting and reportable |

---

## 8. Summary of Novelty Claims

1. First application of mid-layer multimodal token injection (DeepInsert) to EEG-to-text.
2. First empirical characterization of EEG temporal token sparsity (alive/dead/sink) using the Fan et al. framework.
3. Evidence on whether the optimal injection depth for EEG differs from vision/audio — this is a concrete comparative finding.
4. (If extension succeeds) First demonstration of alive-token pruning for EEG, with quantified efficiency/performance trade-off.

---

## 9. References

1. **DeepInsert** — mid-layer multimodal token injection. *(Full citation pending — abstract provided by authors.)*

2. Fan, et al. (2026). *What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models*. arXiv:2603.00510.

3. Alayrac, J.-B., Donahue, J., Luc, P., et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS 2022. arXiv:2204.14198.

4. Palazzo, S., Spampinato, C., Kavasidis, I., Giordano, D., Schmidt, J., Shah, M. (2020). *Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features*. IEEE TPAMI. doi:10.1109/TPAMI.2020.2995909.

5. Dey, S., et al. *(Thought2Text original framework — full citation to be confirmed from paper.)*

---

## 10. Files to Modify

| File | Change |
|---|---|
| [channelnet/model.py](../channelnet/model.py) | Add `return_tokens` flag to `ChannelNetModel.forward()` |
| [model.py](../model.py) | Update `mm_proj` dimensions; add `injection_layer` param; implement layer-wise forward |
| [config.py](../config.py) | Add `injection_layer` and `num_eeg_tokens` to `EEGModelForCausalLMConfig` |
| [finetune_llm.py](../finetune_llm.py) | Pass `injection_layer` from args; add to Stage3Trainer |
| [args.py](../args.py) | Add `--injection_layer` and `--num_eeg_tokens` CLI arguments |
