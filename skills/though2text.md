---
name: project-thought2text-deepinsert
description: Research project reworking Thought2Text with DeepSeek-7B and DeepInsert mid-layer injection experiments — full session context
metadata: 
  node_type: memory
  type: project
  originSessionId: 6f083038-7ea1-4631-be18-42f2e6cb9bfe
---

Extending Thought2Text EEG-to-text framework with DeepSeek-7B backbone and DeepInsert (EACL 2026) mid-layer multimodal token injection.

**Why:** Testing whether injecting EEG tokens at an intermediate transformer layer (instead of layer 0) improves EEG-to-text generation quality and reduces compute cost on DeepSeek-7B (28 layers).

**Repo:** `/lambda/nfs/beki-research/Thought2Text-using-Deepseek7b/`

---

## Architecture Understanding

**Original Thought2Text (3 stages):**
1. Train ChannelNet EEG encoder aligned to CLIP image embeddings
2. Train mm_proj (linear projector) on image-caption pairs — LLM frozen
3. Fine-tune mm_proj on EEG embeddings — LLM still frozen

**Key insight:** The LLM and EEG encoder are BOTH frozen throughout stages 2 and 3. Only `mm_proj` (a single `nn.Linear`) actually trains. This means `all_models/deepseek-llm-7b-chat_all/llm/` is just a copy of the original base model.

**DeepInsert mechanism:**
- Baseline (layer 0): `[text + EEG tokens] → Layer 0 → ... → Layer N-1`
- DeepInsert (layer L): `[text] → Layer 0 → ... → Layer L-1`, then merge with EEG tokens `→ Layer L → ... → Layer N-1`
- EEG tokens bypass early layers entirely via a forward pre-hook (`_make_injection_hook`)
- Paper recommendation: layer 4 is consistently optimal for 32-layer models (≈12.5% through stack)
- For DeepSeek-7B (28 layers): layer 4 ≈ 14% through

**DeepInsert sweep strategy (from paper Appendix C):**
- Phase 1 (cheap): Run inference at multiple depths on ONE trained baseline model using `--injection_layer_override` — no retraining
- Phase 2 (expensive): Retrain only at the best candidate layer(s)
- We do NOT retrain at every depth

---

## What Has Been Done

| Experiment | Status | Location |
|---|---|---|
| DeepSeek-7B integration | ✅ Done | `model.py`, `model_utils.py`, `config.py` |
| DeepInsert injection hook | ✅ Done | `model.py:_make_injection_hook()` |
| Token inject mode (temporal B,10,50) | ✅ Done | `channelnet/model.py:return_tokens` |
| Stage 1 EEG encoder training | ✅ Done | `eeg_encoder_55-95_40_classes/` |
| Baseline DeepSeek Stage 2+3 (layer 0) | ✅ Training overnight (2026-05-19) | `all_models/deepseek-llm-7b-chat_all/` |
| Phase 1 inference-only sweep (layers 0,7,14,21) | ✅ Done | `results/depth_sweep_inference_only/sweep_summary.csv` |
| Ablation no_stage2 (layers 0, 7) | ✅ Done | `results/ablation_v1_no_stage2/` |
| Image data download (all 40 classes) | ✅ Done (5988 files) | `data/images/` |

---

## Existing Results

**Phase 1 sweep (inference-only, baseline model):**
- layer_0: BLEU=0.054, ROUGE-1=0.297, METEOR=0.257 (best — model was trained at layer 0)
- layer_7: BLEU=0.032, ROUGE-1=0.236, METEOR=0.207 (best among non-zero layers)
- layer_14: BLEU=0.029, ROUGE-1=0.217
- layer_21: BLEU=0.030, ROUGE-1=0.228
- ObjAcc=0.531 constant (classification, not affected by injection depth)

**Baseline DeepSeek:** BLEU=0.054, ROUGE-1=0.297, METEOR=0.257, BERTScore=0.891

---

## Next Steps (in order)

1. **`bash run_sweep_layer4.sh`** — inference-only at layer 4 on the rebuilt baseline; extends sweep to include paper's recommended depth (requires `all_models/deepseek-llm-7b-chat_all/` which is being rebuilt overnight)
2. **Compare** layer 4 vs layer 7 from sweep summary → pick winner
3. **`bash run_deepinsert_layer7.sh`** (and/or layer 4 if it wins) — full retrain Stage 3 at best depth → saves to `results/depth_sweep_v2/`
4. Final comparison table: baseline (layer 0) vs DeepInsert winner

---

## Environment / Infrastructure

- Server: Lambda instance, Ubuntu, CUDA 12.8, NVIDIA A10 (23GB VRAM)
- Python: 3.10
- Key package versions: torch==2.4.1, transformers==4.44.0, peft==0.4.0, bitsandbytes==0.43.0, numpy==2.2.6, pandas>=2.0
- **`USE_TF=0` must be exported** before any training/inference script — tensorflow is installed (pulled by tf-keras) and conflicts with transformers via `numpy.core.umath`
- All scripts already have `export USE_TF=0` added
- Base LLM: `base_model/deepseek-llm-7b-chat/` (local, offline)
- NFS path: `/lambda/nfs/beki-research/` — survives instance restarts

## Dependency Issues Resolved

- `pandas==1.5.3` compiled against numpy 1.x → incompatible with numpy 2.x (needed by torch 2.4.1) → fixed to `pandas>=2.0` in requirements.txt
- `numpy==1.24.4` pin removed from requirements.txt (was wrong for torch 2.4.1)
- `jinja2>=3.1.0` required by transformers apply_chat_template
- `torchvision==0.19.1` required for torch 2.4.1 (from pytorch.org/whl/cu121)
- `USE_TF=0` suppresses tensorflow import in transformers (avoids numpy.core.umath error)

## Image Download

- `download_missing_images.py` patched to enumerate class folders directly from parent Drive folder via API (no log file needed)
- Parent Drive folder: `1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu`
- Drills into `images/` subfolder → 40 class folders → downloads caption, sketch, JPEG files
- Target: 5987 files across 40 classes (achieved 5988)

## Key File Locations

- EEG data: `data/block/eeg_55_95_std.pth`
- EEG encoder (Stage 1): `eeg_encoder_55-95_40_classes/`
- Base LLM: `base_model/deepseek-llm-7b-chat/`
- Trained baseline (Stage 2+3): `all_models/deepseek-llm-7b-chat_all/` (rebuilding overnight)
- Results: `results/depth_sweep_inference_only/`, `results/ablation_v1_no_stage2/`, `results/depth_sweep_v2/` (pending)
- Run scripts: `run_sweep_layer4.sh`, `run_deepinsert_layer7.sh`, `run_deepinsert_layer4.sh` (if needed)
