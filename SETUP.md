# Next Session Setup

Run these after spinning up a new instance (same region), before training.

## 1. Fix torchvision
```bash
pip install torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

## 2. Re-download CLIP to HF cache
```bash
huggingface-cli download openai/clip-vit-base-patch32
```

## 3. Finish image download (if not complete)
```bash
cd /lambda/nfs/beki-research/Thought2Text-using-Deepseek7b
python download_images.py
# resumes automatically, skips already-downloaded files
# done when: find data/images -type f | wc -l  →  5987
```

## 4. Run experiments in order
```bash
bash run_deepseek_finetuning.sh       # baseline S2+S3 at layer 0  (~few hours)
bash run_sweep_layer4.sh              # inference-only at layer 4   (~10 min)
# pick winner (layer 4 or 7) from sweep summary, then:
bash run_deepinsert_layer7.sh         # DeepInsert full retrain      (~few hours)
```

## Key paths (all on NFS, survive termination)
- EEG data:       data/block/eeg_55_95_std.pth
- Images:         data/images/
- EEG encoder:    eeg_encoder_55-95_40_classes/
- Base LLM:       base_model/deepseek-llm-7b-chat/
- Results:        results/depth_sweep_v2/
