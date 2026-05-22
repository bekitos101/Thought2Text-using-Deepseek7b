#!/bin/bash
# Inference-only sweep at layer 4 — extends the existing depth sweep.
# Run AFTER run_deepseek_finetuning.sh has created all_models/deepseek-llm-7b-chat_all/.
# Results: results/depth_sweep_inference_only/layer_4.csv

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_TF=0
# TRANSFORMERS_OFFLINE kept for inference (base LLM is local), but NOT set globally
# so evaluate.py can reach the cached roberta-large for BERTScore.
export HF_DATASETS_OFFLINE=1

VENV_DIR="$(cd "$(dirname "$0")" && pwd)/venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

MODEL_PATH="all_models/deepseek-llm-7b-chat_all"
EEG_DATASET="data/block/eeg_55_95_std.pth"
SPLITS_PATH="data/block/block_splits_by_image_all.pth"
IMAGE_DIR="data/images/"
DEST="results/depth_sweep_inference_only/layer_4.csv"

if [ -f "$DEST" ]; then
    echo "SKIP: $DEST already exists."
    exit 0
fi

echo "Running inference-only sweep at injection_layer=4 ..."
TRANSFORMERS_OFFLINE=1 python inference.py \
    --model_path               "$MODEL_PATH" \
    --eeg_dataset              "$EEG_DATASET" \
    --image_dir                "$IMAGE_DIR" \
    --splits_path              "$SPLITS_PATH" \
    --injection_layer_override 4 \
    --dest                     "$DEST"

echo "Done → $DEST"

# Run full evaluation (BLEU, ROUGE-1/2/L, METEOR, BERTScore) on all sweep CSVs.
# roberta-large must be in the HF cache (run once with internet: venv/bin/python -c
# "from transformers import AutoModel; AutoModel.from_pretrained('roberta-large')")
echo "Running full evaluation on sweep results..."
python results/evaluate.py --results-dir results/depth_sweep_inference_only/

echo ""
echo "=== Sweep summary (BLEU | ROUGE-1 | METEOR | BERTScore) ==="
python3 - <<'PYEOF'
import pandas as pd

df = pd.read_csv("results/depth_sweep_inference_only/all_results.csv", index_col=0)
cols = ["Mean BLEU Score", "Mean ROUGE-1", "Mean ROUGE-2", "Mean ROUGE-l",
        "Mean Meteor Score", "Mean BERTScore", "Object Accuracy"]
available = [c for c in cols if c in df.columns]
print(df[available].sort_index().to_string())
print("\n→ Lowest degradation from layer_0 = best candidate for full retrain")
PYEOF
