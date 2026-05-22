#!/bin/bash
# Phase 1 — Inference-only depth sweep (DeepInsert-style, zero training cost).
#
# Takes the existing Stage-2+3 trained model and runs inference at different
# injection depths WITHOUT retraining. Follows DeepInsert Appendix C heuristic:
# test layer candidates on the baseline model to identify the best depth before
# committing to a full retrain.
#
# Usage:
#   bash run_depth_sweep.sh
#
# Results land in results/depth_sweep_inference_only/
# Requires ~400KB per CSV, no new model checkpoints written.

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_TF=0
export HF_DATASETS_OFFLINE=1
# TRANSFORMERS_OFFLINE set only for inference; evaluate.py needs HF cache for roberta-large

MODEL_PATH="all_models/deepseek-llm-7b-chat_all"
EEG_DATASET="data/block/eeg_55_95_std.pth"
SPLITS_PATH="data/block/block_splits_by_image_all.pth"
IMAGE_DIR="data/images/"

# Sweep depths for 30-layer DeepSeek-7B.
# Layer 0 = baseline. Layer 4 ≈ paper recommendation. 7/14/21 = N/4 intervals.
DEPTHS=(0 4 7 14 21)

mkdir -p results/depth_sweep_inference_only

echo "Disk before sweep:"
df -h / | tail -1

for L in "${DEPTHS[@]}"; do
    DEST="results/depth_sweep_inference_only/layer_${L}.csv"

    if [ -f "$DEST" ]; then
        echo "SKIP: $DEST already exists."
        continue
    fi

    echo ""
    echo "============================================================"
    echo "DEPTH SWEEP  injection_layer=${L}  (no retraining)"
    echo "============================================================"

    TRANSFORMERS_OFFLINE=1 python inference.py \
        --model_path        "$MODEL_PATH" \
        --eeg_dataset       "$EEG_DATASET" \
        --image_dir         "$IMAGE_DIR" \
        --splits_path       "$SPLITS_PATH" \
        --injection_layer_override "$L" \
        --dest              "$DEST"

    echo "Layer ${L} done → $DEST"
    df -h / | tail -1
done

echo ""
echo "============================================================"
echo "EVALUATING depth sweep results (BLEU, ROUGE, METEOR, BERTScore)"
echo "============================================================"
python results/evaluate.py --results-dir results/depth_sweep_inference_only/

echo ""
echo "=== Depth Sweep Summary ==="
python3 - <<'PYEOF'
import pandas as pd
df = pd.read_csv("results/depth_sweep_inference_only/all_results.csv", index_col=0)
cols = ["Mean BLEU Score", "Mean ROUGE-1", "Mean ROUGE-2", "Mean ROUGE-l",
        "Mean Meteor Score", "Mean BERTScore", "Object Accuracy"]
available = [c for c in cols if c in df.columns]
print(df[available].sort_index().to_string())
print("\n→ Lowest degradation from layer_0 = best candidate for full retrain")
PYEOF
