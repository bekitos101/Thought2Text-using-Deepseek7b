#!/bin/bash
# Injection-depth ablation sweep for mid-layer EEG token injection.
#
# For each depth in INJECTION_LAYERS, runs:
#   1. Stage-3 fine-tuning  (--no_stage2, tokens injected at layer L)
#   2. Inference            (results saved to results/ablation_layer_L.csv)
#   3. Evaluation           (all CSVs evaluated together at the end)
#
# A baseline run (original single pooled token, injection_layer=0) is
# included first so all results land in results/all_results.csv side-by-side.
#
# Usage:
#   bash run_ablation.sh
#
# Adjust the variables below to match your paths before running.

set -e   # stop on first error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── paths ─────────────────────────────────────────────────────────────────────
LLM="deepseek-ai/deepseek-llm-7b-chat"
EEG_DATASET="data/block/eeg_55_95_std.pth"
SPLITS_PATH="data/block/block_splits_by_image_all.pth"
EEG_ENCODER_PATH="./eeg_encoder_55-95_40_classes"
IMAGE_DIR="data/images/"
SAVED_PRETRAINED="all_models"

# ── ablation config ───────────────────────────────────────────────────────────
# N/4-interval depths for a 28-layer DeepSeek-7B
INJECTION_LAYERS=(0 7 14 21)

mkdir -p results

# ── helpers ───────────────────────────────────────────────────────────────────
run_finetune() {
    local layer=$1
    local token_inject_flag=$2   # "--token_inject" or ""
    local output_dir=$3

    python finetune_llm.py \
        --eeg_dataset       "$EEG_DATASET" \
        --splits_path       "$SPLITS_PATH" \
        --eeg_encoder_path  "$EEG_ENCODER_PATH" \
        --image_dir         "$IMAGE_DIR" \
        --output            "$output_dir" \
        --llm_backbone_name_or_path "$LLM" \
        --saved_pretrained_model_path "$SAVED_PRETRAINED" \
        --no_stage2 \
        --load_in_8bit \
        --batch_size        4 \
        --injection_layer   "$layer" \
        $token_inject_flag
}

run_inference() {
    local model_path=$1
    local dest_csv=$2

    python inference.py \
        --model_path  "$model_path" \
        --eeg_dataset "$EEG_DATASET" \
        --image_dir   "$IMAGE_DIR" \
        --splits_path "$SPLITS_PATH" \
        --dest        "$dest_csv"
}

# ── baseline: original single pooled token, layer 0 ─────────────────────────
echo "================================================================"
echo "BASELINE  token_inject=False  injection_layer=0"
echo "================================================================"

BASELINE_DIR="all_models/ablation_baseline"
BASELINE_CSV="results/ablation_baseline.csv"

run_finetune 0 "" "$BASELINE_DIR"
run_inference "$BASELINE_DIR" "$BASELINE_CSV"
echo "Baseline done → $BASELINE_CSV"

# ── ablation sweep ────────────────────────────────────────────────────────────
for L in "${INJECTION_LAYERS[@]}"; do
    echo ""
    echo "================================================================"
    echo "ABLATION  token_inject=True  injection_layer=${L}"
    echo "================================================================"

    MODEL_DIR="all_models/ablation_layer_${L}"
    RESULTS_CSV="results/ablation_layer_${L}.csv"

    run_finetune "$L" "--token_inject" "$MODEL_DIR"
    run_inference "$MODEL_DIR" "$RESULTS_CSV"
    echo "Layer ${L} done → $RESULTS_CSV"
done

# ── evaluate all CSVs together ────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "EVALUATING all results"
echo "================================================================"
python results/evaluate.py

# ── print summary table ───────────────────────────────────────────────────────
echo ""
python3 - <<'PYEOF'
import pandas as pd, os

summary_path = "results/all_results.csv"
if not os.path.exists(summary_path):
    print("No all_results.csv found — evaluation may have failed.")
    exit(1)

df = pd.read_csv(summary_path, index_col=0)

# keep the columns most relevant for this ablation
cols = ["Mean BLEU Score", "Mean BLEU Unigram Score",
        "Mean ROUGE-1", "Mean ROUGE-l", "Mean Meteor Score", "Object Accuracy"]
cols = [c for c in cols if c in df.columns]

# sort rows so ablation_layer_* appear in depth order, baseline first
order = ["ablation_baseline"] + [f"ablation_layer_{l}" for l in [0, 7, 14, 21]]
present = [r for r in order if r in df.index] + \
          [r for r in df.index if r not in order]
df = df.loc[present, cols]

print("\n=== Injection-Depth Ablation Results ===\n")
print(df.to_string())
print()
PYEOF
