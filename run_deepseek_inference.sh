#!/bin/bash
# Run inference with the fine-tuned DeepSeek-LLM-7B-Chat model.
# Usage:
#   bash run_deepseek_inference.sh
#
# Mirrors run_all_inference.sh but targets the DeepSeek-7B checkpoints
# produced by run_deepseek_finetuning.sh.

LLM="deepseek-ai/deepseek-llm-7b-chat"
LLM_NAME=$(echo "$LLM" | awk -F '/' '{print $2}')

MODEL_PATH="all_models/${LLM_NAME}_all"
MODEL_PATH_NO_STAGE2="all_models/${LLM_NAME}_no_stage2_all"

mkdir -p results

RESULTS_CSV="results/results_${LLM_NAME}_all.csv"
RESULTS_CSV_NO_STAGE2="results/results_${LLM_NAME}_no_stage2_all.csv"

echo "=== DeepSeek inference: Stage 2+3 model ==="
python inference.py \
    --model_path "$MODEL_PATH" \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --image_dir data/images/ \
    --dest "$RESULTS_CSV" \
    --splits_path data/block/block_splits_by_image_all.pth

if [ $? -ne 0 ]; then
    echo "Error during Stage 2+3 inference."
    exit 1
fi

echo "=== DeepSeek inference: Stage 3-only model ==="
python inference.py \
    --model_path "$MODEL_PATH_NO_STAGE2" \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --image_dir data/images/ \
    --dest "$RESULTS_CSV_NO_STAGE2" \
    --splits_path data/block/block_splits_by_image_all.pth

if [ $? -ne 0 ]; then
    echo "Error during Stage 3-only inference."
    exit 1
fi

echo "=== Done. Results saved to: $RESULTS_CSV and $RESULTS_CSV_NO_STAGE2 ==="
