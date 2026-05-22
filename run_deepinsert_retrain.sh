#!/bin/bash
# Phase 2 DeepInsert retrain — true two-stage bypass at injection_layer=21.
#
# Architecture: text-only through layers 0..20, EEG physically inserted at
# layer 21, full combined sequence through layers 21..29.  Only mm_proj trains.
#
# Stage 2 is SKIPPED by detecting all_models/deepseek-llm-7b-chat_all already
# exists (saved_pretrained_model_path trick in finetune_llm.py), so mm_proj
# is warm-started from the existing trained checkpoint.
#
# Output: all_models/deepinsert_layer21_retrain/
# Log:    /tmp/deepinsert_retrain.log

set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

VENV_DIR="$(cd "$(dirname "$0")" && pwd)/venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$(dirname "$0")"

LLM="all_models/deepseek-llm-7b-chat_all/llm"
EEG_DATASET="data/block/eeg_55_95_std.pth"
SPLITS_PATH="data/block/block_splits_by_image_all.pth"
EEG_ENCODER_PATH="./eeg_encoder_55-95_40_classes"
IMAGE_DIR="data/images/"
OUTPUT_DIR="all_models/deepinsert_layer21_retrain"

echo "============================================================"
echo "DeepInsert Phase 2 retrain  injection_layer=21"
echo "Output: $OUTPUT_DIR"
echo "Log:    /tmp/deepinsert_retrain.log"
echo "============================================================"

python finetune_llm.py \
    --eeg_dataset               "$EEG_DATASET" \
    --splits_path               "$SPLITS_PATH" \
    --eeg_encoder_path          "$EEG_ENCODER_PATH" \
    --image_dir                 "$IMAGE_DIR" \
    --output                    "$OUTPUT_DIR" \
    --llm_backbone_name_or_path "$LLM" \
    --saved_pretrained_model_path all_models \
    --no_stage2 \
    --load_in_8bit \
    --batch_size                4 \
    --gradient_accumulation_steps 4 \
    --num_epochs_eeg            5 \
    --learning_rate             2e-5 \
    --warmup_ratio              0.03 \
    --injection_layer           21 \
    2>&1 | tee /tmp/deepinsert_retrain.log

echo "Training complete. Model saved to $OUTPUT_DIR"
echo "Run inference with:"
echo "  python inference.py --model_path $OUTPUT_DIR --eeg_dataset $EEG_DATASET --image_dir $IMAGE_DIR --splits_path $SPLITS_PATH --dest results/deepinsert_layer21_retrain.csv"
