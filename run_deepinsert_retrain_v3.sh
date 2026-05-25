#!/bin/bash
# DeepInsert retrain v3 — full two-stage training at injection_layer=21.
#
# Fix over v1/v2: Stage 2 (CLIP image alignment) now runs with injection_layer=21
# so mm_proj learns to map image embeddings into layer-21 hidden-state space,
# not token-embedding space. Stage 3 then fine-tunes on EEG at the same layer.
#
# Stage 2 output:  all_models/deepinsert_stage2_layer21/deepseek-llm-7b-chat_all/
# Stage 3 output:  all_models/deepinsert_layer21_v3/
# Log:             /tmp/deepinsert_retrain_v3.log

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
CLIP_MODEL="all_models/clip-vit-base-patch32"
EEG_DATASET="data/block/eeg_55_95_std.pth"
SPLITS_PATH="data/block/block_splits_by_image_all.pth"
EEG_ENCODER_PATH="./eeg_encoder_55-95_40_classes"
IMAGE_DIR="data/images/"
OUTPUT_DIR="all_models/deepinsert_layer21_v3"
STAGE2_CACHE="all_models/deepinsert_stage2_layer21"

echo "============================================================"
echo "DeepInsert v3 retrain  injection_layer=21  (full stage 2+3)"
echo "Stage 2 cache: $STAGE2_CACHE"
echo "Output:        $OUTPUT_DIR"
echo "Log:           /tmp/deepinsert_retrain_v3.log"
echo "============================================================"

python finetune_llm.py \
    --eeg_dataset               "$EEG_DATASET" \
    --splits_path               "$SPLITS_PATH" \
    --eeg_encoder_path          "$EEG_ENCODER_PATH" \
    --image_dir                 "$IMAGE_DIR" \
    --output                    "$OUTPUT_DIR" \
    --llm_backbone_name_or_path "$LLM" \
    --clip_model               "$CLIP_MODEL" \
    --saved_pretrained_model_path "$STAGE2_CACHE" \
    --load_in_8bit \
    --batch_size                4 \
    --gradient_accumulation_steps 4 \
    --num_epochs_image          5 \
    --num_epochs_eeg            10 \
    --learning_rate             2e-5 \
    --warmup_ratio              0.1 \
    --injection_layer           21 \
    2>&1 | tee /tmp/deepinsert_retrain_v3.log

echo "Training complete. Model saved to $OUTPUT_DIR"
echo "Run inference with:"
echo "  python inference.py --model_path $OUTPUT_DIR --eeg_dataset $EEG_DATASET --image_dir $IMAGE_DIR --splits_path $SPLITS_PATH --dest results/deepinsert_layer21_v3.csv"
