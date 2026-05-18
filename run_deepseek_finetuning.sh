#!/bin/bash
# Fine-tune with DeepSeek-LLM-7B-Chat backbone (Stages 2 + 3).
# Usage:
#   bash run_deepseek_finetuning.sh
#
# Mirrors run_all_fine_tuning.sh but targets the DeepSeek-7B model.
# The model_utils.py layer handles chat-template and LoRA target differences
# automatically — no other code changes are required.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LLM="base_model/deepseek-llm-7b-chat"
LLM_NAME=$(echo "$LLM" | awk -F '/' '{print $2}')

OUTPUT_DIR="all_models/${LLM_NAME}_all"
OUTPUT_DIR_NO_STAGE2="all_models/${LLM_NAME}_no_stage2_all"

echo "=== DeepSeek fine-tuning: Stage 2 + Stage 3 ==="
# --- original (bf16 for Ampere+ GPUs, batch_size 16) ---
# python finetune_llm.py \
#     --eeg_dataset data/block/eeg_55_95_std.pth \
#     --splits_path data/block/block_splits_by_image_all.pth \
#     --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
#     --image_dir data/images/ \
#     --output "$OUTPUT_DIR" \
#     --llm_backbone_name_or_path "$LLM" \
#     --load_in_8bit \
#     --bf16
# --- updated: fp16 (Pascal/Titan Xp), batch_size 4 for 12.8GB VRAM ---
python finetune_llm.py \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --splits_path data/block/block_splits_by_image_all.pth \
    --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
    --image_dir data/images/ \
    --output "$OUTPUT_DIR" \
    --llm_backbone_name_or_path "$LLM" \
    --saved_pretrained_model_path all_models \
    --load_in_8bit \
    --batch_size 4

if [ $? -ne 0 ]; then
    echo "Error during Stage 2+3 fine-tuning."
    exit 1
fi

echo "=== DeepSeek fine-tuning: Stage 3 only (no_stage2 ablation) ==="
# --- original ---
# python finetune_llm.py \
#     --eeg_dataset data/block/eeg_55_95_std.pth \
#     --splits_path data/block/block_splits_by_image_all.pth \
#     --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
#     --image_dir data/images/ \
#     --output "$OUTPUT_DIR_NO_STAGE2" \
#     --llm_backbone_name_or_path "$LLM" \
#     --no_stage2 \
#     --load_in_8bit \
#     --bf16
# --- updated ---
python finetune_llm.py \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --splits_path data/block/block_splits_by_image_all.pth \
    --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
    --image_dir data/images/ \
    --output "$OUTPUT_DIR_NO_STAGE2" \
    --llm_backbone_name_or_path "$LLM" \
    --saved_pretrained_model_path all_models \
    --no_stage2 \
    --load_in_8bit \
    --fp16 \
    --batch_size 4

if [ $? -ne 0 ]; then
    echo "Error during Stage 3-only fine-tuning."
    exit 1
fi

echo "=== Done. Models saved to: $OUTPUT_DIR and $OUTPUT_DIR_NO_STAGE2 ==="
