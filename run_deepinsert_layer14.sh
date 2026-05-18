#!/bin/bash
# DeepInsert depth experiment: Stage 2 + Stage 3 at injection_layer=14 (N/2).
# Reuses Stage 2 checkpoint from run_deepseek_finetuning.sh if available.
# Results: results/depth_sweep_v2/deepinsert_layer_14.csv

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

VENV_DIR="$(cd "$(dirname "$0")" && pwd)/venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

LLM="all_models/deepseek-llm-7b-chat_all/llm"
EEG_DATASET="data/block/eeg_55_95_std.pth"
SPLITS_PATH="data/block/block_splits_by_image_all.pth"
EEG_ENCODER_PATH="./eeg_encoder_55-95_40_classes"
IMAGE_DIR="data/images/"

OUTPUT_DIR="all_models/deepinsert_layer_14"
RESULTS_CSV="results/depth_sweep_v2/deepinsert_layer_14.csv"

mkdir -p results/depth_sweep_v2

echo "Disk before training:"
df -h / | tail -1

echo ""
echo "============================================================"
echo "DeepInsert  Stage 2 + Stage 3  injection_layer=14"
echo "============================================================"

if [ -f "$RESULTS_CSV" ]; then
    echo "SKIP: $RESULTS_CSV already exists."
else
    if [ ! -d "$OUTPUT_DIR/llm" ]; then
        echo "--- Training ---"
        python finetune_llm.py \
            --eeg_dataset           "$EEG_DATASET" \
            --splits_path           "$SPLITS_PATH" \
            --eeg_encoder_path      "$EEG_ENCODER_PATH" \
            --image_dir             "$IMAGE_DIR" \
            --output                "$OUTPUT_DIR" \
            --llm_backbone_name_or_path "$LLM" \
            --saved_pretrained_model_path all_models \
            --load_in_8bit \
            --batch_size            4 \
            --injection_layer       14

        echo "Training done. Disk:"
        df -h / | tail -1
    else
        echo "RESUME: checkpoint found, skipping training, running inference only."
    fi

    echo "--- Inference ---"
    python inference.py \
        --model_path    "$OUTPUT_DIR" \
        --eeg_dataset   "$EEG_DATASET" \
        --image_dir     "$IMAGE_DIR" \
        --splits_path   "$SPLITS_PATH" \
        --dest          "$RESULTS_CSV"

    echo "Inference done. Freeing checkpoint..."
    rm -rf "$OUTPUT_DIR/llm"
    df -h / | tail -1
    echo "Done → $RESULTS_CSV"
fi
