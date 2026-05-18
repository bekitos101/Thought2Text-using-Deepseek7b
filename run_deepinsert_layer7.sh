#!/bin/bash
# Phase 2 — DeepInsert-style depth experiment: Stage 2 + Stage 3 at injection_layer=7.
#
# Identical pipeline to run_deepseek_finetuning.sh (Stage 2 alignment preserved),
# with one change: EEG token bypasses the first 7 layers and joins at layer 7.
# Motivated by DeepInsert (Choraria et al., EACL 2026): cross-modal token interactions
# are concentrated in deeper layers; early layers are redundant for EEG-text fusion.
#
# Phase 1 depth sweep (inference-only heuristic) identified layer 7 as the depth
# that degrades least without retraining → primary candidate for full retrain.
#
# Usage:
#   bash run_deepinsert_layer7.sh
#
# Disk: ~7.3GB for checkpoint, deleted after inference to free space.
# Results: results/depth_sweep_v2/deepinsert_layer_7.csv

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# activate project virtualenv
VENV_DIR="$(cd "$(dirname "$0")" && pwd)/venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

LLM="all_models/deepseek-llm-7b-chat_all/llm"
EEG_DATASET="data/block/eeg_55_95_std.pth"
SPLITS_PATH="data/block/block_splits_by_image_all.pth"
EEG_ENCODER_PATH="./eeg_encoder_55-95_40_classes"
IMAGE_DIR="data/images/"

OUTPUT_DIR="all_models/deepinsert_layer_7"
RESULTS_CSV="results/depth_sweep_v2/deepinsert_layer_7.csv"

mkdir -p results/depth_sweep_v2

echo "Disk before training:"
df -h / | tail -1

# ── Stage 2 + Stage 3 ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "DeepInsert  Stage 2 + Stage 3  injection_layer=7"
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
            --injection_layer       7

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

# ── Evaluate and compare against baseline + depth sweep ──────────────────────
echo ""
echo "============================================================"
echo "COMPARING results"
echo "============================================================"
python3 - <<'PYEOF'
import pandas as pd, numpy as np, re, os, statistics
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

def clean_text(text):
    cleaned = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\[\]{}\-\s]", "", text)
    lines = re.split(r"(?<=[.!?]) +", cleaned)
    seen, unique = set(), []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line not in seen:
            seen.add(line); unique.append(line)
    return " ".join(unique[:1])

def prep(df):
    refs = df["Expected Caption"].tolist()
    cands = []
    for c in df["Generated Caption"].fillna("").tolist():
        cl = clean_text(c) if c.strip() else ""
        cands.append(cl if cl.strip() else "No response.")
    return refs, ["No response" if len(c) <= 1 else c for c in cands]

def score(refs, cands):
    sf = SmoothingFunction().method4
    r = [[x.split()] for x in refs]; c = [x.split() for x in cands]
    bleu  = round(np.mean([corpus_bleu([ri],[ci],smoothing_function=sf) for ri,ci in zip(r,c)]),3)
    bleu1 = round(np.mean([corpus_bleu([ri],[ci],smoothing_function=sf,weights=(1,0,0,0)) for ri,ci in zip(r,c)]),3)
    rouge = Rouge()
    sc = [rouge.get_scores(ci,ri,avg=True) for ri,ci in zip(refs,cands)]
    r1   = round(np.mean([s["rouge-1"]["f"] for s in sc]),3)
    rl   = round(np.mean([s["rouge-l"]["f"] for s in sc]),3)
    tok_c = [word_tokenize(c.replace("<s>","").replace("</s>","").strip()) for c in cands]
    tok_r = [word_tokenize(r) for r in refs]
    meteor = round(sum(single_meteor_score(r,c) for r,c in zip(tok_r,tok_c))/len(tok_r),3)
    return bleu, bleu1, r1, rl, meteor

results = {}

# original trained model (layer 0)
orig = pd.read_csv("results/baseline/all_results.csv", index_col=0).loc["results_deepseek-llm-7b-chat_all"]
results["Original S2+S3 layer_0"] = {
    "BLEU": orig["Mean BLEU Score"], "BLEU-1": orig["Mean BLEU Unigram Score"],
    "ROUGE-1": orig["Mean ROUGE-1"], "ROUGE-L": orig["Mean ROUGE-l"],
    "METEOR": orig["Mean Meteor Score"], "ObjAcc": orig["Object Accuracy"]
}

# depth sweep (inference only)
sweep_dir = "results/depth_sweep_inference_only"
for fname in sorted(os.listdir(sweep_dir)):
    if not fname.endswith(".csv") or fname == "sweep_summary.csv": continue
    df = pd.read_csv(os.path.join(sweep_dir, fname))
    refs, cands = prep(df)
    b, b1, r1, rl, m = score(refs, cands)
    acc = round((df["Expected object"] == df["Predicted object"]).mean(), 3)
    label = f"Sweep {fname.replace('.csv','')}"
    results[label] = {"BLEU": b, "BLEU-1": b1, "ROUGE-1": r1, "ROUGE-L": rl, "METEOR": m, "ObjAcc": acc}

# Phase 2 result
v2_path = "results/depth_sweep_v2/deepinsert_layer_7.csv"
if os.path.exists(v2_path):
    df = pd.read_csv(v2_path)
    refs, cands = prep(df)
    b, b1, r1, rl, m = score(refs, cands)
    acc = round((df["Expected object"] == df["Predicted object"]).mean(), 3)
    results["DeepInsert S2+S3 layer_7"] = {"BLEU": b, "BLEU-1": b1, "ROUGE-1": r1, "ROUGE-L": rl, "METEOR": m, "ObjAcc": acc}

df_out = pd.DataFrame(results).T
print("\n=== Full Comparison ===\n")
print(df_out.to_string())
df_out.to_csv("results/depth_sweep_v2/full_comparison.csv")
print("\nSaved to results/depth_sweep_v2/full_comparison.csv")
PYEOF
