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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_PATH="all_models/deepseek-llm-7b-chat_all"
EEG_DATASET="data/block/eeg_55_95_std.pth"
SPLITS_PATH="data/block/block_splits_by_image_all.pth"
IMAGE_DIR="data/images/"

# N/4-interval depths for 28-layer DeepSeek-7B (layer 0 = current baseline)
DEPTHS=(0 7 14 21)

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

    python inference.py \
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
echo "EVALUATING depth sweep results"
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

SWEEP_DIR = "results/depth_sweep_inference_only"

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

rows = {}
for fname in sorted(os.listdir(SWEEP_DIR)):
    if not fname.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(SWEEP_DIR, fname))
    refs, cands = prep(df)
    b, b1, r1, rl, m = score(refs, cands)
    acc = round((df["Expected object"] == df["Predicted object"]).mean(), 3)
    label = fname.replace(".csv","")
    rows[label] = {"BLEU": b, "BLEU-1": b1, "ROUGE-1": r1, "ROUGE-L": rl, "METEOR": m, "ObjAcc": acc}

summary = pd.DataFrame(rows).T.sort_index()
out = os.path.join(SWEEP_DIR, "sweep_summary.csv")
summary.to_csv(out)

print("\n=== Depth Sweep Results (inference-only, no retraining) ===\n")
print(summary.to_string())
print(f"\nSaved to {out}")
PYEOF
