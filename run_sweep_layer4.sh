#!/bin/bash
# Inference-only sweep at layer 4 — extends the existing depth sweep.
# Run AFTER run_deepseek_finetuning.sh has created all_models/deepseek-llm-7b-chat_all/.
# Results: results/depth_sweep_inference_only/layer_4.csv

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
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
python inference.py \
    --model_path               "$MODEL_PATH" \
    --eeg_dataset              "$EEG_DATASET" \
    --image_dir                "$IMAGE_DIR" \
    --splits_path              "$SPLITS_PATH" \
    --injection_layer_override 4 \
    --dest                     "$DEST"

echo "Done → $DEST"

# Print updated sweep summary alongside existing results
python3 - <<'PYEOF'
import pandas as pd, numpy as np, re, os
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
    bleu1 = round(np.mean([corpus_bleu([ri],[ci],smoothing_function=sf,weights=(1,0,0,0)) for ri,ci in zip(r,c)]),3)
    rouge = Rouge()
    sc = [rouge.get_scores(ci,ri,avg=True) for ri,ci in zip(refs,cands)]
    r1 = round(np.mean([s["rouge-1"]["f"] for s in sc]),3)
    tok_c = [word_tokenize(c.replace("<s>","").replace("</s>","").strip()) for c in cands]
    tok_r = [word_tokenize(r) for r in refs]
    meteor = round(sum(single_meteor_score(r,c) for r,c in zip(tok_r,tok_c))/len(tok_r),3)
    return bleu1, r1, meteor

rows = {}
for fname in sorted(os.listdir(SWEEP_DIR)):
    if not fname.endswith(".csv") or fname == "sweep_summary.csv": continue
    df = pd.read_csv(os.path.join(SWEEP_DIR, fname))
    refs, cands = prep(df)
    b1, r1, m = score(refs, cands)
    rows[fname.replace(".csv","")] = {"BLEU-1": b1, "ROUGE-1": r1, "METEOR": m}

summary = pd.DataFrame(rows).T.sort_index()
print("\n=== Updated Depth Sweep (inference-only) ===\n")
print(summary.to_string())
print("\n→ Lowest degradation = best candidate for full retrain")
PYEOF
