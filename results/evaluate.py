"""
evaluate.py
-----------
Reproduces metrics_based_evaluation_notebook.ipynb as a plain script.
Computes BLEU, ROUGE, METEOR, BERTScore for every CSV in a results folder
and saves a summary to all_results.csv in that folder.

Usage:
    python results/evaluate.py                          # runs on results/ dir
    python results/evaluate.py --results-dir path/to/  # runs on any dir
"""

import argparse
import os
import re
import statistics

import nltk
import numpy as np
import pandas as pd
import bert_score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing prediction CSVs (default: same dir as this script)",
    )
    return parser.parse_args()


RESULTS_DIR = None  # set in main


# ── metric functions (unchanged from notebook) ───────────────────────────────

def compute_bleu(reference, candidate):
    reference = [[ref.split()] for ref in reference]
    candidate = [cand.split() for cand in candidate]
    sf = SmoothingFunction().method4
    scores = [corpus_bleu([r], [c], smoothing_function=sf) for r, c in zip(reference, candidate)]
    return round(np.mean(scores), 3), round(np.std(scores), 3)


def compute_bleu_unigram(reference, candidate):
    reference = [[ref.split()] for ref in reference]
    candidate = [cand.split() for cand in candidate]
    sf = SmoothingFunction().method4
    scores = [corpus_bleu([r], [c], smoothing_function=sf, weights=(1,0,0,0))
              for r, c in zip(reference, candidate)]
    return round(np.mean(scores), 3), round(np.std(scores), 3)


def compute_rouge(reference, candidate):
    rouge = Rouge()
    scores = [rouge.get_scores(c, r, avg=True) for r, c in zip(reference, candidate)]
    r1 = [s["rouge-1"]["f"] for s in scores]
    r2 = [s["rouge-2"]["f"] for s in scores]
    rl = [s["rouge-l"]["f"] for s in scores]
    return (round(np.mean(r1), 3), round(np.mean(r2), 3), round(np.mean(rl), 3),
            round(np.std(r1), 3),  round(np.std(r2), 3),  round(np.std(rl), 3))


def compute_meteor_scores(reference, candidate):
    tok_cands = [word_tokenize(c.replace("<s>", "").replace("</s>", "").strip()) for c in candidate]
    tok_refs  = [word_tokenize(r) for r in reference]
    scores = [single_meteor_score(r, c) for r, c in zip(tok_refs, tok_cands)]
    return round(sum(scores) / len(scores), 3), round(statistics.stdev(scores), 3)


def compute_bert_score(reference, candidate):
    P, R, F1 = bert_score.score(candidate, reference, lang="en", verbose=False)
    return round(F1.mean().item(), 3), round(F1.std().item(), 3)


# ── text cleaning (unchanged from notebook) ──────────────────────────────────

def clean_text(text):
    cleaned = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\[\]{}\-\s]", "", text)
    lines = re.split(r"(?<=[.!?]) +", cleaned)
    seen, unique = set(), []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return " ".join(unique[:1])


def cleanup_pred_captions(predicted_captions):
    predicted_captions = predicted_captions.fillna("")
    result = []
    for caption in predicted_captions:
        if caption.strip():
            cleaned = clean_text(caption)
            result.append(cleaned if cleaned.strip() else "No response.")
        else:
            result.append("No response.")
    return result


# ── per-file evaluation ───────────────────────────────────────────────────────

def run(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(df.columns[0], axis=1, inplace=True)

    references  = df["Expected Caption"].tolist()
    candidates  = cleanup_pred_captions(df["Generated Caption"])
    candidates  = ["No response" if len(c) <= 1 else c for c in candidates]

    results = {}

    mean, std = compute_bleu(references, candidates)
    results["Mean BLEU Score"], results["SD BLEU Score"] = mean, std

    mean, std = compute_bleu_unigram(references, candidates)
    results["Mean BLEU Unigram Score"], results["SD BLEU Unigram Score"] = mean, std

    r1, r2, rl, r1s, r2s, rls = compute_rouge(references, candidates)
    results.update({"Mean ROUGE-1": r1, "SD ROUGE-1": r1s,
                    "Mean ROUGE-2": r2, "SD ROUGE-2": r2s,
                    "Mean ROUGE-l": rl, "SD ROUGE-l": rls})

    mean, std = compute_meteor_scores(references, candidates)
    results["Mean Meteor Score"], results["SD Meteor Score"] = mean, std

    mean, std = compute_bert_score(references, candidates)
    results["Mean BERTScore"], results["SD BERTScore"] = mean, std

    # object accuracy
    acc = (df["Expected object"] == df["Predicted object"]).mean()
    results["Object Accuracy"] = round(acc, 3)

    return results


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    RESULTS_DIR = os.path.abspath(args.results_dir)
    print(f"Evaluating CSVs in: {RESULTS_DIR}")

    all_res = {}
    _skip = {"all_results.csv", "sweep_summary.csv"}
    csv_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv") and f not in _skip)

    for fname in csv_files:
        print(f"\nEvaluating: {fname} ...")
        try:
            results = run(os.path.join(RESULTS_DIR, fname))
            all_res[fname.replace(".csv", "")] = results
            for k, v in results.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  ERROR: {e}")

    out_path = os.path.join(RESULTS_DIR, "all_results.csv")
    pd.DataFrame(all_res).transpose().to_csv(out_path)
    print(f"\nSaved to {out_path}")
