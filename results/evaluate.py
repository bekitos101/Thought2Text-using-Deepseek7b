"""
evaluate.py
-----------
Computes BLEU, ROUGE, METEOR, BERTScore, CLIPScore for every CSV in a results
folder and saves a summary to all_results.csv in that folder.

CLIPScore compares generated captions against the actual images (reference-free),
which is more appropriate for EEG-to-text where exact phrasing cannot be recovered.
Requires "Ground Truth Image" column and a local CLIP model directory.

Usage:
    python results/evaluate.py                              # runs on results/ dir
    python results/evaluate.py --results-dir path/to/      # runs on any dir
    python results/evaluate.py --clip-model path/to/clip   # custom CLIP path
    python results/evaluate.py --no-clip                   # skip CLIPScore
"""

import argparse
import os
import re
import statistics

import nltk
import numpy as np
import pandas as pd
import bert_score
import torch
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
    parser.add_argument(
        "--clip-model",
        default=None,
        help="Path to local CLIP model dir (auto-detected if not set)",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Skip CLIPScore computation",
    )
    return parser.parse_args()


RESULTS_DIR = None  # set in main


# ── metric functions ──────────────────────────────────────────────────────────

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


def compute_clip_score(image_paths, candidates, clip_model, clip_processor,
                       project_root, batch_size=32):
    """Reference-free CLIPScore: cosine similarity between image and generated caption.

    Scores in [0, 1]. Returns (mean, std) or (None, None) if images unavailable.
    """
    from PIL import Image

    scores = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_caps  = candidates[start:start + batch_size]

        images, texts = [], []
        for img_path, cap in zip(batch_paths, batch_caps):
            if not cap or cap in ("No response", "No response.") or len(cap.strip()) <= 1:
                continue
            full = os.path.join(project_root, img_path) if project_root else img_path
            if not os.path.exists(full):
                continue
            try:
                images.append(Image.open(full).convert("RGB"))
                texts.append(cap)
            except Exception:
                continue

        if not images:
            continue

        device = next(clip_model.parameters()).device
        inputs = clip_processor(
            text=texts, images=images,
            return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = clip_model(**inputs)
            img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            scores.extend((img_emb * txt_emb).sum(dim=-1).clamp(min=0).tolist())

    if not scores:
        return None, None
    return round(float(np.mean(scores)), 3), round(float(np.std(scores)), 3)


# ── helpers ───────────────────────────────────────────────────────────────────

def find_project_root(results_dir):
    """Walk up from results_dir until data/images/ is found."""
    d = os.path.abspath(results_dir)
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "data", "images")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def find_clip_model(results_dir):
    """Look for the locally-saved CLIP model relative to the project root."""
    root = find_project_root(results_dir)
    if root is None:
        return None
    candidate = os.path.join(root, "all_models", "clip-vit-base-patch32")
    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
        return candidate
    return None


def load_clip(clip_model_path):
    from transformers import CLIPModel, CLIPProcessor
    print(f"  Loading CLIP from {clip_model_path} ...")
    model = CLIPModel.from_pretrained(clip_model_path)
    processor = CLIPProcessor.from_pretrained(clip_model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, processor


# ── text cleaning ─────────────────────────────────────────────────────────────

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

def run(csv_path, clip_model=None, clip_processor=None, project_root=None):
    df = pd.read_csv(csv_path)
    df.drop(df.columns[0], axis=1, inplace=True)

    references = df["Expected Caption"].tolist()
    candidates = cleanup_pred_captions(df["Generated Caption"])
    candidates = ["No response" if len(c) <= 1 else c for c in candidates]

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

    acc = (df["Expected object"] == df["Predicted object"]).mean()
    results["Object Accuracy"] = round(acc, 3)

    # CLIPScore — reference-free, compares caption against actual image
    if clip_model is not None and "Ground Truth Image" in df.columns:
        mean, std = compute_clip_score(
            df["Ground Truth Image"].tolist(), candidates,
            clip_model, clip_processor, project_root,
        )
        if mean is not None:
            results["Mean CLIPScore"], results["SD CLIPScore"] = mean, std

    return results


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    RESULTS_DIR = os.path.abspath(args.results_dir)
    print(f"Evaluating CSVs in: {RESULTS_DIR}")

    clip_model, clip_processor, project_root = None, None, None
    if not args.no_clip:
        clip_path = args.clip_model or find_clip_model(RESULTS_DIR)
        project_root = find_project_root(RESULTS_DIR)
        if clip_path and project_root:
            try:
                clip_model, clip_processor = load_clip(clip_path)
                print(f"  Project root for images: {project_root}")
            except Exception as e:
                print(f"  CLIPScore disabled: {e}")
        else:
            if not clip_path:
                print("  CLIPScore disabled (no CLIP model found; use --clip-model to set path)")
            if not project_root:
                print("  CLIPScore disabled (data/images/ not found relative to results dir)")

    all_res = {}
    _skip = {"all_results.csv", "sweep_summary.csv"}
    csv_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv") and f not in _skip)

    for fname in csv_files:
        print(f"\nEvaluating: {fname} ...")
        try:
            results = run(
                os.path.join(RESULTS_DIR, fname),
                clip_model=clip_model,
                clip_processor=clip_processor,
                project_root=project_root,
            )
            all_res[fname.replace(".csv", "")] = results
            for k, v in results.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  ERROR: {e}")

    out_path = os.path.join(RESULTS_DIR, "all_results.csv")
    pd.DataFrame(all_res).transpose().to_csv(out_path)
    print(f"\nSaved to {out_path}")
