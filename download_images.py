"""
download_images.py
------------------
Downloads image class folders from the shared Google Drive using file IDs
already enumerated in /tmp/gdown_images.log.

Only downloads files needed by the pipeline:
  - *_sketch.JPEG   (used by EEGDataset and EEGFineTuningDataset for training)
  - *_caption.txt   (used by EEGFineTuningDataset and EEGInferenceDataset)
  - *.JPEG          (original image, stored as path in CSV — never actually opened)

Spectrograms (*_spectro_*.JPEG) are skipped — they are not referenced anywhere
in the Python codebase.

Usage:
    python download_images.py [--dry-run]
"""

import os
import re
import sys
import time
import argparse
import requests
from pathlib import Path

LOG_FILE    = "/tmp/gdown_images.log"
IMAGES_DIR  = Path(__file__).parent / "data" / "images"
DELAY_SEC   = 1.0   # seconds between downloads (avoids rate-limiting)
MAX_RETRIES = 3

# Only these suffixes are used by the Python code
NEEDED_SUFFIXES = ("_sketch.JPEG", "_caption.txt", ".JPEG")

# Files ending in _spectro_*.JPEG are NOT needed
SKIP_PATTERN = re.compile(r"_spectro_\d+\.JPEG$")


def parse_log(log_path: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse gdown log and return:
        { class_name: [(file_id, file_name), ...] }
    Only includes files matching NEEDED_SUFFIXES and not SKIP_PATTERN.
    """
    folder_map: dict[str, list[tuple[str, str]]] = {}
    current_class = None

    # Match only real GDrive IDs (≥25 alphanum/dash/underscore chars) followed
    # by an ImageNet-style class name (n + digits). This excludes the spurious
    # "Retrieving folder contents completed" line in the gdown log.
    folder_re  = re.compile(r"^Retrieving folder ([A-Za-z0-9_-]{25,}) (n\d+)$")
    file_re    = re.compile(r"^Processing file (\S+) (.+)$")

    with open(log_path) as f:
        for line in f:
            line = line.rstrip()
            m = folder_re.match(line)
            if m:
                current_class = m.group(2)  # group(1)=folder_id, group(2)=class_name
                if current_class not in folder_map:
                    folder_map[current_class] = []
                continue

            m = file_re.match(line)
            if m and current_class:
                file_id, file_name = m.group(1), m.group(2)
                if SKIP_PATTERN.search(file_name):
                    continue
                if any(file_name.endswith(s) for s in NEEDED_SUFFIXES):
                    folder_map[current_class].append((file_id, file_name))

    return folder_map


def download_file(file_id: str, dest_path: Path, dry_run: bool) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True  # already downloaded

    if dry_run:
        print(f"    [dry-run] would download → {dest_path.name}")
        return True

    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if dest_path.stat().st_size > 0:
                return True
            dest_path.unlink(missing_ok=True)
        except Exception as e:
            wait = 2 ** attempt
            print(f"    attempt {attempt+1} failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded without downloading")
    args = parser.parse_args()

    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        print("Run the original gdown command first to generate the file listing.")
        sys.exit(1)

    folder_map = parse_log(LOG_FILE)
    total_classes   = len(folder_map)
    total_needed    = sum(len(v) for v in folder_map.values())

    print(f"Classes found in log : {total_classes}")
    print(f"Files to download    : {total_needed}  (sketch + caption + original only)")
    print(f"Output directory     : {IMAGES_DIR}")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")
    print()

    downloaded = skipped = failed = 0

    for class_name, files in sorted(folder_map.items()):
        class_dir = IMAGES_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        need = [(fid, fname) for fid, fname in files
                if not (class_dir / fname).exists()
                or (class_dir / fname).stat().st_size == 0]

        print(f"[{class_name}]  {len(files)} needed, {len(files) - len(need)} already on disk, {len(need)} to fetch")

        for file_id, file_name in need:
            dest = class_dir / file_name
            ok = download_file(file_id, dest, args.dry_run)
            if ok:
                downloaded += 1
            else:
                print(f"  FAILED: {file_name}")
                failed += 1
            if not args.dry_run:
                time.sleep(DELAY_SEC)

        skipped += len(files) - len(need)

    print(f"\nDone.  downloaded={downloaded}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
