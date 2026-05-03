"""
download_missing_images.py
--------------------------
Uses the Google Drive API v3 (with API key) to enumerate ALL files in each
image class folder — no 50-file limit — then downloads only the missing
caption, sketch, and original JPEG files.

Usage:
    python download_missing_images.py --api-key YOUR_KEY [--dry-run]
"""

import argparse
import os
import re
import time
from pathlib import Path

import requests

# ── constants ────────────────────────────────────────────────────────────────
IMAGES_DIR   = Path(__file__).parent / "data" / "images"
LOG_FILE     = "/tmp/gdown_images.log"
DRIVE_API    = "https://www.googleapis.com/drive/v3/files"
DOWNLOAD_URL = "https://drive.google.com/uc?id={id}&export=download"
SKIP_PATTERN = re.compile(r"_spectro_\d+\.JPEG$")
NEEDED_SUFFIXES = ("_caption.txt", "_sketch.JPEG", ".JPEG")
DELAY_SEC    = 0.5
MAX_RETRIES  = 3


# ── helpers ──────────────────────────────────────────────────────────────────
def parse_folder_ids(log_path: str) -> dict:
    """Returns {class_name: folder_id} from the gdown enumeration log."""
    folder_re = re.compile(r"^Retrieving folder ([A-Za-z0-9_-]{25,}) (n\d+)$")
    result = {}
    with open(log_path) as f:
        for line in f:
            m = folder_re.match(line.rstrip())
            if m:
                folder_id, class_name = m.group(1), m.group(2)
                result[class_name] = folder_id
    return result


def list_all_files(folder_id: str, api_key: str) -> list:
    """
    Returns [(file_id, file_name), ...] for ALL files in the Drive folder,
    using the API's pagination (nextPageToken) — no 50-file cap.
    """
    files = []
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "nextPageToken,files(id,name)",
        "pageSize": 1000,
        "key": api_key,
    }
    while True:
        resp = requests.get(DRIVE_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        files.extend((f["id"], f["name"]) for f in data.get("files", []))
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return files


def is_needed(fname: str) -> bool:
    if SKIP_PATTERN.search(fname):
        return False
    return any(fname.endswith(s) for s in NEEDED_SUFFIXES)


def download_file(file_id: str, dest: Path, dry_run: bool) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True  # already present

    if dry_run:
        print(f"    [dry-run] {dest.name}")
        return True

    url = DOWNLOAD_URL.format(id=file_id)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            if dest.stat().st_size > 0:
                return True
            dest.unlink(missing_ok=True)
        except Exception as e:
            wait = 2 ** attempt
            print(f"    attempt {attempt+1} failed ({e}), retrying in {wait}s …")
            time.sleep(wait)
    return False


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Google Drive API key")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folder_map = parse_folder_ids(LOG_FILE)
    print(f"Classes: {len(folder_map)}")
    if args.dry_run:
        print("DRY RUN — nothing will be written\n")

    downloaded = skipped = failed = 0

    for class_name, folder_id in sorted(folder_map.items()):
        class_dir = IMAGES_DIR / class_name

        # Enumerate ALL files via API (no 50-file limit)
        try:
            all_files = list_all_files(folder_id, args.api_key)
        except Exception as e:
            print(f"[{class_name}] API error: {e}")
            continue

        needed = [(fid, fname) for fid, fname in all_files if is_needed(fname)]
        missing = [(fid, fname) for fid, fname in needed
                   if not (class_dir / fname).exists()
                   or (class_dir / fname).stat().st_size == 0]

        print(f"[{class_name}]  total={len(all_files)}  needed={len(needed)}  "
              f"on_disk={len(needed)-len(missing)}  to_fetch={len(missing)}")

        for file_id, file_name in missing:
            dest = class_dir / file_name
            ok = download_file(file_id, dest, args.dry_run)
            if ok:
                downloaded += 1
            else:
                print(f"  FAILED: {file_name}")
                failed += 1
            if not args.dry_run:
                time.sleep(DELAY_SEC)

        skipped += len(needed) - len(missing)

    print(f"\nDone.  downloaded={downloaded}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
