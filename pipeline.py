#!/usr/bin/env python3
"""
Single iPhone trigger: log photo → analyze with Claude → store results.

Usage:
    python pipeline.py --filepath /path/to/photo.jpg
    python pipeline.py --filepath /path/to/photo.jpg --timestamp "2026-03-31T10:00:00"

iPhone Shortcuts SSH command:
    ssh user@mac python3 /path/to/plant-monitoring/pipeline.py --filepath "$PHOTO_PATH"
"""

import argparse
import sys
import time
from pathlib import Path

import analyze_photo
import log_photo as logger

SYNC_TIMEOUT = 30   # seconds to wait for iCloud to sync the file
SYNC_INTERVAL = 2   # seconds between checks


def wait_for_sync(filepath: str, timeout: int = SYNC_TIMEOUT, interval: int = SYNC_INTERVAL) -> None:
    """Block until the file exists on disk, or raise if timeout is exceeded."""
    path = Path(filepath)
    elapsed = 0
    print(f"[sync] Waiting for file to sync: {filepath}")
    while not path.exists():
        if elapsed >= timeout:
            raise TimeoutError(
                f"File did not appear after {timeout}s — iCloud may still be syncing: {filepath}"
            )
        print(f"[sync] Not found yet, retrying in {interval}s... ({elapsed}s elapsed)")
        time.sleep(interval)
        elapsed += interval
    print(f"[sync] File ready after {elapsed}s")


def run(filepath: str, timestamp: str | None = None) -> None:
    wait_for_sync(filepath)

    print("── Step 1: Log photo ───────────────────────────────────")
    photo_id = logger.log_photo(filepath, timestamp=timestamp)

    print("\n── Step 2: Analyze photo ───────────────────────────────")
    analyze_photo.analyze(filepath, photo_id)

    print("\nDone.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Log and analyze a plant photo end-to-end")
    ap.add_argument("--filepath", required=True, help="Path to the photo file")
    ap.add_argument("--timestamp", default=None, help="ISO 8601 timestamp (default: now)")
    args = ap.parse_args()

    run(args.filepath, args.timestamp)


if __name__ == "__main__":
    main()
