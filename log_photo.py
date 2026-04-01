#!/usr/bin/env python3
"""
Log iPhone photo details to a SQLite database.

Usage:
    python log_photo.py --filepath /path/to/photo.jpg
    python log_photo.py --filepath /path/to/photo.jpg --id "my-plant-001" --timestamp "2026-03-31T10:00:00"

Trigger from iPhone Shortcuts via SSH:
    ssh user@mac python3 /path/to/log_photo.py --filepath "$PHOTO_PATH"
"""

import argparse
import sys
import uuid
from datetime import datetime

import db


def log_photo(filepath: str, photo_id: str | None = None, timestamp: str | None = None) -> str:
    """Insert a photo record and return its id."""
    photo_id = photo_id or str(uuid.uuid4())
    timestamp = timestamp or datetime.now().isoformat()
    db.insert_photo(filepath, photo_id, timestamp)
    print(f"Logged: id={photo_id} filepath={filepath} timestamp={timestamp}")
    return photo_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Log a photo to the plant monitoring DB")
    parser.add_argument("--filepath", required=True, help="Path to the photo file")
    parser.add_argument("--id", dest="photo_id", default=None, help="Photo ID (default: auto-generated UUID)")
    parser.add_argument("--timestamp", default=None, help="ISO 8601 timestamp (default: now)")
    args = parser.parse_args()

    if args.timestamp:
        try:
            datetime.fromisoformat(args.timestamp)
        except ValueError:
            print(f"Error: invalid timestamp '{args.timestamp}'. Use ISO 8601 format, e.g. 2026-03-31T10:00:00", file=sys.stderr)
            sys.exit(1)

    log_photo(args.filepath, args.photo_id, args.timestamp)


if __name__ == "__main__":
    main()
