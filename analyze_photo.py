#!/usr/bin/env python3
"""
Analyze a plant photo with Claude vision and store the result in SQLite.

Usage:
    python analyze_photo.py --filepath /path/to/photo.jpg --photo-id <uuid>

photo-id must match an existing row in the photos table (written by log_photo / pipeline).
"""

import argparse
import base64
import io
import sys

import anthropic
from PIL import Image, ImageOps

import db
import parser as response_parser
import prompt as prompt_builder


# ── Image prep (same approach as test_claud_vision.py) ──────────────────────

def prepare_image(path: str, max_size: tuple = (1568, 1568), quality: int = 85) -> str:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(max_size, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        return base64.standard_b64encode(buffer.read()).decode("utf-8")


# ── Core pipeline ────────────────────────────────────────────────────────────

def analyze(filepath: str, photo_id: str) -> str:
    """
    Run the full analysis pipeline for a single photo.
    Always writes to the DB (with parse_error flag if needed).
    Returns the analysis_id.
    """
    print(f"[analyze] Preparing image: {filepath}")
    image_b64 = prepare_image(filepath)

    print("[analyze] Calling Claude API...")
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=prompt_builder.SYSTEM_PROMPT,
        messages=prompt_builder.build_messages(image_b64),
    )
    raw = response.content[0].text
    print(f"[analyze] Raw response:\n{raw}\n")

    parsed, error = response_parser.parse(raw)

    if error:
        print(f"[analyze] Parse error (will be flagged in DB): {error}", file=sys.stderr)
    else:
        print(f"[analyze] Parsed OK — species={parsed['plant_species']}, health={parsed['health_score']}/10")

    analysis_id = db.insert_analysis(photo_id, raw, parsed)
    print(f"[analyze] Stored analysis: id={analysis_id} parse_error={0 if parsed else 1}")
    return analysis_id


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a plant photo and store results")
    ap.add_argument("--filepath", required=True)
    ap.add_argument("--photo-id", required=True, dest="photo_id")
    args = ap.parse_args()

    analyze(args.filepath, args.photo_id)


if __name__ == "__main__":
    main()
