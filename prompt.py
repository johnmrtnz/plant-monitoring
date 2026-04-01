"""
Builds the structured prompt sent to Claude for plant photo analysis.
"""

SYSTEM_PROMPT = """\
You are a plant health expert. When given a photo, you analyze it and return \
a single JSON object — no markdown, no explanation, just the raw JSON.
"""

USER_PROMPT = """\
Analyze this plant photo and return ONLY a JSON object with exactly these fields:

{
  "plant_species": "common name of the plant, or 'unknown' if unsure",
  "health_score": <integer 0-10, where 10 is perfectly healthy>,
  "health_signals": [<list of strings describing visible health indicators>],
  "growth_stage": "<one of: seedling, vegetative, flowering, fruiting, dormant, unknown>",
  "issues": [<list of strings describing any problems observed, empty list if none>],
  "care_flags": [<list of zero or more of: needs_water, needs_repot, check_for_pests, needs_pruning, needs_fertilizer>],
  "notes": "<one or two sentence free-text summary>"
}

Return nothing except the JSON object.
"""


def build_messages(image_b64: str, media_type: str = "image/jpeg") -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": USER_PROMPT},
            ],
        }
    ]
