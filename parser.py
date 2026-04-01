"""
Parses and validates the JSON response from Claude.
Returns a validated dict on success, or None on any failure.
"""

import json
import re

REQUIRED_FIELDS = {
    "plant_species": str,
    "health_score": (int, float),
    "health_signals": list,
    "growth_stage": str,
    "issues": list,
    "care_flags": list,
    "notes": str,
}

VALID_GROWTH_STAGES = {"seedling", "vegetative", "flowering", "fruiting", "dormant", "unknown"}
VALID_CARE_FLAGS = {"needs_water", "needs_repot", "check_for_pests", "needs_pruning", "needs_fertilizer", "needs_light"}


def _extract_json(text: str) -> str:
    """Strip any accidental markdown fences around the JSON."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    return match.group(0)


def parse(raw: str) -> tuple[dict | None, str | None]:
    """
    Parse and validate Claude's raw text response.

    Returns:
        (parsed_dict, None)        on success
        (None, error_message)      on any failure
    """
    try:
        json_str = _extract_json(raw)
        data = json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        return None, f"JSON decode error: {e}"

    # Check required fields and types
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            return None, f"Missing required field: '{field}'"
        if not isinstance(data[field], expected_type):
            return None, f"Field '{field}' has wrong type: expected {expected_type}, got {type(data[field])}"

    # Validate health_score range
    if not (0 <= data["health_score"] <= 10):
        return None, f"health_score out of range: {data['health_score']}"

    # Validate growth_stage enum
    if data["growth_stage"] not in VALID_GROWTH_STAGES:
        return None, f"Invalid growth_stage: '{data['growth_stage']}'"

    # Validate care_flags values (warn but don't fail on unknown flags)
    unknown_flags = [f for f in data["care_flags"] if f not in VALID_CARE_FLAGS]
    if unknown_flags:
        print(f"[parser] Warning: unknown care_flags will be stored as-is: {unknown_flags}")

    return data, None
