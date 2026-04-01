"""
Database layer — initialisation and all insert/query operations.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "photos.db"


# ── Schema ─────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id          TEXT PRIMARY KEY,
            filepath    TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id            TEXT PRIMARY KEY,
            photo_id      TEXT NOT NULL REFERENCES photos(id),
            plant_species TEXT,
            health_score  REAL,
            growth_stage  TEXT,
            health_signals TEXT,   -- JSON array
            issues        TEXT,    -- JSON array
            care_flags    TEXT,    -- JSON array
            notes         TEXT,
            raw_response  TEXT NOT NULL,
            parse_error   INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


# ── Photos ──────────────────────────────────────────────────────────────────

def insert_photo(filepath: str, photo_id: str | None = None, timestamp: str | None = None) -> str:
    """Insert a photo record and return its id."""
    photo_id = photo_id or str(uuid.uuid4())
    timestamp = timestamp or datetime.now().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO photos (id, filepath, timestamp) VALUES (?, ?, ?)",
            (photo_id, filepath, timestamp),
        )
    return photo_id


# ── Analyses ────────────────────────────────────────────────────────────────

def insert_analysis(photo_id: str, raw_response: str, parsed: dict | None) -> str:
    """
    Insert an analysis record.
    If parsed is None, stores raw_response with parse_error=1.
    """
    analysis_id = str(uuid.uuid4())
    parse_error = 0 if parsed else 1

    def _json(val):
        return json.dumps(val) if val is not None else None

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO analyses
                (id, photo_id, plant_species, health_score, growth_stage,
                 health_signals, issues, care_flags, notes, raw_response, parse_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                photo_id,
                parsed.get("plant_species") if parsed else None,
                parsed.get("health_score") if parsed else None,
                parsed.get("growth_stage") if parsed else None,
                _json(parsed.get("health_signals")) if parsed else None,
                _json(parsed.get("issues")) if parsed else None,
                _json(parsed.get("care_flags")) if parsed else None,
                parsed.get("notes") if parsed else None,
                raw_response,
                parse_error,
            ),
        )
    return analysis_id
