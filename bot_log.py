"""Append-only prediction + outcome log for post-tournament accuracy tracking."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from bot_config import LOG_DIR

LOG_CSV = LOG_DIR / "predictions.csv"
FIELDS = [
    "logged_at", "match_no", "date", "group", "home", "away", "city",
    "p_home", "p_draw", "p_away", "predicted", "confidence",
    "home_elo", "away_elo",
    "tip", "tip_ev",
    "final_home_score", "final_away_score", "actual", "was_correct",
    "tip_points",
]


def _ensure_header() -> None:
    if LOG_CSV.exists():
        return
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def log_prediction(row: dict) -> None:
    _ensure_header()
    row = {k: row.get(k) for k in FIELDS}
    row["logged_at"] = datetime.utcnow().isoformat()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
