"""Async data fetcher for finished international match results.

Pulls the martj42 results.csv (updated daily). Only matches dated on/after
the WC2026 kickoff are considered, so historical noise doesn't trigger
spurious 'result detected' events.
"""
from __future__ import annotations

from datetime import datetime
from io import StringIO

import httpx
import pandas as pd

from bot_config import MARTJ42_RESULTS_URL


WC2026_KICKOFF = datetime(2026, 6, 11)


async def fetch_recent_results() -> pd.DataFrame:
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(MARTJ42_RESULTS_URL)
        r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= WC2026_KICKOFF].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    return df


def find_result(df: pd.DataFrame, date: datetime, home: str, away: str) -> tuple[int, int] | None:
    """Look for a match on the given date with (home, away) — either order."""
    same_day = df[df["date"].dt.date == date.date()]
    if same_day.empty:
        return None
    hit = same_day[(same_day["home_team"] == home) & (same_day["away_team"] == away)]
    if hit.empty:
        # martj42 may swap home/away if neutral venue
        hit = same_day[(same_day["home_team"] == away) & (same_day["away_team"] == home)]
        if hit.empty:
            return None
        row = hit.iloc[0]
        return int(row["away_score"]), int(row["home_score"])
    row = hit.iloc[0]
    return int(row["home_score"]), int(row["away_score"])
