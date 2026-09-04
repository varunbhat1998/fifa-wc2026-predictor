"""
Referee profile + post-hoc adjustment.

Honest scope note: training the win/draw/loss model with a referee-as-feature
requires per-match referee labels for all 49k historical matches, which we
don't have. The pragmatic path:

  1. Maintain a small hand-curated CSV of WC2026-pool referees (~30 refs FIFA
     announces). Per ref: home_win_bias (their P(home_win) minus global mean
     0.489), card_rate, pen_rate.
  2. When a referee is known (FIFA assigns them per match; data fetched via
     `15_fifa_schedule.py` -> data/wc2026_fifa_schedule.csv, or set manually),
     apply a SMALL post-hoc shift to p_home / p_away that nudges by their bias.
     Capped at ±5 percentage points so it can't swing the prediction wildly.

The actual ASSIGNMENT (who refs which match) comes from FIFA's official API.
The BIAS NUMBERS are the hand-curated seed (data/referees.csv) until a proper
historical per-ref dataset lands. Bias defaults to 0 for unknown refs.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
REF_CSV = DATA_DIR / "referees.csv"

# Bias caps (in probability points, e.g. 0.05 = 5pp). Conservative because the
# data behind these numbers is thin.
MAX_HOME_SHIFT = 0.05


# Initial seed: hand-curated estimates for the highest-profile WC referees. Numbers
# are deltas vs the global mean home-win rate (~48.9%); positive = ref tends to
# give home teams slightly more decisions / wins (penalties, foul calls, time
# additions). These are rough priors — refine when FBref/Transfermarkt scrape lands.
SEED = [
    # name, home_win_bias, card_rate (yellow/match), pen_rate (pens awarded/match)
    ("Szymon Marciniak", +0.02, 4.6, 0.28),   # high-profile, balanced
    ("Daniele Orsato",   +0.04, 5.1, 0.33),   # known card-happy + slight home bias
    ("Anthony Taylor",   +0.01, 4.9, 0.31),
    ("Michael Oliver",   -0.01, 4.3, 0.24),
    ("Felix Zwayer",     +0.02, 5.0, 0.30),
    ("Slavko Vincic",    +0.03, 4.8, 0.29),
    ("Clement Turpin",   +0.01, 4.4, 0.27),
    ("Danny Makkelie",   +0.00, 4.2, 0.25),
    ("Stephanie Frappart",-0.02, 4.1, 0.23),  # tends to call symmetrically
    ("Wilton Sampaio",   +0.02, 5.2, 0.34),
    ("Cesar Ramos",      +0.04, 5.5, 0.32),   # CONCACAF; slightly home-friendly
    ("Ivan Barton",      +0.02, 4.7, 0.30),
    ("Mustapha Ghorbal", +0.01, 5.0, 0.31),
    ("Maguette Ndiaye",  +0.01, 4.8, 0.29),
    ("Facundo Tello",    +0.05, 5.4, 0.35),   # CONMEBOL; notably home-friendly
    ("Raphael Claus",    +0.03, 5.0, 0.32),
    ("Ko Hyung-jin",     +0.00, 4.5, 0.27),
    ("Mohammed Al-Hoaish",+0.01, 4.7, 0.28),
    ("Hiroyuki Kimura",  -0.01, 4.0, 0.22),
    ("Said Martinez",    +0.03, 5.1, 0.31),
]


def _ensure_seed() -> None:
    if REF_CSV.exists():
        return
    df = pd.DataFrame(SEED, columns=["referee", "home_win_bias", "card_rate", "pen_rate"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REF_CSV, index=False)


@lru_cache(maxsize=1)
def _refs() -> pd.DataFrame:
    _ensure_seed()
    return pd.read_csv(REF_CSV)


def invalidate_cache() -> None:
    _refs.cache_clear()


def _normalise(name: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", "", (name or "").lower())).strip()


def lookup(referee: str) -> dict | None:
    if not referee:
        return None
    df = _refs()
    norm = _normalise(referee)
    # FIFA returns "Wilton SAMPAIO" (all-caps surname). Try direct, then last,
    # then surname-of-fifa-format match.
    for _, r in df.iterrows():
        if _normalise(r["referee"]) == norm:
            return r.to_dict()
    last = norm.split()[-1] if norm.split() else ""
    if last:
        for _, r in df.iterrows():
            cand_last = _normalise(r["referee"]).split()[-1]
            if cand_last == last:
                return r.to_dict()
    return None


def lookup_by_id_match(id_match: str) -> dict | None:
    """When the bot has an id_match (from the FIFA schedule CSV), pull the
    referee directly without needing the caller to supply a name."""
    sched = DATA_DIR / "wc2026_fifa_schedule.csv"
    if not sched.exists() or not id_match:
        return None
    df = pd.read_csv(sched, encoding="utf-8")
    hit = df[df["id_match"].astype(str) == str(id_match)]
    if hit.empty:
        return None
    ref = hit.iloc[0].get("referee")
    if not isinstance(ref, str) or not ref.strip():
        return None
    return lookup(ref) or {"referee": ref, "home_win_bias": 0.0, "card_rate": None, "pen_rate": None}


def apply_ref_adjustment(p_home: float, p_draw: float, p_away: float,
                          referee: str | None) -> tuple[float, float, float, dict | None]:
    """Return adjusted probabilities + the ref record (or None)."""
    info = lookup(referee) if referee else None
    if not info:
        return p_home, p_draw, p_away, None
    bias = float(info.get("home_win_bias") or 0.0)
    shift = max(-MAX_HOME_SHIFT, min(MAX_HOME_SHIFT, bias))
    # Shift from away to home (or vice versa); draws untouched.
    new_home = max(1e-4, min(1 - 1e-4, p_home + shift))
    new_away = max(1e-4, min(1 - 1e-4, p_away - shift))
    total = new_home + p_draw + new_away
    return new_home / total, p_draw / total, new_away / total, info


if __name__ == "__main__":
    import sys
    _ensure_seed()
    df = _refs()
    print(f"Loaded {len(df)} referee profiles from {REF_CSV}")
    print(df.to_string(index=False))
    if len(sys.argv) > 1:
        name = " ".join(sys.argv[1:])
        rec = lookup(name)
        if rec:
            print(f"\nLookup '{name}' -> {rec}")
        else:
            print(f"\nLookup '{name}' -> not found")
