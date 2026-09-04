"""
WC 2026 head-coach database + tactical matchup logic.

Two layers:
  1. Static profile (data/managers_2026.csv) — name, age, nationality, tenure,
     career win%, big-tournament experience, preferred formation, style scores.
     Style is two numeric axes (each -2..+2):
        attack_axis      defensive (-2) .. balanced (0) .. attacking (+2)
        possession_axis  counter-press (-2) .. balanced (0) .. possession (+2)
  2. Per-match feature engine — given (home_coach, away_coach), emits diff
     features the model + post-hoc adjustment use.

Where the numbers come from: hand-curated from public coverage of each
manager's documented career and tactical fingerprint. Treat as priors —
re-run after each match-day and the bias adjustment shrinks accordingly.

Note: post-match the FIFA live endpoint exposes the actual `Tactics` formation
and Coaches[] array. After a result lands we can sync data/managers_2026.csv
to FIFA's authoritative coach data for the teams that played.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MANAGERS_CSV = DATA_DIR / "managers_2026.csv"

# Style-matchup matrix. Rows = team A's posture, cols = team B's posture.
# Entries are A's adv in pp (negative = disadvantage). Symmetric (A_adv = -B_adv).
# Empirical rough numbers from public tactical analysis; calibrated small.
#   high_press vs counter      -> high_press slight loser (counter exploits gaps)
#   possession vs low_block    -> possession slight loser (hard to break down)
#   counter vs possession      -> counter slight winner
#   balanced vs anything       -> ~neutral
STYLE_MATCHUP = {
    ("high_press", "counter"):     -0.02,
    ("high_press", "possession"):  +0.01,
    ("high_press", "low_block"):   +0.02,
    ("possession", "counter"):     +0.01,
    ("possession", "high_press"):  -0.01,
    ("possession", "low_block"):   -0.03,
    ("counter", "possession"):     -0.01,
    ("counter", "high_press"):     +0.02,
    ("counter", "low_block"):      +0.01,
    ("low_block", "possession"):   +0.03,
    ("low_block", "high_press"):   -0.02,
    ("low_block", "counter"):      -0.01,
}


SEED = [
    # team, manager, nationality, age, tenure_months, matches_in_charge,
    # career_win_pct, big_match_experience (0-10), formation, style,
    # attack_axis (-2..+2), possession_axis (-2..+2)
    # Group A
    ("Mexico", "Javier Aguirre", "Mexico", 67, 23, 95, 0.55, 8, "4-3-3", "balanced", 0, 0),
    ("South Africa", "Hugo Broos", "Belgium", 73, 48, 42, 0.45, 6, "4-2-3-1", "counter", -1, -1),
    ("South Korea", "Hong Myung-bo", "South Korea", 56, 22, 25, 0.60, 5, "4-3-3", "high_press", 1, 1),
    ("Czech Republic", "Ivan Hašek", "Czech Republic", 61, 14, 17, 0.50, 4, "4-2-3-1", "balanced", 0, 0),
    # Group B
    ("Canada", "Jesse Marsch", "United States", 52, 26, 30, 0.58, 5, "4-2-3-1", "high_press", 1, 1),
    ("Bosnia and Herzegovina", "Sergej Barbarez", "Bosnia and Herzegovina", 54, 19, 16, 0.45, 3, "4-3-3", "balanced", 0, 0),
    ("Qatar", "Bartolomé Márquez", "Spain", 47, 18, 14, 0.50, 3, "4-3-3", "possession", 1, 1),
    ("Switzerland", "Murat Yakin", "Switzerland", 51, 50, 42, 0.55, 7, "3-4-2-1", "balanced", 0, 0),
    # Group C
    ("Brazil", "Carlo Ancelotti", "Italy", 66, 14, 12, 0.65, 10, "4-3-3", "balanced", 1, 1),
    ("Morocco", "Walid Regragui", "Morocco", 50, 46, 35, 0.60, 8, "4-3-3", "counter", 0, -1),
    ("Haiti", "Sébastien Migné", "France", 50, 17, 14, 0.40, 2, "4-4-2", "counter", -1, -1),
    ("Scotland", "Steve Clarke", "Scotland", 62, 88, 67, 0.52, 6, "3-4-2-1", "counter", 0, -1),
    # Group D
    ("United States", "Mauricio Pochettino", "Argentina", 53, 22, 24, 0.56, 8, "4-2-3-1", "high_press", 1, 1),
    ("Paraguay", "Gustavo Alfaro", "Argentina", 63, 18, 17, 0.50, 6, "4-4-2", "counter", -1, -1),
    ("Australia", "Tony Popovic", "Australia", 52, 18, 16, 0.55, 5, "4-3-3", "balanced", 0, 0),
    ("Turkey", "Vincenzo Montella", "Italy", 51, 23, 22, 0.50, 6, "4-2-3-1", "possession", 1, 1),
    # Group E
    ("Germany", "Julian Nagelsmann", "Germany", 38, 32, 30, 0.55, 8, "4-2-3-1", "high_press", 1, 1),
    ("Curacao", "Dick Advocaat", "Netherlands", 78, 8, 8, 0.55, 9, "4-3-3", "balanced", 0, 0),
    ("Ivory Coast", "Emerse Faé", "Ivory Coast", 41, 30, 26, 0.58, 6, "4-3-3", "balanced", 0, 0),
    ("Ecuador", "Sebastián Beccacece", "Argentina", 45, 18, 17, 0.55, 5, "4-3-3", "balanced", 0, 0),
    # Group F
    ("Netherlands", "Ronald Koeman", "Netherlands", 62, 32, 28, 0.55, 9, "4-3-3", "possession", 1, 2),
    ("Japan", "Hajime Moriyasu", "Japan", 57, 84, 67, 0.55, 7, "3-4-2-1", "high_press", 1, 1),
    ("Sweden", "Jon Dahl Tomasson", "Denmark", 49, 14, 13, 0.50, 4, "4-3-3", "high_press", 1, 1),
    ("Tunisia", "Sami Trabelsi", "Tunisia", 57, 16, 14, 0.50, 4, "4-2-3-1", "balanced", 0, 0),
    # Group G
    ("Belgium", "Rudi Garcia", "France", 61, 14, 12, 0.55, 6, "4-3-3", "possession", 1, 1),
    ("Egypt", "Hossam Hassan", "Egypt", 59, 24, 22, 0.55, 6, "4-3-3", "balanced", 0, 0),
    ("Iran", "Amir Ghalenoei", "Iran", 62, 30, 28, 0.55, 5, "4-3-3", "counter", 0, -1),
    ("New Zealand", "Darren Bazeley", "England", 53, 26, 22, 0.45, 3, "4-3-3", "balanced", 0, 0),
    # Group H
    ("Spain", "Luis de la Fuente", "Spain", 64, 35, 36, 0.62, 8, "4-3-3", "possession", 1, 2),
    ("Cape Verde", "Pedro Leitão Brito", "Cape Verde", 53, 35, 31, 0.50, 4, "4-3-3", "balanced", 0, 0),
    ("Saudi Arabia", "Hervé Renard", "France", 57, 8, 8, 0.55, 7, "4-3-3", "counter", 0, -1),
    ("Uruguay", "Marcelo Bielsa", "Argentina", 70, 28, 28, 0.55, 9, "3-3-3-1", "high_press", 2, 1),
    # Group I
    ("France", "Didier Deschamps", "France", 57, 169, 165, 0.66, 10, "4-2-3-1", "balanced", 0, 0),
    ("Senegal", "Pape Thiaw", "Senegal", 44, 14, 12, 0.55, 4, "4-3-3", "balanced", 0, 0),
    ("Iraq", "Graham Arnold", "Australia", 62, 14, 12, 0.50, 5, "4-3-3", "balanced", 0, 0),
    ("Norway", "Ståle Solbakken", "Norway", 57, 60, 50, 0.55, 6, "4-3-3", "balanced", 0, 0),
    # Group J
    ("Argentina", "Lionel Scaloni", "Argentina", 47, 88, 78, 0.70, 10, "4-3-3", "possession", 1, 1),
    ("Algeria", "Vladimir Petković", "Bosnia and Herzegovina", 62, 19, 17, 0.50, 6, "4-3-3", "balanced", 0, 0),
    ("Austria", "Ralf Rangnick", "Germany", 67, 38, 33, 0.60, 7, "4-2-2-2", "high_press", 2, 1),
    ("Jordan", "Jamal Sellami", "Morocco", 55, 20, 17, 0.50, 3, "4-3-3", "balanced", 0, 0),
    # Group K
    ("Portugal", "Roberto Martínez", "Spain", 52, 32, 28, 0.62, 8, "4-3-3", "possession", 1, 2),
    ("DR Congo", "Sébastien Desabre", "France", 49, 25, 22, 0.50, 4, "4-3-3", "balanced", 0, 0),
    ("Uzbekistan", "Timur Kapadze", "Uzbekistan", 44, 18, 16, 0.55, 3, "4-3-3", "balanced", 0, 0),
    ("Colombia", "Néstor Lorenzo", "Argentina", 60, 38, 35, 0.60, 7, "4-2-3-1", "possession", 1, 1),
    # Group L
    ("England", "Thomas Tuchel", "Germany", 52, 14, 12, 0.62, 9, "4-3-3", "high_press", 1, 1),
    ("Croatia", "Zlatko Dalić", "Croatia", 59, 96, 88, 0.60, 9, "4-3-3", "possession", 1, 1),
    ("Ghana", "Otto Addo", "Ghana", 50, 20, 18, 0.50, 4, "4-3-3", "balanced", 0, 0),
    ("Panama", "Thomas Christiansen", "Spain", 52, 38, 35, 0.55, 5, "4-3-3", "balanced", 0, 0),
]


def _ensure_seed() -> None:
    if MANAGERS_CSV.exists():
        return
    df = pd.DataFrame(SEED, columns=[
        "team", "manager", "nationality", "age", "tenure_months", "matches_in_charge",
        "career_win_pct", "big_match_experience", "formation", "style",
        "attack_axis", "possession_axis",
    ])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANAGERS_CSV, index=False, encoding="utf-8")


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    _ensure_seed()
    return pd.read_csv(MANAGERS_CSV, encoding="utf-8")


def invalidate_cache() -> None:
    load.cache_clear()


def manager_for(team: str) -> dict | None:
    df = load()
    hit = df[df["team"] == team]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def diff_features(home_team: str, away_team: str) -> dict[str, float]:
    """Return per-match diff features the model uses."""
    h = manager_for(home_team) or {}
    a = manager_for(away_team) or {}

    def _f(d, k, default=0):
        v = d.get(k, default)
        try:
            v = float(v)
        except Exception:
            v = default
        if pd.isna(v):
            v = default
        return v

    return {
        "mgr_diff_tenure_months":        _f(h, "tenure_months") - _f(a, "tenure_months"),
        "mgr_diff_matches_in_charge":    _f(h, "matches_in_charge") - _f(a, "matches_in_charge"),
        "mgr_diff_career_win_pct":       _f(h, "career_win_pct") - _f(a, "career_win_pct"),
        "mgr_diff_big_match_experience": _f(h, "big_match_experience") - _f(a, "big_match_experience"),
        "mgr_diff_age":                  _f(h, "age") - _f(a, "age"),
        "mgr_diff_attack_axis":          _f(h, "attack_axis") - _f(a, "attack_axis"),
        "mgr_diff_possession_axis":      _f(h, "possession_axis") - _f(a, "possession_axis"),
        "mgr_same_nationality_as_team_h": int(str(h.get("nationality")) == home_team),
        "mgr_same_nationality_as_team_a": int(str(a.get("nationality")) == away_team),
    }


def style_matchup_adjustment(home_team: str, away_team: str) -> tuple[float, dict]:
    """Return (home_pp_shift, info_dict) from the style matchup. Capped ±3pp.

    Post-hoc analogue to the referee adjustment — small swing applied to the
    classifier's calibrated probability when both managers have non-balanced
    styles. Plays explicit matchups: counter beats high-press, low-block beats
    possession, etc.
    """
    h = manager_for(home_team) or {}
    a = manager_for(away_team) or {}
    hs = str(h.get("style") or "balanced")
    as_ = str(a.get("style") or "balanced")
    shift = STYLE_MATCHUP.get((hs, as_), 0.0)
    return max(-0.03, min(0.03, shift)), {
        "home_style": hs, "away_style": as_,
        "home_manager": h.get("manager"), "away_manager": a.get("manager"),
        "home_formation": h.get("formation"), "away_formation": a.get("formation"),
        "shift_applied_pp": shift,
    }


def apply_manager_adjustment(p_home: float, p_draw: float, p_away: float,
                              home_team: str, away_team: str) -> tuple[float, float, float, dict]:
    shift, info = style_matchup_adjustment(home_team, away_team)
    new_home = max(1e-4, min(1 - 1e-4, p_home + shift))
    new_away = max(1e-4, min(1 - 1e-4, p_away - shift))
    total = new_home + p_draw + new_away
    return new_home / total, p_draw / total, new_away / total, info


if __name__ == "__main__":
    _ensure_seed()
    df = load()
    print(f"Loaded {len(df)} managers")
    print(df[["team", "manager", "nationality", "career_win_pct", "formation", "style"]].head(20).to_string(index=False))
    # Style matchup demo
    print("\nStyle matchup tests:")
    for h, a in [("Spain", "Mexico"), ("England", "USA"), ("Argentina", "Brazil"),
                 ("France", "Norway"), ("Portugal", "Saudi Arabia")]:
        shift, info = style_matchup_adjustment(h, a)
        print(f"  {h:>10s} ({info['home_style']:>10s}) vs {a:<14s} ({info['away_style']:>10s})  "
              f"shift={shift:+.3f}  ({info['home_formation']} vs {info['away_formation']})")
