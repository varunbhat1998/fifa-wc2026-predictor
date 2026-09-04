"""
Feature engine for serving predictions on unplayed matches.

The training pipeline (03_features.py) walks history chronologically. For a
*new* fixture we instead use the latest state cached on disk:

  team_elo_current.csv  -> Elo per team
  matches.csv           -> all historical results (used to reconstruct form +
                           head-to-head on demand)

This module exposes `compute_features(home, away, ...)` returning a one-row
DataFrame with the exact 41 columns the trained ensemble expects.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from confederations import confederation_of
from venues import altitude_of

DATA_DIR = Path(__file__).parent / "data"

FORM_WINDOWS = [5, 10, 20]
WEIGHTED_DECAY = 0.8
H2H_WINDOW = 5

CONFEDERATIONS = ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC", "OTHER"]
TOURNAMENT_CLASSES = ["wc_finals", "continental_finals", "qualifier",
                      "minor_tourney", "friendly"]


@lru_cache(maxsize=1)
def _matches() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "matches.csv")
    # parse_dates= silently keeps the column as object dtype if any row fails to
    # parse (which can happen after retrain appends with a different format).
    # Force-coerce and drop unparseable rows so downstream `df["date"] < X`
    # comparisons never blow up.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score", "date"]).sort_values("date")
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def _managers() -> dict[str, dict]:
    path = DATA_DIR / "managers_2026.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8")
    return {r["team"]: r.to_dict() for _, r in df.iterrows()}


@lru_cache(maxsize=1)
def _team_profiles() -> dict[str, dict[str, float]]:
    path = DATA_DIR / "team_profiles_2026.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[r["team"]] = {c: float(r[c]) for c in df.columns if c != "team"}
    return out


@lru_cache(maxsize=1)
def _squads() -> pd.DataFrame:
    path = DATA_DIR / "squads_2026.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _profile_from_starters(team: str, starters: list[str]) -> dict[str, float] | None:
    """Recompute team_strength features using ONLY the supplied 11 starters.
    Returns None when squad data is missing — caller falls back to 26-man profile."""
    sq = _squads()
    if sq.empty:
        return None
    team_sq = sq[sq["team"] == team]
    if team_sq.empty:
        return None
    # Match starters to squad rows via _name_match (substring on last name).
    import re
    def _norm(s):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", "", (s or "").lower())).strip()
    norm_full = {_norm(p): r for _, r in team_sq.iterrows() for p in [r["player"]]}
    matched_rows = []
    for s in starters:
        ns = _norm(s)
        if ns in norm_full:
            matched_rows.append(norm_full[ns]); continue
        last = ns.split()[-1] if ns.split() else ""
        candidates = [r for k, r in norm_full.items() if k.split() and k.split()[-1] == last]
        if len(candidates) == 1:
            matched_rows.append(candidates[0])
    if len(matched_rows) < 7:        # too few matched — bail
        return None
    sub_df = pd.DataFrame(matched_rows)
    # Delegate to the same aggregator that 11_team_strength.py uses.
    import importlib, sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    _ts = importlib.import_module("11_team_strength")
    return _ts.compute_team_profile(sub_df)


@lru_cache(maxsize=1)
def _elo_current() -> dict[str, float]:
    df = pd.read_csv(DATA_DIR / "team_elo_current.csv")
    return dict(zip(df["team"], df["rating"]))


def invalidate_cache() -> None:
    """Call after retrain to pick up the new state files."""
    _matches.cache_clear()
    _elo_current.cache_clear()
    _team_profiles.cache_clear()
    _squads.cache_clear()
    _managers.cache_clear()


def _team_last_n(team: str, before: datetime, n: int) -> pd.DataFrame:
    df = _matches()
    mask = ((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < before)
    return df[mask].tail(n)


def _team_form(team: str, before: datetime) -> dict[str, float]:
    df = _team_last_n(team, before, max(FORM_WINDOWS))
    if df.empty:
        return {k: float("nan") for k in (
            "ppg_last5", "ppg_last10", "ppg_last20",
            "gf_last5", "gf_last10", "gf_last20",
            "ga_last5", "ga_last10", "ga_last20",
            "form_weighted", "last_match_date",
        )}
    is_home = df["home_team"].values == team
    pts = np.where(
        df["home_score"].values == df["away_score"].values, 1.0,
        np.where(
            (is_home & (df["home_score"].values > df["away_score"].values)) |
            (~is_home & (df["away_score"].values > df["home_score"].values)),
            3.0, 0.0,
        ),
    )
    gf = np.where(is_home, df["home_score"].values, df["away_score"].values)
    ga = np.where(is_home, df["away_score"].values, df["home_score"].values)

    out = {}
    for w in FORM_WINDOWS:
        out[f"ppg_last{w}"] = float(pts[-w:].mean()) if len(pts) else float("nan")
        out[f"gf_last{w}"] = float(gf[-w:].mean()) if len(gf) else float("nan")
        out[f"ga_last{w}"] = float(ga[-w:].mean()) if len(ga) else float("nan")
    recent = pts[-10:]
    if len(recent):
        weights = WEIGHTED_DECAY ** np.arange(len(recent) - 1, -1, -1)
        out["form_weighted"] = float(np.average(recent, weights=weights))
    else:
        out["form_weighted"] = float("nan")
    out["last_match_date"] = df["date"].iloc[-1].to_pydatetime()
    return out


def _h2h(home: str, away: str, before: datetime) -> dict[str, float]:
    df = _matches()
    pair_mask = (
        ((df["home_team"] == home) & (df["away_team"] == away)) |
        ((df["home_team"] == away) & (df["away_team"] == home))
    )
    df = df[pair_mask & (df["date"] < before)].tail(H2H_WINDOW)
    if df.empty:
        return {"h2h_n": 0, "h2h_home_win_rate": np.nan,
                "h2h_draw_rate": np.nan, "h2h_avg_gd": np.nan}
    hw = dr = aw = 0
    gd_total = 0.0
    for _, r in df.iterrows():
        if r["home_team"] == home:
            sh, sa = r["home_score"], r["away_score"]
        else:
            sh, sa = r["away_score"], r["home_score"]
        gd_total += sh - sa
        if sh > sa: hw += 1
        elif sh == sa: dr += 1
        else: aw += 1
    n = len(df)
    return {"h2h_n": n, "h2h_home_win_rate": hw / n,
            "h2h_draw_rate": dr / n, "h2h_avg_gd": gd_total / n}


@dataclass
class FixtureInput:
    home: str
    away: str
    date: datetime
    tournament_class: str       # wc_finals | continental_finals | ...
    city: str | None
    neutral: bool
    match_num_in_tournament: int = 1   # for early-upset signal
    # Optional confirmed-XI override: when populated, squad_diff_* features are
    # recomputed using ONLY the 11 starters of each team instead of the 26-man
    # squad. List of player names (canonical, matching squads_2026.csv).
    home_starters: list[str] | None = None
    away_starters: list[str] | None = None
    referee: str | None = None


def compute_features(fx: FixtureInput) -> pd.DataFrame:
    elo = _elo_current()
    home_elo = elo.get(fx.home, 1500.0)
    away_elo = elo.get(fx.away, 1500.0)
    elo_diff = home_elo - away_elo
    ha = 0.0 if fx.neutral else 100.0

    home_form = _team_form(fx.home, fx.date)
    away_form = _team_form(fx.away, fx.date)
    h2h = _h2h(fx.home, fx.away, fx.date)

    home_rest = ((fx.date - home_form["last_match_date"]).days
                 if isinstance(home_form["last_match_date"], datetime) else np.nan)
    away_rest = ((fx.date - away_form["last_match_date"]).days
                 if isinstance(away_form["last_match_date"], datetime) else np.nan)

    row: dict[str, float] = {
        "elo_diff": elo_diff,
        "elo_diff_with_ha": elo_diff + ha,
    }
    for w in FORM_WINDOWS:
        row[f"ppg_diff_last{w}"] = home_form[f"ppg_last{w}"] - away_form[f"ppg_last{w}"]
        row[f"gf_diff_last{w}"]  = home_form[f"gf_last{w}"]  - away_form[f"gf_last{w}"]
        row[f"ga_diff_last{w}"]  = home_form[f"ga_last{w}"]  - away_form[f"ga_last{w}"]
    row["form_weighted_diff"] = home_form["form_weighted"] - away_form["form_weighted"]
    row["days_rest_diff"] = home_rest - away_rest if (home_rest == home_rest and away_rest == away_rest) else np.nan

    row.update({k: v for k, v in h2h.items()})

    for klass in TOURNAMENT_CLASSES:
        row[f"is_{klass}"] = int(fx.tournament_class == klass)

    row["is_neutral"] = int(fx.neutral)
    row["home_in_own_country"] = int(not fx.neutral)
    alt = altitude_of(fx.city)
    row["venue_altitude_m"] = alt
    row["venue_high_altitude"] = int(alt >= 1500)

    hc = confederation_of(fx.home)
    ac = confederation_of(fx.away)
    row["same_confederation"] = int(hc == ac)
    for conf in CONFEDERATIONS:
        row[f"home_conf_{conf}"] = int(hc == conf)
        row[f"away_conf_{conf}"] = int(ac == conf)

    row["match_num_in_tournament"] = fx.match_num_in_tournament
    row["early_tournament"] = int(fx.match_num_in_tournament <= 6)

    # Manager features (constant per team — same shape as squad profile).
    mgrs = _managers()
    hm = mgrs.get(fx.home, {})
    am = mgrs.get(fx.away, {})
    for col in ("tenure_months", "matches_in_charge", "career_win_pct",
                "big_match_experience", "age", "attack_axis", "possession_axis"):
        h = hm.get(col, np.nan); a = am.get(col, np.nan)
        try:
            h = float(h); a = float(a)
        except Exception:
            h = a = np.nan
        row[f"mgr_diff_{col}"] = h - a if (h == h and a == a) else np.nan
    for s in ("high_press", "possession", "counter", "low_block", "balanced"):
        row[f"home_mgr_is_{s}"] = int(str(hm.get("style") or "") == s)
        row[f"away_mgr_is_{s}"] = int(str(am.get("style") or "") == s)

    # Squad profile diffs. Default = 26-man squad snapshot. If confirmed-XI is
    # provided, recompute the per-team profile from just the 11 starters — that
    # is the "best-accuracy at T-75min" path the bot uses on match day.
    profiles = _team_profiles()
    hp = profiles.get(fx.home, {})
    ap = profiles.get(fx.away, {})
    if fx.home_starters:
        hp = _profile_from_starters(fx.home, fx.home_starters) or hp
    if fx.away_starters:
        ap = _profile_from_starters(fx.away, fx.away_starters) or ap
    for col in (
        "caps_top22", "caps_top11", "goal_rate_top4_fw", "fw_goals_top4_total",
        "age_top22_avg", "age_top22_std", "top5_league_pct", "big_club_pct",
        "chemistry_max_cluster", "chemistry_top3_sum", "unique_clubs", "gk_avg_caps",
    ):
        h = hp.get(col, np.nan)
        a = ap.get(col, np.nan)
        row[f"squad_diff_{col}"] = h - a if (h == h and a == a) else np.nan

    # Extras returned alongside (the API uses these to decorate responses).
    row["__home_elo"] = home_elo
    row["__away_elo"] = away_elo
    row["__home_form_ppg10"] = home_form["ppg_last10"]
    row["__away_form_ppg10"] = away_form["ppg_last10"]

    return pd.DataFrame([row])


# ---- confidence scoring (mirrors IPL "signal agreement" idea) ----

def signal_agreement(features_row: pd.Series, predicted_class: str) -> tuple[int, list[str]]:
    """How many independent signals point to the predicted winner side?

    Returns (count, [signal_names_that_agree]). We only count Home vs Away signals;
    when the prediction is 'draw' we report a baseline of 0.
    """
    if predicted_class == "draw":
        return 0, []
    favours_home = predicted_class == "home_win"
    signals = {
        "elo":          features_row["elo_diff"] > 0,
        "elo_with_ha":  features_row["elo_diff_with_ha"] > 0,
        "form_recent":  features_row["form_weighted_diff"] > 0,
        "form_ppg10":   features_row["ppg_diff_last10"] > 0,
        "goals_for":    features_row["gf_diff_last10"] > 0,
        "goals_against":features_row["ga_diff_last10"] < 0,  # fewer conceded is good
        "h2h":          (features_row["h2h_home_win_rate"] or 0) > (features_row["h2h_draw_rate"] or 0),
        "venue_home":   features_row["home_in_own_country"] == 1,
    }
    agree = [name for name, val in signals.items() if bool(val) == favours_home]
    return len(agree), agree
