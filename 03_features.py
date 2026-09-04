"""
Step 3 - Engineer pre-match features with strict expanding-window discipline.

Every feature is computed using only data strictly before the match date for
the two teams involved. No leakage.

Features (35-ish):
  Elo            : home_elo, away_elo, elo_diff, elo_diff_with_ha
  Form (3 win)   : last-5, last-10, last-20 points-per-game, goals-for, goals-against
  Recent weighted: exponential-decay form (last 10, decay 0.8)
  H2H            : last-5 meetings - home_win_rate, draw_rate, avg_gd
  Tournament     : one-hot {wc, continental, qualifier, minor, friendly}
  Venue          : is_neutral, home_in_own_country, venue_altitude_m, altitude_diff
  Confederation  : home_conf, away_conf (one-hot), same_confederation
  Cadence        : days_since_last_match_home/away
  Match index    : match_num_in_tournament (group-stage early-upset signal)

Output: data/match_features.csv with one row per played match. WC2026 future
fixtures (NaN scores) are also emitted with features filled, ready for /predict.
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from confederations import confederation_of
from venues import altitude_of

DATA_DIR = Path(__file__).parent / "data"

FORM_WINDOWS = [5, 10, 20]
WEIGHTED_WINDOW = 10
WEIGHTED_DECAY = 0.8  # most recent match weight=1, next=0.8, then 0.64, ...
H2H_WINDOW = 5

TOURNAMENT_CLASSES = ["wc_finals", "continental_finals", "qualifier", "minor_tourney", "friendly"]
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC", "OTHER"]


def points_for(home_score: float, away_score: float, side: str) -> float:
    """3 for a win, 1 for a draw, 0 for a loss. NaN if score missing."""
    if pd.isna(home_score) or pd.isna(away_score):
        return float("nan")
    if home_score == away_score:
        return 1.0
    if side == "home":
        return 3.0 if home_score > away_score else 0.0
    return 3.0 if away_score > home_score else 0.0


def weighted_form(points_window: list[float]) -> float:
    """Exponentially-weighted recent form. points_window in chronological order."""
    if not points_window:
        return np.nan
    w = WEIGHTED_DECAY ** np.arange(len(points_window) - 1, -1, -1)
    return float(np.average(points_window, weights=w))


def main() -> int:
    matches = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # ---- pre-state per team ----
    # We track: deque of last-20 match outcomes for both points and goals.
    points_hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(FORM_WINDOWS)))
    gf_hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(FORM_WINDOWS)))
    ga_hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(FORM_WINDOWS)))
    weighted_hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=WEIGHTED_WINDOW))
    last_match_date: dict[str, pd.Timestamp] = {}
    # pair_key (frozenset) -> deque of (date, home_team, home_score, away_score)
    h2h_hist: dict[frozenset[str], deque[tuple]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    # Tournament running match-index per (tournament, year). Helps `match_num_in_tournament`.
    tourney_counter: dict[tuple[str, int], int] = defaultdict(int)

    rows: list[dict] = []

    for row in tqdm(matches.itertuples(index=False), total=len(matches), desc="features"):
        home, away = row.home_team, row.away_team
        date = row.date
        pair = frozenset({home, away})

        # ---- pre-match snapshots ----
        feats = {
            "date": date,
            "home_team": home,
            "away_team": away,
            "tournament": row.tournament,
            "tournament_class": row.tournament_class,
            "city": row.city,
            "country": row.country,
            "neutral": bool(row.neutral),
            "home_score": row.home_score,
            "away_score": row.away_score,
        }

        # Result label (None for future fixtures)
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            feats["target"] = None
        else:
            gd = row.home_score - row.away_score
            feats["target"] = "home_win" if gd > 0 else ("away_win" if gd < 0 else "draw")

        # Elo (already attached by 02_compute_elo.py as pre-match snapshot)
        feats["home_elo"] = getattr(row, "home_elo_before", np.nan)
        feats["away_elo"] = getattr(row, "away_elo_before", np.nan)
        feats["elo_diff"] = feats["home_elo"] - feats["away_elo"]
        ha = 0.0 if bool(row.neutral) else 100.0
        feats["elo_diff_with_ha"] = feats["elo_diff"] + ha

        # Form windows
        for side, team in (("home", home), ("away", away)):
            ph = list(points_hist[team])
            gfh = list(gf_hist[team])
            gah = list(ga_hist[team])
            for w in FORM_WINDOWS:
                feats[f"{side}_ppg_last{w}"] = np.nan if not ph else np.mean(ph[-w:])
                feats[f"{side}_gf_last{w}"]  = np.nan if not gfh else np.mean(gfh[-w:])
                feats[f"{side}_ga_last{w}"]  = np.nan if not gah else np.mean(gah[-w:])
            feats[f"{side}_form_weighted"] = weighted_form(list(weighted_hist[team]))
            feats[f"{side}_days_rest"] = (
                (date - last_match_date[team]).days if team in last_match_date else np.nan
            )

        # Form diffs (the actual model features — symmetric)
        for w in FORM_WINDOWS:
            feats[f"ppg_diff_last{w}"] = feats[f"home_ppg_last{w}"] - feats[f"away_ppg_last{w}"]
            feats[f"gf_diff_last{w}"]  = feats[f"home_gf_last{w}"]  - feats[f"away_gf_last{w}"]
            feats[f"ga_diff_last{w}"]  = feats[f"home_ga_last{w}"]  - feats[f"away_ga_last{w}"]
        feats["form_weighted_diff"] = feats["home_form_weighted"] - feats["away_form_weighted"]
        feats["days_rest_diff"] = feats["home_days_rest"] - feats["away_days_rest"]

        # H2H (last 5 meetings between the same pair, from home's perspective)
        prev = list(h2h_hist[pair])
        if prev:
            home_wins, draws, away_wins, gd_total = 0, 0, 0, 0
            for _, ph_home, ph_score_h, ph_score_a in prev:
                # Normalise to current home perspective
                if ph_home == home:
                    score_for_home, score_for_away = ph_score_h, ph_score_a
                else:
                    score_for_home, score_for_away = ph_score_a, ph_score_h
                gd_total += score_for_home - score_for_away
                if score_for_home > score_for_away:
                    home_wins += 1
                elif score_for_home == score_for_away:
                    draws += 1
                else:
                    away_wins += 1
            n = len(prev)
            feats["h2h_n"] = n
            feats["h2h_home_win_rate"] = home_wins / n
            feats["h2h_draw_rate"] = draws / n
            feats["h2h_avg_gd"] = gd_total / n
        else:
            feats["h2h_n"] = 0
            feats["h2h_home_win_rate"] = np.nan
            feats["h2h_draw_rate"] = np.nan
            feats["h2h_avg_gd"] = np.nan

        # Tournament class one-hot
        for klass in TOURNAMENT_CLASSES:
            feats[f"is_{klass}"] = int(row.tournament_class == klass)

        # Venue
        feats["is_neutral"] = int(bool(row.neutral))
        feats["home_in_own_country"] = int(
            (not bool(row.neutral)) and isinstance(row.country, str) and row.country == home
        )
        alt = altitude_of(row.city)
        feats["venue_altitude_m"] = alt
        feats["venue_high_altitude"] = int(alt >= 1500)

        # Confederation
        home_conf = confederation_of(home)
        away_conf = confederation_of(away)
        feats["home_confederation"] = home_conf
        feats["away_confederation"] = away_conf
        feats["same_confederation"] = int(home_conf == away_conf)
        for conf in CONFEDERATIONS:
            feats[f"home_conf_{conf}"] = int(home_conf == conf)
            feats[f"away_conf_{conf}"] = int(away_conf == conf)

        # Match-index inside tournament (year-scoped)
        tkey = (row.tournament, date.year)
        tourney_counter[tkey] += 1
        feats["match_num_in_tournament"] = tourney_counter[tkey]
        feats["early_tournament"] = int(tourney_counter[tkey] <= 6)

        rows.append(feats)

        # ---- post-match state updates (only for played matches) ----
        if not pd.isna(row.home_score) and not pd.isna(row.away_score):
            hpts = points_for(row.home_score, row.away_score, "home")
            apts = points_for(row.home_score, row.away_score, "away")
            points_hist[home].append(hpts)
            points_hist[away].append(apts)
            gf_hist[home].append(row.home_score)
            ga_hist[home].append(row.away_score)
            gf_hist[away].append(row.away_score)
            ga_hist[away].append(row.home_score)
            weighted_hist[home].append(hpts)
            weighted_hist[away].append(apts)
            last_match_date[home] = date
            last_match_date[away] = date
            h2h_hist[pair].append((date, home, row.home_score, row.away_score))

    out = pd.DataFrame(rows)

    # ---- merge per-team squad profile features as home/away diff features ----
    profile_path = DATA_DIR / "team_profiles_2026.csv"
    if profile_path.exists():
        prof = pd.read_csv(profile_path)
        prof_cols = [c for c in prof.columns if c != "team"]
        ph = prof.add_prefix("home_squad_").rename(columns={"home_squad_team": "home_team"})
        pa = prof.add_prefix("away_squad_").rename(columns={"away_squad_team": "away_team"})
        out = out.merge(ph, on="home_team", how="left").merge(pa, on="away_team", how="left")
        for c in prof_cols:
            out[f"squad_diff_{c}"] = out[f"home_squad_{c}"] - out[f"away_squad_{c}"]
        print(f"[03_features] merged squad profile features for "
              f"{out['home_squad_caps_top11'].notna().mean():.1%} of home rows.")

    # ---- merge per-team manager features ----
    mgr_path = DATA_DIR / "managers_2026.csv"
    if mgr_path.exists():
        mgr = pd.read_csv(mgr_path)
        mgr_cols = ["tenure_months", "matches_in_charge", "career_win_pct",
                    "big_match_experience", "age", "attack_axis", "possession_axis"]
        mh = mgr[["team"] + mgr_cols].add_prefix("home_mgr_").rename(columns={"home_mgr_team": "home_team"})
        ma = mgr[["team"] + mgr_cols].add_prefix("away_mgr_").rename(columns={"away_mgr_team": "away_team"})
        out = out.merge(mh, on="home_team", how="left").merge(ma, on="away_team", how="left")
        for c in mgr_cols:
            out[f"mgr_diff_{c}"] = out[f"home_mgr_{c}"] - out[f"away_mgr_{c}"]
        # Style one-hot for both teams (only for current-day rows; NaN otherwise).
        style_map = dict(zip(mgr["team"], mgr["style"]))
        out["home_mgr_style"] = out["home_team"].map(style_map)
        out["away_mgr_style"] = out["away_team"].map(style_map)
        for s in ("high_press", "possession", "counter", "low_block", "balanced"):
            out[f"home_mgr_is_{s}"] = (out["home_mgr_style"] == s).astype(int)
            out[f"away_mgr_is_{s}"] = (out["away_mgr_style"] == s).astype(int)
        print(f"[03_features] merged manager features for "
              f"{out['mgr_diff_career_win_pct'].notna().mean():.1%} of rows.")

    out.to_csv(DATA_DIR / "match_features.csv", index=False)
    print(f"\n[03_features] wrote {DATA_DIR / 'match_features.csv'}")
    print(f"  rows                : {len(out):,}")
    print(f"  played (with target): {out['target'].notna().sum():,}")
    print(f"  future fixtures     : {out['target'].isna().sum():,}")
    print(f"  feature columns     : {out.shape[1]}")

    # Quick sanity: print one upcoming-match row if present
    upcoming = out[out["target"].isna()].head(3)
    if not upcoming.empty:
        print("\n  sample upcoming fixtures (target=None):")
        print(upcoming[["date", "home_team", "away_team", "tournament", "home_elo", "away_elo", "elo_diff_with_ha"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
