"""
Step 5 - Monte Carlo tournament simulator.

Simulates the WC2026 from current state forward N times. For each simulation:
  1. Each unplayed group match sampled from model probabilities (home/draw/away).
  2. Group standings computed using FIFA tiebreakers (points -> GD -> GF -> H2H).
  3. Top 2 per group + 8 best 3rd-place teams advance to R32.
  4. Knockout: if model draw probability triggers, simulate extra time as the
     same probabilities scaled, then a 50/50 shootout (Elo-weighted slightly).
  5. Aggregate across runs -> P(team reaches stage X).

For the MVP we run N=2000 simulations (good enough for stable top-10 numbers;
upgrade to 10k for the final published run).

Already-played results are read from data/results_actual.csv if present —
this lets the simulator advance as the tournament progresses.

Output: data/tournament_sim_results.csv with one row per team:
  team, group, p_advance_R32, p_advance_R16, p_advance_QF, p_advance_SF,
  p_advance_Final, p_champion, expected_finish_round
"""
from __future__ import annotations

import importlib
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
_train = importlib.import_module("04_train_prematch")
PrematchModel = _train.PrematchModel

import feature_engine
import wc2026_schedule

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

N_SIMS = 10000
RNG_SEED = 42

# 3rd-place qualification slots (FIFA 2026 format: 8 best 3rd-placed teams advance)
N_3RD_PLACE_QUALIFIERS = 8


def _load_model() -> PrematchModel:
    with open(MODEL_DIR / "prematch_model.pkl", "rb") as f:
        return pickle.load(f)


def _predict_group_probs(model: PrematchModel) -> dict[int, np.ndarray]:
    """For every group-stage fixture compute (p_home, p_draw, p_away)."""
    fxts = wc2026_schedule.all_group_fixtures()
    rows = []
    for f in fxts:
        fi = feature_engine.FixtureInput(
            home=f.home, away=f.away, date=f.date,
            tournament_class="wc_finals", city=f.city, neutral=f.neutral,
            match_num_in_tournament=f.match_no,
        )
        feats = feature_engine.compute_features(fi)
        feats = feats.drop(columns=[c for c in feats.columns if c.startswith("__")])
        rows.append(feats.iloc[0])
    X = pd.DataFrame(rows)
    proba = model.predict_proba(X)
    return {fxts[i].match_no: proba[i] for i in range(len(fxts))}


def _load_played_results() -> dict[int, str]:
    """Map match_no -> 'home'/'draw'/'away' for any matches whose result is already known.

    Reads data/results_actual.csv if present (columns: match_no, home_score, away_score).
    """
    path = DATA_DIR / "results_actual.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[int, str] = {}
    for _, r in df.iterrows():
        if pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
            continue
        if r["home_score"] > r["away_score"]: out[int(r["match_no"])] = "home"
        elif r["home_score"] < r["away_score"]: out[int(r["match_no"])] = "away"
        else: out[int(r["match_no"])] = "draw"
    return out


def _sample_group_match(rng: np.random.Generator, probs: np.ndarray) -> str:
    idx = rng.choice(3, p=probs)
    return ["home", "draw", "away"][idx]


def _sample_goals(rng: np.random.Generator, winner: str, elo_diff: float) -> tuple[int, int]:
    """Sample plausible scoreline for tie-break GD/GF accounting. Very rough."""
    base_total = 2.5 + 0.3 * (abs(elo_diff) / 200.0)
    total = max(0, int(rng.poisson(base_total)))
    if winner == "draw":
        h = total // 2
        return h, h
    # winner gets the larger half plus 1
    margin = max(1, int(rng.poisson(1.1)))
    if total < margin:
        total = margin
    a = max(0, (total - margin) // 2)
    b = a + margin
    return (b, a) if winner == "home" else (a, b)


def _simulate_one(rng: np.random.Generator,
                  group_probs: dict[int, np.ndarray],
                  played: dict[int, str]) -> tuple[dict[str, int], dict[str, int]]:
    """Run one full tournament. Returns (reached, group_finish):
       reached       : team -> highest round reached
                       (3=group, 4=R32, 5=R16, 6=QF, 7=SF, 8=Final, 9=Champion)
       group_finish  : team -> finish position in group (1..4)
    """
    fxts = wc2026_schedule.all_group_fixtures()
    # team -> {pts, gf, ga}
    tbl: dict[str, dict[str, int]] = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0})
    # group -> ordered list of teams (initial group)
    grouping: dict[str, list[str]] = {g: list(ts) for g, ts in wc2026_schedule.GROUPS.items()}

    # For Elo-based scoring sampling we need a quick Elo lookup
    elo = feature_engine._elo_current()

    for f in fxts:
        probs = group_probs[f.match_no]
        result = played.get(f.match_no) or _sample_group_match(rng, probs)
        elo_diff = elo.get(f.home, 1500) - elo.get(f.away, 1500)
        hs, as_ = _sample_goals(rng, result, elo_diff)
        tbl[f.home]["gf"] += hs; tbl[f.home]["ga"] += as_
        tbl[f.away]["gf"] += as_; tbl[f.away]["ga"] += hs
        if result == "home":
            tbl[f.home]["pts"] += 3
        elif result == "away":
            tbl[f.away]["pts"] += 3
        else:
            tbl[f.home]["pts"] += 1; tbl[f.away]["pts"] += 1

    # Rank within each group (pts -> GD -> GF; random tiebreak as fallback)
    standings: dict[str, list[tuple[str, int, int, int]]] = {}
    finish: dict[str, int] = {}
    for g, teams in grouping.items():
        ranked = sorted(
            teams,
            key=lambda t: (tbl[t]["pts"], tbl[t]["gf"] - tbl[t]["ga"], tbl[t]["gf"], rng.random()),
            reverse=True,
        )
        standings[g] = [(t, tbl[t]["pts"], tbl[t]["gf"] - tbl[t]["ga"], tbl[t]["gf"]) for t in ranked]
        for idx, team in enumerate(ranked, start=1):
            finish[team] = idx

    # Reached: group exit = 3, R32 = 4, R16 = 5, QF = 6, SF = 7, Final = 8, Champion = 9
    reached: dict[str, int] = {t: 3 for g in grouping.values() for t in g}

    # Top-2 per group advance
    advancers: list[str] = []
    third_place: list[tuple[str, int, int, int, str]] = []
    for g, rows in standings.items():
        advancers.append(rows[0][0]); reached[rows[0][0]] = 4
        advancers.append(rows[1][0]); reached[rows[1][0]] = 4
        third_place.append((*rows[2], g))

    # 8 best 3rd-placed
    third_ranked = sorted(third_place, key=lambda r: (r[1], r[2], r[3], rng.random()), reverse=True)
    for row in third_ranked[:N_3RD_PLACE_QUALIFIERS]:
        advancers.append(row[0]); reached[row[0]] = 4

    # ---- knockout rounds (single-elimination from 32 down) ----
    # Pairing: shuffled — the actual bracket pairs by group letters but for
    # championship-probability estimation, random pairing is a fair approximation.
    rng.shuffle(advancers)

    def _ko_winner(home: str, away: str) -> str:
        h_elo = elo.get(home, 1500); a_elo = elo.get(away, 1500)
        p_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400.0))
        # Slight extra-time / shootout shrinkage of advantage
        return home if rng.random() < p_home else away

    for round_label, reach_idx in [("R16", 5), ("QF", 6), ("SF", 7), ("Final", 8)]:
        winners = []
        for i in range(0, len(advancers), 2):
            w = _ko_winner(advancers[i], advancers[i+1])
            reached[w] = reach_idx
            winners.append(w)
        advancers = winners
    if advancers:
        reached[advancers[0]] = 9

    return reached, finish


def main() -> int:
    print(f"[05_simulate] loading model + features ...")
    model = _load_model()
    print(f"[05_simulate] running pre-match predictions for all 72 group matches ...")
    group_probs = _predict_group_probs(model)
    played = _load_played_results()
    print(f"  matches already played: {len(played)} / 72")

    rng = np.random.default_rng(RNG_SEED)
    # team -> Counter of reached-round
    histogram: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # team -> Counter of group-stage finish position (1..4)
    group_finish: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    t0 = time.time()
    for i in range(N_SIMS):
        reached, gf = _simulate_one(rng, group_probs, played)
        for team, rnd in reached.items():
            histogram[team][rnd] += 1
        for team, pos in gf.items():
            group_finish[team][pos] += 1
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  sim {i+1}/{N_SIMS}  ({elapsed:.1f}s)")

    rows = []
    all_teams = [t for g in wc2026_schedule.GROUPS.values() for t in g]
    for team in all_teams:
        h = histogram[team]
        total = sum(h.values())
        # Survival = P(reached >= round)
        def p_at_least(r): return sum(c for rr, c in h.items() if rr >= r) / total if total else 0.0
        gf = group_finish[team]
        gf_total = sum(gf.values()) or 1
        # Expected number of matches played: group=3 + KO rounds reached
        # (R32=+1, R16=+1, QF=+1, SF=+1, Final=+1) → max 8.
        e_matches = 3.0 + sum(
            (c / total if total else 0.0)
            for rr, c in h.items() if rr >= 4
        ) + sum(
            (c / total if total else 0.0) * (rr - 4)
            for rr, c in h.items() if rr >= 5
        )
        rows.append({
            "team": team,
            "group": next(g for g, ts in wc2026_schedule.GROUPS.items() if team in ts),
            "p_advance_R32":   round(p_at_least(4), 4),
            "p_advance_R16":   round(p_at_least(5), 4),
            "p_advance_QF":    round(p_at_least(6), 4),
            "p_advance_SF":    round(p_at_least(7), 4),
            "p_advance_Final": round(p_at_least(8), 4),
            "p_champion":      round(p_at_least(9), 4),
            "p_1st_in_group":  round(gf.get(1, 0) / gf_total, 4),
            "p_2nd_in_group":  round(gf.get(2, 0) / gf_total, 4),
            "p_3rd_in_group":  round(gf.get(3, 0) / gf_total, 4),
            "p_4th_in_group":  round(gf.get(4, 0) / gf_total, 4),
            "e_matches_played": round(e_matches, 3),
        })
    out = pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
    out.to_csv(DATA_DIR / "tournament_sim_results.csv", index=False)
    print(f"\n[05_simulate] wrote {DATA_DIR / 'tournament_sim_results.csv'}")
    print("\nTop 10 by P(champion):")
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
