"""
Step 2 - Build Elo time-series using eloratings.net methodology.

K-factors vary by competition (WC final = 60, friendly = 20). Home advantage adds
+100 to the home rating when computing the expected score (unless neutral=True).
Goal difference inflates the rating swing: G=1 for 1-goal wins, 1.5 for 2-goal,
(11+gd)/8 for 3-goal+.

Outputs:
  data/team_elo_history.csv   long-form (date, team, rating_before, rating_after)
  data/team_elo_current.csv   latest rating per team
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0

K_BY_CLASS = {
    "wc_finals": 60.0,
    "continental_finals": 50.0,
    "qualifier": 40.0,
    "minor_tourney": 30.0,
    "friendly": 20.0,
}


def goal_diff_multiplier(gd: int) -> float:
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load the FULL matches.csv and preserve all rows (including unplayed future
    # fixtures). We compute Elo on the played subset only, but merge the Elo
    # columns back into the full frame so nothing is deleted from matches.csv.
    full = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"], low_memory=False)
    full["_orig_index"] = full.index
    matches = full.dropna(subset=["home_score", "away_score"]).copy()
    matches = matches.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = defaultdict(lambda: INITIAL_RATING)
    history_rows: list[dict] = []
    home_elo_col: list[float] = []
    away_elo_col: list[float] = []
    home_elo_after: list[float] = []
    away_elo_after: list[float] = []

    for row in tqdm(matches.itertuples(index=False), total=len(matches), desc="elo"):
        home, away = row.home_team, row.away_team
        is_neutral = bool(row.neutral)
        k = K_BY_CLASS.get(row.tournament_class, 30.0)
        gd = int(row.home_score - row.away_score)
        g = goal_diff_multiplier(gd)

        r_home = ratings[home]
        r_away = ratings[away]

        eff_home = r_home + (0.0 if is_neutral else HOME_ADVANTAGE)
        exp_home = expected_score(eff_home, r_away)

        if gd > 0:
            res_home = 1.0
        elif gd < 0:
            res_home = 0.0
        else:
            res_home = 0.5

        delta = k * g * (res_home - exp_home)

        new_home = r_home + delta
        new_away = r_away - delta

        home_elo_col.append(r_home)
        away_elo_col.append(r_away)
        home_elo_after.append(new_home)
        away_elo_after.append(new_away)

        history_rows.append(
            {"date": row.date, "team": home, "rating_before": r_home, "rating_after": new_home}
        )
        history_rows.append(
            {"date": row.date, "team": away, "rating_before": r_away, "rating_after": new_away}
        )

        ratings[home] = new_home
        ratings[away] = new_away

    matches["home_elo_before"] = home_elo_col
    matches["away_elo_before"] = away_elo_col
    matches["home_elo_after"] = home_elo_after
    matches["away_elo_after"] = away_elo_after

    # Merge Elo columns back into the ORIGINAL full frame (which still contains
    # unplayed future fixtures). This prevents matches.csv from shrinking on
    # every retrain — those rows would otherwise be lost forever.
    elo_cols = ["_orig_index", "home_elo_before", "away_elo_before",
                "home_elo_after", "away_elo_after"]
    merged = full.merge(matches[elo_cols], on="_orig_index", how="left",
                        suffixes=("_old", ""))
    for c in ("home_elo_before", "away_elo_before", "home_elo_after", "away_elo_after"):
        old = c + "_old"
        if old in merged.columns:
            merged.drop(columns=[old], inplace=True)
    merged.drop(columns=["_orig_index"], inplace=True)
    merged.to_csv(DATA_DIR / "matches.csv", index=False)

    history = pd.DataFrame(history_rows)
    current = (
        history.sort_values("date")
        .groupby("team", as_index=False)
        .tail(1)
        .rename(columns={"rating_after": "rating"})
        [["team", "date", "rating"]]
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )

    history.to_csv(DATA_DIR / "team_elo_history.csv", index=False)
    current.to_csv(DATA_DIR / "team_elo_current.csv", index=False)
    return history, current


def main() -> int:
    history, current = compute()
    print("\n[02_compute_elo] top 30 by current Elo:")
    print(current.head(30).to_string(index=False))
    print(f"\n  total teams tracked: {len(current)}")
    print(f"  history rows       : {len(history):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
