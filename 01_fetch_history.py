"""
Step 1 - Fetch historical international football results.

Source: https://github.com/martj42/international_results (daily-updated, ~49k matches 1872 -> today).

Pulls results.csv, shootouts.csv, goalscorers.csv from the GitHub raw mirror,
normalises team names (Germany / West Germany merge, USSR -> Russia, etc.),
and writes data/matches.csv ready for Elo + feature engineering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = {
    "results.csv": f"{BASE}/results.csv",
    "shootouts.csv": f"{BASE}/shootouts.csv",
    "goalscorers.csv": f"{BASE}/goalscorers.csv",
}

# Successor-state mappings. Pre-dissolution rows keep their original label so
# the Elo time-series stays causal; we add a `team_canonical` column that maps
# legacy entities to their modern successor for current-day lookups.
SUCCESSOR_MAP = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Soviet Union": "Russia",
    "CIS": "Russia",
    "Zaire": "DR Congo",
    "Republic of Ireland": "Ireland",
}

# Tournament-class mapping for Elo K-factor (used downstream in 02_compute_elo.py).
# Coarse buckets matching eloratings.net's K=60/50/40/30/20 ladder.
TOURNAMENT_CLASS = {
    "FIFA World Cup": "wc_finals",
    "Copa America": "continental_finals",
    "UEFA Euro": "continental_finals",
    "African Cup of Nations": "continental_finals",
    "AFC Asian Cup": "continental_finals",
    "CONCACAF Championship": "continental_finals",
    "Gold Cup": "continental_finals",
    "Confederations Cup": "continental_finals",
    "FIFA World Cup qualification": "qualifier",
    "UEFA Euro qualification": "qualifier",
    "Copa America qualification": "qualifier",
    "African Cup of Nations qualification": "qualifier",
    "AFC Asian Cup qualification": "qualifier",
    "UEFA Nations League": "minor_tourney",
    "CONCACAF Nations League": "minor_tourney",
    "Friendly": "friendly",
}


def classify_tournament(name: str) -> str:
    # Check qualifiers / longer-prefix entries first to avoid "FIFA World Cup"
    # swallowing "FIFA World Cup qualification".
    for prefix, klass in sorted(TOURNAMENT_CLASS.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(prefix):
            return klass
    return "minor_tourney"


def download(url: str, dest: Path) -> None:
    print(f"  fetching {url} ...", end=" ", flush=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"{len(r.content)//1024} KB")


def fetch_all() -> dict[str, pd.DataFrame]:
    out = {}
    for name, url in FILES.items():
        dest = RAW_DIR / name
        download(url, dest)
        out[name] = pd.read_csv(dest)
    return out


def build_matches(results: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Canonical labels for modern lookups (don't rewrite the historical rows).
    df["home_canonical"] = df["home_team"].replace(SUCCESSOR_MAP)
    df["away_canonical"] = df["away_team"].replace(SUCCESSOR_MAP)

    df["tournament_class"] = df["tournament"].map(classify_tournament)
    df["is_friendly"] = df["tournament_class"].eq("friendly")
    df["is_wc"] = df["tournament"].str.startswith("FIFA World Cup", na=False) & ~df[
        "tournament"
    ].str.contains("qualification", na=False)

    # Result columns.
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["result"] = "draw"
    df.loc[df["goal_diff"] > 0, "result"] = "home_win"
    df.loc[df["goal_diff"] < 0, "result"] = "away_win"

    # Merge shootout winners (knockout-stage ties only).
    if not shootouts.empty:
        shootouts = shootouts.copy()
        shootouts["date"] = pd.to_datetime(shootouts["date"])
        shootouts = shootouts.rename(columns={"winner": "shootout_winner"})
        df = df.merge(
            shootouts[["date", "home_team", "away_team", "shootout_winner"]],
            on=["date", "home_team", "away_team"],
            how="left",
        )
    else:
        df["shootout_winner"] = pd.NA

    return df


def main() -> int:
    print(f"[01_fetch_history] downloading from {BASE} ...")
    files = fetch_all()

    matches = build_matches(files["results.csv"], files["shootouts.csv"])
    out_path = DATA_DIR / "matches.csv"
    matches.to_csv(out_path, index=False)

    print(f"\n[01_fetch_history] wrote {out_path}")
    print(f"  rows           : {len(matches):,}")
    print(f"  date range     : {matches['date'].min().date()} -> {matches['date'].max().date()}")
    print(f"  unique teams   : {matches['home_canonical'].nunique()}")
    print(f"  WC finals rows : {matches['is_wc'].sum()}")
    print(f"  shootouts      : {matches['shootout_winner'].notna().sum()}")
    print(f"  tournament classes:")
    for klass, n in matches["tournament_class"].value_counts().items():
        print(f"    {klass:25s} {n:,}")

    print(f"\n  result distribution:")
    for res, n in matches["result"].value_counts().items():
        print(f"    {res:10s} {n:,} ({100*n/len(matches):.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
