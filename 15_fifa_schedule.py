"""
Pull the authoritative WC2026 schedule, stadiums and referee assignments from FIFA's
v3 API. This REPLACES the hand-curated ESPN-derived wc2026_schedule._GROUP_RAW
as the primary source. Run before tournament start and re-run nightly to pick up
referee appointments + any time changes.

Outputs:
  data/wc2026_fifa_schedule.csv  - 104 rows; columns:
      match_no, id_match, id_stage, id_group, date_utc, group_label, stage,
      home, away, home_id, away_id, home_country, away_country,
      stadium_id, stadium_name, referee, referee_country
  data/wc2026_fifa_referees.csv   - one row per referee assignment (refreshed)
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fifa_api import FifaApi

DATA_DIR = Path(__file__).parent / "data"

# FIFA -> canonical (martj42-aligned) team-name overrides.
TEAM_NAME_MAP = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czech Republic",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Curaçao": "Curacao",
    "DR Congo": "DR Congo",
    "Cape Verde": "Cape Verde",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "Iran",
}


def canonical_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def stage_of(stage_label: str | None, group_label: str | None) -> str:
    """Map FIFA stage_label -> our internal stage enum (group/R32/R16/QF/SF/3rd/Final)."""
    if group_label:
        return "group"
    if not stage_label:
        return "?"
    s = stage_label.lower()
    if "round of 32" in s or "first ko" in s or "round32" in s:
        return "R32"
    if "round of 16" in s or "round16" in s:
        return "R16"
    if "quarter" in s:
        return "QF"
    if "semi" in s:
        return "SF"
    if "3rd" in s or "third" in s or "play-off" in s:
        return "3rd"
    if "final" in s:
        return "Final"
    return s


async def main() -> int:
    api = FifaApi()
    try:
        matches = await api.wc2026_matches()
    finally:
        await api.close()

    if not matches:
        print("[15_fifa_schedule] FIFA API returned 0 matches", file=sys.stderr)
        return 1

    rows = []
    for i, m in enumerate(matches, start=1):
        rows.append({
            "match_no": i,                       # 1..104 in chronological order
            "id_match": m.id_match,
            "id_stage": m.id_stage,
            "id_group": m.id_group,
            "date_utc": m.date_utc.isoformat(),
            "stage": stage_of(m.stage_label, m.group_label),
            "group_label": m.group_label.replace("Group ", "").strip() if m.group_label else None,
            "match_number_official": m.match_number,
            "home": canonical_team(m.home_team),
            "away": canonical_team(m.away_team),
            "home_id": m.home_id, "away_id": m.away_id,
            "home_country": m.home_country, "away_country": m.away_country,
            "stadium_id": m.stadium_id,
            "stadium_name": m.stadium_name,
            "referee": m.referee,
            "referee_country": m.referee_country,
        })
    df = pd.DataFrame(rows)
    out = DATA_DIR / "wc2026_fifa_schedule.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[15_fifa_schedule] wrote {out}  ({len(df)} matches)")

    print(f"\n  refs assigned: {df['referee'].notna().sum()} / {len(df)}")
    print(f"  stadiums covered: {df['stadium_name'].nunique()} distinct")
    print(f"  group-stage matches: {(df['stage']=='group').sum()}")
    print(f"  knockout matches  : {(df['stage']!='group').sum()}")

    # Per-referee summary (useful for /referees command + ref-bias seeding).
    refs = (
        df.dropna(subset=["referee"])
          .groupby(["referee", "referee_country"], as_index=False)
          .size()
          .rename(columns={"size": "n_matches"})
          .sort_values("n_matches", ascending=False)
    )
    refs.to_csv(DATA_DIR / "wc2026_fifa_referees.csv", index=False)
    print(f"\n  distinct referees: {len(refs)}")
    print("  top 10:")
    print(refs.head(10).to_string(index=False))

    # Sanity: confirm both first match (Mexico v South Africa) and last (Final).
    print("\n  first match :", df.iloc[0][["date_utc", "home", "away", "stadium_name", "referee"]].to_dict())
    print("  last match  :", df.iloc[-1][["date_utc", "home", "away", "stadium_name", "stage"]].to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
