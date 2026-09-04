"""
Build data/odds_2026.csv from kicktipp.de Oddset closing odds.

The user provided 6 matchday screenshots showing Oddset pre-match closing
odds (decimal format, "1 / X / 2"). We hand-transcribed them here so the
tuner has authoritative historical odds for the 48 matches of WC2026.

Each row gets:
  - match_no (mapped from wc2026_fifa_schedule.csv by date + team names)
  - home_odds, draw_odds, away_odds (raw decimal from Oddset)
  - implied_home/draw/away (de-vigged probabilities — overround removed by
    dividing each raw 1/odds by the sum of all three)
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from datetime import timedelta

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# Date in DD.MM.YY, kickoff in HH:MM Berlin local time (CEST = UTC+2). We map
# match_no by date + team names so the local kickoff is fine.
ROWS_RAW = [
    # Matchday 1
    ("11.06.26", "Mexico", "South Africa", 1.43, 4.40, 7.75),
    ("12.06.26", "South Korea", "Czech Republic", 2.55, 3.10, 2.95),
    ("12.06.26", "Canada", "Bosnia and Herzegovina", 1.82, 3.40, 4.60),
    ("13.06.26", "USA", "Paraguay", 2.00, 3.25, 3.90),
    ("13.06.26", "Qatar", "Switzerland", 14.5, 7.50, 1.19),
    ("14.06.26", "Brazil", "Morocco", 1.66, 3.75, 5.25),
    ("14.06.26", "Haiti", "Scotland", 4.75, 4.33, 1.63),
    ("14.06.26", "Australia", "Turkey", 5.25, 3.70, 1.70),
    # Matchday 2
    ("14.06.26", "Germany", "Curacao", 1.05, 17.0, 51.0),
    ("14.06.26", "Netherlands", "Japan", 2.00, 3.50, 3.70),
    ("15.06.26", "Ivory Coast", "Ecuador", 3.25, 2.90, 2.45),
    ("15.06.26", "Sweden", "Tunisia", 1.87, 3.40, 4.50),
    ("15.06.26", "Spain", "Cape Verde", 1.10, 10.5, 23.0),
    ("15.06.26", "Belgium", "Egypt", 1.53, 4.20, 6.00),
    ("16.06.26", "Saudi Arabia", "Uruguay", 7.00, 4.10, 1.49),
    ("16.06.26", "Iran", "New Zealand", 1.80, 3.40, 4.80),
    # Matchday 3
    ("16.06.26", "France", "Senegal", 1.51, 4.50, 5.75),
    ("17.06.26", "Iraq", "Norway", 13.0, 6.75, 1.21),
    ("17.06.26", "Argentina", "Algeria", 1.55, 4.20, 5.75),
    ("17.06.26", "Austria", "Jordan", 1.39, 4.80, 7.75),
    ("17.06.26", "Portugal", "DR Congo", 1.29, 5.50, 10.5),
    ("17.06.26", "England", "Croatia", 1.68, 3.80, 5.00),
    ("18.06.26", "Ghana", "Panama", 2.30, 3.20, 3.20),
    ("18.06.26", "Uzbekistan", "Colombia", 9.25, 5.00, 1.34),
    # Matchday 4
    ("18.06.26", "Czech Republic", "South Africa", 1.87, 3.40, 4.33),
    ("18.06.26", "Switzerland", "Bosnia and Herzegovina", 1.53, 4.20, 6.00),
    ("19.06.26", "Canada", "Qatar", 1.25, 6.25, 11.5),
    ("19.06.26", "Mexico", "South Korea", 2.05, 3.10, 4.00),
    ("19.06.26", "USA", "Australia", 1.57, 4.00, 6.00),
    ("20.06.26", "Scotland", "Morocco", 5.25, 3.60, 1.68),
    ("20.06.26", "Brazil", "Haiti", 1.10, 11.0, 23.0),
    ("20.06.26", "Turkey", "Paraguay", 2.05, 3.30, 3.75),
    # Matchday 5
    ("20.06.26", "Netherlands", "Sweden", 1.68, 4.00, 4.60),
    ("20.06.26", "Germany", "Ivory Coast", 1.49, 4.75, 5.75),
    ("21.06.26", "Ecuador", "Curacao", 1.15, 8.75, 15.5),
    ("21.06.26", "Tunisia", "Japan", 7.50, 4.33, 1.46),
    ("21.06.26", "Spain", "Saudi Arabia", 1.11, 11.0, 20.0),
    ("21.06.26", "Belgium", "Iran", 1.42, 4.75, 7.25),
    ("22.06.26", "Uruguay", "Cape Verde", 1.40, 4.60, 8.50),
    ("22.06.26", "New Zealand", "Egypt", 5.50, 3.75, 1.65),
    # Matchday 6
    ("22.06.26", "Argentina", "Austria", 1.47, 4.40, 6.75),
    ("22.06.26", "France", "Iraq", 1.09, 12.0, 26.0),
    ("23.06.26", "Norway", "Senegal", 2.40, 3.40, 2.80),
    ("23.06.26", "Jordan", "Algeria", 7.00, 4.33, 1.47),
    ("23.06.26", "Portugal", "Uzbekistan", 1.15, 8.75, 16.5),
    ("23.06.26", "England", "Ghana", 1.19, 7.25, 15.0),
    ("24.06.26", "Panama", "Croatia", 6.50, 4.33, 1.50),
    ("24.06.26", "Colombia", "DR Congo", 1.53, 4.00, 6.50),
]

# Name normalisation map: Oddset/kicktipp -> FIFA schedule canonical name.
NAME_MAP = {
    "USA": "United States",
    "Turkey": "Turkey",         # FIFA uses Turkey too
    "Curacao": "Curacao",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
}


def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return s.strip().lower()


def main() -> int:
    sched = pd.read_csv(DATA / "wc2026_fifa_schedule.csv")
    sched["date_only"] = sched["date_utc"].str[:10]
    sched["home_n"] = sched["home"].map(norm)
    sched["away_n"] = sched["away"].map(norm)

    out_rows = []
    unmatched = []
    for date_de, home, away, ho, do, ao in ROWS_RAW:
        # Parse Berlin local date (DD.MM.YY -> YYYY-MM-DD).
        d_obj = pd.to_datetime(date_de, format="%d.%m.%y")
        canon_h = norm(NAME_MAP.get(home, home))
        canon_a = norm(NAME_MAP.get(away, away))

        # Try date (Berlin local), plus +/-1 day for UTC-rollover matches.
        hit = None
        for delta in (0, 1, -1):
            d = (d_obj + timedelta(days=delta)).strftime("%Y-%m-%d")
            cand = sched[sched["date_only"] == d]
            same = cand[
                ((cand["home_n"] == canon_h) & (cand["away_n"] == canon_a))
                | ((cand["home_n"] == canon_a) & (cand["away_n"] == canon_h))
            ]
            if not same.empty:
                hit = same.iloc[0]
                break
        if hit is None:
            unmatched.append((date_de, home, away))
            continue

        # If kicktipp listed teams swapped vs the FIFA schedule (rare), flip odds.
        if norm(hit["home"]) == canon_h:
            mh, md, ma = ho, do, ao
        else:
            mh, md, ma = ao, do, ho

        # De-vig: implied prob = (1/odds) / sum(1/odds)
        ih, id_, ia = 1.0 / mh, 1.0 / md, 1.0 / ma
        s = ih + id_ + ia
        out_rows.append({
            "match_no": int(hit["match_no"]),
            "date": hit["date_only"],
            "home": hit["home"],
            "away": hit["away"],
            "home_odds": round(mh, 3),
            "draw_odds": round(md, 3),
            "away_odds": round(ma, 3),
            "implied_home": round(ih / s, 4),
            "implied_draw": round(id_ / s, 4),
            "implied_away": round(ia / s, 4),
            "overround": round(s, 4),  # 1.0 = no margin; typically 1.05-1.10
        })

    if unmatched:
        print(f"[odds_csv] UNMATCHED ({len(unmatched)}):")
        for u in unmatched:
            print(f"  {u}")

    df = pd.DataFrame(out_rows).drop_duplicates(subset=["match_no"]).sort_values("match_no")
    out = DATA / "odds_2026.csv"
    df.to_csv(out, index=False)
    print(f"\n[odds_csv] wrote {out}  ({len(df)} matches)")
    print(f"  avg overround: {df['overround'].mean():.4f}  (Oddset's bookie margin)")
    print(f"  match_no range: {df['match_no'].min()} - {df['match_no'].max()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
