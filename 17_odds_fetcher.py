"""
Live odds fetcher for upcoming WC2026 matches.

Pulls H/D/A odds from the-odds-api.com (free tier, 500 calls/month) across
every available bookmaker for upcoming WC matches. Computes the MEDIAN odds
per outcome — more robust than picking a single bookie, and within 1-2pp of
what Bet365 would price (which isn't on the-odds-api directly).

Run order:
  1. fetch from API
  2. dedupe per match (median across bookies)
  3. match to our wc2026_fifa_schedule.csv by date + teams
  4. de-vig: implied_probs = (1/odds) / sum(1/odds)
  5. merge into data/odds_2026.csv (preserves manually-pasted historical rows)
"""
from __future__ import annotations

import os
import statistics
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DATA = Path(__file__).parent / "data"
API_BASE = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"


def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.strip().lower()


# Bookie team-name -> FIFA canonical (handles a few common mismatches).
TEAM_MAP = {
    "usa": "united states",
    "south korea": "korea republic",  # the-odds-api uses "South Korea", FIFA "Korea Republic" — adjust if needed
    "cape verde": "cabo verde",
    "dr congo": "congo dr",
    "turkiye": "turkey",
    "curacao": "curacao",
}


def _canon(team: str) -> str:
    n = norm(team)
    return TEAM_MAP.get(n, n)


def fetch_live_odds(api_key: str) -> list[dict]:
    """Pull current WC odds from the-odds-api. Returns one row per match
    with median home/draw/away odds across all bookies."""
    r = requests.get(
        API_BASE,
        params={"apiKey": api_key, "regions": "eu,uk,us", "markets": "h2h"},
        timeout=20,
    )
    r.raise_for_status()
    matches = r.json()
    print(f"[odds_fetcher] pulled {len(matches)} upcoming matches "
          f"(remaining: {r.headers.get('x-requests-remaining')})")

    rows = []
    for m in matches:
        home = m["home_team"]
        away = m["away_team"]
        commence = m.get("commence_time", "")
        home_odds_list = []
        draw_odds_list = []
        away_odds_list = []
        n_bookies = 0
        for bk in m.get("bookmakers", []):
            markets = bk.get("markets", [])
            if not markets:
                continue
            outcomes = markets[0].get("outcomes", [])
            d = {o["name"]: o["price"] for o in outcomes}
            ho = d.get(home); ao = d.get(away); do = d.get("Draw")
            if ho and do and ao:
                home_odds_list.append(ho)
                draw_odds_list.append(do)
                away_odds_list.append(ao)
                n_bookies += 1
        if n_bookies == 0:
            continue
        mh = statistics.median(home_odds_list)
        md = statistics.median(draw_odds_list)
        ma = statistics.median(away_odds_list)
        # de-vig
        ih, id_, ia = 1.0 / mh, 1.0 / md, 1.0 / ma
        s = ih + id_ + ia
        rows.append({
            "home": home, "away": away, "commence_time": commence,
            "home_odds": round(mh, 3), "draw_odds": round(md, 3), "away_odds": round(ma, 3),
            "implied_home": round(ih / s, 4),
            "implied_draw": round(id_ / s, 4),
            "implied_away": round(ia / s, 4),
            "overround": round(s, 4),
            "n_bookies": n_bookies,
        })
    return rows


def match_to_schedule(rows: list[dict]) -> pd.DataFrame:
    """Join API rows to wc2026_fifa_schedule.csv by date (+/-1 day) + teams."""
    sched = pd.read_csv(DATA / "wc2026_fifa_schedule.csv")
    sched["date_only"] = sched["date_utc"].str[:10]
    sched["home_n"] = sched["home"].map(_canon)
    sched["away_n"] = sched["away"].map(_canon)

    out_rows = []
    unmatched = []
    for r in rows:
        hN, aN = _canon(r["home"]), _canon(r["away"])
        d_obj = pd.to_datetime(r["commence_time"][:10])
        hit = None
        for delta in (0, 1, -1):
            d = (d_obj + timedelta(days=delta)).strftime("%Y-%m-%d")
            cand = sched[sched["date_only"] == d]
            same = cand[
                ((cand["home_n"] == hN) & (cand["away_n"] == aN))
                | ((cand["home_n"] == aN) & (cand["away_n"] == hN))
            ]
            if not same.empty:
                hit = same.iloc[0]
                break
        if hit is None:
            unmatched.append((r["commence_time"][:10], r["home"], r["away"]))
            continue
        # Flip odds if home/away are swapped vs the FIFA schedule.
        if _canon(hit["home"]) == hN:
            mh, md, ma = r["home_odds"], r["draw_odds"], r["away_odds"]
            ih, id_, ia = r["implied_home"], r["implied_draw"], r["implied_away"]
        else:
            mh, md, ma = r["away_odds"], r["draw_odds"], r["home_odds"]
            ih, id_, ia = r["implied_away"], r["implied_draw"], r["implied_home"]
        out_rows.append({
            "match_no": int(hit["match_no"]),
            "date": hit["date_only"],
            "home": hit["home"], "away": hit["away"],
            "home_odds": mh, "draw_odds": md, "away_odds": ma,
            "implied_home": ih, "implied_draw": id_, "implied_away": ia,
            "overround": r["overround"],
        })

    if unmatched:
        print(f"[odds_fetcher] unmatched: {len(unmatched)}")
        for u in unmatched[:5]:
            print(f"  {u}")
    return pd.DataFrame(out_rows).drop_duplicates(subset=["match_no"]).sort_values("match_no")


def main() -> int:
    key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not key:
        print("[odds_fetcher] THE_ODDS_API_KEY not set; skipping live odds refresh")
        return 0

    try:
        rows = fetch_live_odds(key)
    except Exception as e:
        print(f"[odds_fetcher] API call failed: {e} — keeping existing odds_2026.csv")
        return 0

    new_df = match_to_schedule(rows)
    print(f"[odds_fetcher] matched {len(new_df)} upcoming matches to schedule")

    # Merge with existing data/odds_2026.csv (preserve historical/manual rows,
    # overwrite any rows where we now have fresher live odds).
    out_path = DATA / "odds_2026.csv"
    if out_path.exists():
        old = pd.read_csv(out_path)
        # Drop rows that we have fresher data for
        old = old[~old["match_no"].isin(new_df["match_no"])]
        merged = pd.concat([old, new_df], ignore_index=True).sort_values("match_no")
    else:
        merged = new_df

    merged.to_csv(out_path, index=False)
    print(f"[odds_fetcher] wrote {out_path} ({len(merged)} total rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
