"""
Confirmed-XI lineup fetcher with fallback chain.

Sources (tried in order until one succeeds):
  1. Manual override file (data/lineups_manual.json) — populated by the bot's
     /lineup command. Highest priority because the user pasted it directly.
  2. FIFA v3 API (/api/v3/live/football/...) — AUTHORITATIVE. Returns starters
     with Status=1 ~75min before kickoff. Includes referee + formation coords.
  3. API-Football v3 (RapidAPI) — needs RAPIDAPI_KEY env var; backup.
  4. FotMob match-details JSON — last-resort backup.
  5. None — caller falls back to the full 26-man squad.

Returns a `MatchLineups` dataclass:
  home_starters / away_starters : list of player names matching squads_2026.csv
  home_subs / away_subs         : remaining squad members (best effort)
  referee                       : referee name or None
  source                        : which provider succeeded
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from fifa_api import FifaApi, WC_COMPETITION_ID, WC_SEASON_ID_2026

DATA_DIR = Path(__file__).parent / "data"
MANUAL_OVERRIDE_PATH = DATA_DIR / "lineups_manual.json"
FIFA_SCHEDULE_PATH = DATA_DIR / "wc2026_fifa_schedule.csv"


@dataclass
class MatchLineups:
    home_team: str
    away_team: str
    date: str
    home_starters: list[str] = field(default_factory=list)
    away_starters: list[str] = field(default_factory=list)
    home_subs: list[str] = field(default_factory=list)
    away_subs: list[str] = field(default_factory=list)
    referee: str | None = None
    source: str = "none"
    confirmed: bool = False     # True iff actual confirmed XI, False if probable


# --------------------------- manual override ---------------------------

def _load_manual_store() -> dict:
    if not MANUAL_OVERRIDE_PATH.exists():
        return {}
    try:
        return json.loads(MANUAL_OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manual_store(store: dict) -> None:
    MANUAL_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_OVERRIDE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _match_key(home: str, away: str, date: str) -> str:
    return f"{date}::{home}::{away}".lower().replace(" ", "_")


def set_manual_lineup(
    home: str, away: str, date: str,
    home_starters: list[str], away_starters: list[str],
    referee: str | None = None,
) -> None:
    """Called by the bot's /lineup command to record a user-supplied XI."""
    store = _load_manual_store()
    store[_match_key(home, away, date)] = {
        "home_team": home, "away_team": away, "date": date,
        "home_starters": home_starters,
        "away_starters": away_starters,
        "referee": referee,
        "set_at": datetime.utcnow().isoformat(),
    }
    _save_manual_store(store)


def _try_manual(home: str, away: str, date: str) -> MatchLineups | None:
    store = _load_manual_store()
    rec = store.get(_match_key(home, away, date))
    if not rec:
        return None
    return MatchLineups(
        home_team=home, away_team=away, date=date,
        home_starters=rec.get("home_starters", []),
        away_starters=rec.get("away_starters", []),
        referee=rec.get("referee"),
        source="manual", confirmed=True,
    )


# --------------------------- FIFA official ---------------------------

def _fifa_ids_for(home: str, away: str, date: str) -> tuple[str, str] | None:
    """Look up (id_stage, id_match) for the fixture from the FIFA schedule CSV."""
    if not FIFA_SCHEDULE_PATH.exists():
        return None
    df = pd.read_csv(FIFA_SCHEDULE_PATH, encoding="utf-8")
    df = df[df["date_utc"].astype(str).str.startswith(date)]
    if df.empty:
        return None
    hit = df[((df["home"] == home) & (df["away"] == away))
             | ((df["home"] == away) & (df["away"] == home))]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return str(row["id_stage"]), str(row["id_match"])


async def _try_fifa(home: str, away: str, date: str) -> MatchLineups | None:
    ids = _fifa_ids_for(home, away, date)
    if not ids:
        return None
    id_stage, id_match = ids
    api = FifaApi()
    try:
        lu = await api.lineup(WC_COMPETITION_ID, WC_SEASON_ID_2026, id_stage, id_match)
    finally:
        await api.close()
    if not lu:
        return None
    # Map FIFA's home-team-first ordering to the caller's requested order.
    # FIFA always returns the actual home team in HomeTeam. If the caller asked
    # the same way, perfect; if swapped (rare), we transpose.
    df = pd.read_csv(FIFA_SCHEDULE_PATH, encoding="utf-8")
    df = df[df["id_match"].astype(str) == id_match]
    fifa_home = (df.iloc[0]["home"] if not df.empty else home)
    swap = (fifa_home != home)
    hs = [p["name"] for p in (lu.away_starters if swap else lu.home_starters) if p.get("name")]
    as_ = [p["name"] for p in (lu.home_starters if swap else lu.away_starters) if p.get("name")]
    hsubs = [p["name"] for p in (lu.away_subs if swap else lu.home_subs) if p.get("name")]
    asubs = [p["name"] for p in (lu.home_subs if swap else lu.away_subs) if p.get("name")]
    if not (hs and as_):
        return None   # lineup not yet populated (pre-75min)
    return MatchLineups(
        home_team=home, away_team=away, date=date,
        home_starters=hs, away_starters=as_,
        home_subs=hsubs, away_subs=asubs,
        referee=lu.referee, source="fifa", confirmed=True,
    )


# --------------------------- API-Football ---------------------------

async def _try_api_football(home: str, away: str, date: str) -> MatchLineups | None:
    key = os.environ.get("RAPIDAPI_KEY") or os.environ.get("API_FOOTBALL_KEY")
    if not key:
        return None
    headers = {"x-rapidapi-host": "v3.football.api-sports.io",
               "x-rapidapi-key": key}
    async with httpx.AsyncClient(timeout=20.0) as c:
        # Step 1: find the fixture id for date + teams.
        r = await c.get(
            "https://v3.football.api-sports.io/fixtures",
            params={"date": date, "league": 1, "season": 2026},
            headers=headers,
        )
        if r.status_code != 200:
            return None
        fixtures = r.json().get("response", [])
        fid = None
        for fx in fixtures:
            t = fx.get("teams", {})
            h = (t.get("home", {}) or {}).get("name", "")
            a = (t.get("away", {}) or {}).get("name", "")
            if home.lower() in h.lower() or a.lower() in home.lower():
                fid = fx.get("fixture", {}).get("id")
                ref = fx.get("fixture", {}).get("referee")
                break
        if not fid:
            return None
        # Step 2: fetch lineups for that fixture.
        r = await c.get(
            "https://v3.football.api-sports.io/fixtures/lineups",
            params={"fixture": fid}, headers=headers,
        )
        if r.status_code != 200:
            return None
        teams_lineups = r.json().get("response", [])
        if len(teams_lineups) < 2:
            return None
    out = MatchLineups(home_team=home, away_team=away, date=date,
                       source="api_football", confirmed=True, referee=ref)
    for tl in teams_lineups:
        tname = (tl.get("team") or {}).get("name", "")
        starters = [p.get("player", {}).get("name") for p in tl.get("startXI", []) or []]
        subs = [p.get("player", {}).get("name") for p in tl.get("substitutes", []) or []]
        starters = [s for s in starters if s]
        subs = [s for s in subs if s]
        if home.lower() in tname.lower():
            out.home_starters = starters; out.home_subs = subs
        elif away.lower() in tname.lower():
            out.away_starters = starters; out.away_subs = subs
    return out if (out.home_starters and out.away_starters) else None


# --------------------------- FotMob ---------------------------

_FOTMOB_SEARCH = "https://www.fotmob.com/api/searchapi/suggest"
_FOTMOB_MATCH  = "https://www.fotmob.com/api/matchDetails"


async def _try_fotmob(home: str, away: str, date: str) -> MatchLineups | None:
    """Best-effort scrape of FotMob's public JSON. URL format shifts often; we
    catch all errors and return None on any failure."""
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "fifa-predictor/1.0"}) as c:
            r = await c.get(_FOTMOB_SEARCH, params={"term": f"{home} {away}"})
            if r.status_code != 200:
                return None
            hits = (r.json() or {}).get("matchSuggest", [])
            match_id = None
            for h in hits:
                if date in (h.get("matchDate") or ""):
                    match_id = h.get("id"); break
            if not match_id:
                return None
            r = await c.get(_FOTMOB_MATCH, params={"matchId": match_id})
            if r.status_code != 200:
                return None
            data = r.json() or {}
        lineup = data.get("content", {}).get("lineup") or data.get("lineup") or {}
        if not lineup:
            return None
        home_l = lineup.get("homeTeam") or lineup.get("home", {})
        away_l = lineup.get("awayTeam") or lineup.get("away", {})

        def collect(team_block) -> tuple[list[str], list[str]]:
            starters, subs = [], []
            for grp in (team_block.get("starters") or []):
                for p in grp:
                    n = p.get("name", {}).get("fullName") or p.get("name")
                    if n: starters.append(n)
            for p in team_block.get("subs") or []:
                n = p.get("name", {}).get("fullName") or p.get("name")
                if n: subs.append(n)
            return starters, subs

        hs, hsubs = collect(home_l)
        as_, asubs = collect(away_l)
        ref = (data.get("content", {}).get("matchFacts") or {}).get("infoBox", {}).get("Referee", {}).get("text")
        if not (hs and as_):
            return None
        return MatchLineups(
            home_team=home, away_team=away, date=date,
            home_starters=hs, away_starters=as_,
            home_subs=hsubs, away_subs=asubs,
            referee=ref, source="fotmob", confirmed=True,
        )
    except Exception:
        return None


# --------------------------- public ---------------------------

async def fetch_lineup(home: str, away: str, date: str) -> MatchLineups | None:
    """Try each source in priority order. Return None if all fail."""
    # Manual override has highest priority.
    try:
        m = _try_manual(home, away, date)
        if m is not None:
            return m
    except Exception:
        pass
    # FIFA's official API is the authoritative source for lineups.
    try:
        r = await _try_fifa(home, away, date)
        if r is not None:
            return r
    except Exception as e:
        print(f"[lineup_fetcher] FIFA path failed: {e}")
    # Backups for any odd FIFA outage.
    for coro in (_try_api_football(home, away, date), _try_fotmob(home, away, date)):
        try:
            r = await coro
        except Exception:
            r = None
        if r is not None:
            return r
    return None


# --------------------------- match player names to squad ---------------------------

def _normalise_name(n: str) -> str:
    n = re.sub(r"[^a-z ]+", "", n.lower())
    return re.sub(r"\s+", " ", n).strip()


def map_to_squad(team: str, lineup_names: list[str], squads_df: pd.DataFrame) -> list[dict]:
    """Match lineup names to squad rows. Returns squad rows (dict) for matched
    players in lineup order; unmatched names are skipped."""
    team_squad = squads_df[squads_df["team"] == team].copy()
    if team_squad.empty:
        return []
    # Build a normalized lookup. Try exact, then last-name match, then surname-startswith.
    by_full = {_normalise_name(p): r for _, r in team_squad.iterrows() for p in [r["player"]]}
    by_last = {}
    for _, r in team_squad.iterrows():
        parts = _normalise_name(r["player"]).split()
        if parts:
            by_last.setdefault(parts[-1], []).append(r)
    out = []
    for name in lineup_names:
        norm = _normalise_name(name)
        if norm in by_full:
            out.append(by_full[norm].to_dict()); continue
        last = norm.split()[-1] if norm.split() else ""
        if last in by_last and len(by_last[last]) == 1:
            out.append(by_last[last][0].to_dict()); continue
        # else: skip (unmatched)
    return out


# --------------------------- CLI smoke test ---------------------------

if __name__ == "__main__":
    import asyncio, sys
    if len(sys.argv) < 4:
        print("usage: python 13_lineup_fetcher.py <home> <away> <YYYY-MM-DD>")
        sys.exit(2)
    home, away, date = sys.argv[1], sys.argv[2], sys.argv[3]
    res = asyncio.run(fetch_lineup(home, away, date))
    if res is None:
        print("No lineup found from any source — will fall back to 26-man squad.")
    else:
        print(f"source={res.source}  referee={res.referee}")
        print(f"  {home} XI: {res.home_starters}")
        print(f"  {away} XI: {res.away_starters}")
