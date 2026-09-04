"""
Async client for FIFA's public v3 data API.

Discovered endpoints (no key required, en-GB locale default):
  /api/v3/calendar/matches?idCompetition=17&from=YYYY-MM-DD&count=N
      All matches across all seasons. Filter by Date prefix for a tournament.

  /api/v3/live/football/{idCompetition}/{idSeason}/{idStage}/{idMatch}
      Live + post-match payload. Includes:
        HomeTeam.Players[]   one row per registered player; Status=1 marks starters
        AwayTeam.Players[]   same
        Officials[]          OfficialType 1 = referee, 2/3 = ARs, 4 = 4th, 5 = VAR
        Stadium, Weather, MatchStatus, Period, BallPossession, Bookings,
        Goals, Substitutions, Tactics formation strings, LineupX / LineupY
        coordinates per player.

  /api/v3/players/{idPlayer}
      Player profile: birth date, height, weight, preferred foot,
      InternationalCaps, InternationalDebut, Goals.

  /api/v3/teams/{idTeam}                team profile
  /api/v3/teams/association/{country}   teams owned by an association

WC2026 constants:
  idCompetition = 17       (FIFA World Cup)
  idSeason      = 285023   (2026 edition)
  idStage       = 289273   (group / first stage)

These come from inspecting /api/v3/calendar/matches?idCompetition=17&from=2026-01-01.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

BASE = "https://api.fifa.com/api/v3"
WC_COMPETITION_ID = "17"
WC_SEASON_ID_2026 = "285023"


def _en(field) -> str | None:
    """Pull the en-GB description from a FIFA i18n list."""
    if not field:
        return None
    if isinstance(field, list):
        for entry in field:
            if entry.get("Locale", "").lower().startswith("en"):
                return entry.get("Description")
        return field[0].get("Description") if field else None
    return field


@dataclass
class FifaMatch:
    id_match: str
    id_stage: str
    id_group: str | None
    date_utc: datetime
    home_team: str
    home_id: str
    home_country: str
    away_team: str
    away_id: str
    away_country: str
    stadium_id: str | None
    stadium_name: str | None
    group_label: str | None
    stage_label: str | None
    match_number: int | None
    referee: str | None        # OfficialType=1
    referee_country: str | None
    raw: dict[str, Any]


@dataclass
class FifaLineup:
    id_match: str
    home_starters: list[dict]   # each: {id, name, shirt, position, captain, x, y}
    home_subs: list[dict]
    away_starters: list[dict]
    away_subs: list[dict]
    referee: str | None
    referee_country: str | None
    officials: list[dict]
    match_status: int | None
    raw: dict[str, Any]


class FifaApi:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            timeout=20.0,
            headers={"User-Agent": "fifa-predictor/1.0", "Accept": "application/json"},
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---- raw GET helper ----
    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        try:
            r = await self._client.get(f"{BASE}{path}", params=params or {})
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except (json.JSONDecodeError, ValueError):
            return None

    # ---- WC2026 schedule ----
    async def wc2026_matches(self) -> list[FifaMatch]:
        d = await self._get(
            "/calendar/matches",
            {"idCompetition": WC_COMPETITION_ID, "from": "2026-01-01", "count": "200"},
        )
        if not d:
            return []
        out: list[FifaMatch] = []
        for m in d.get("Results", []):
            if str(m.get("IdSeason")) != WC_SEASON_ID_2026:
                continue
            date = m.get("Date")
            if not date:
                continue
            home = m.get("Home") or {}
            away = m.get("Away") or {}
            stadium = m.get("Stadium") or {}
            ref = None
            ref_country = None
            for o in m.get("Officials") or []:
                if o.get("OfficialType") == 1:
                    ref = _en(o.get("NameShort")) or _en(o.get("Name"))
                    ref_country = o.get("IdCountry")
                    break
            out.append(FifaMatch(
                id_match=str(m.get("IdMatch")),
                id_stage=str(m.get("IdStage")),
                id_group=str(m.get("IdGroup")) if m.get("IdGroup") else None,
                date_utc=datetime.fromisoformat(date.replace("Z", "+00:00")),
                home_team=_en(home.get("TeamName")) or "",
                home_id=str(home.get("IdTeam") or ""),
                home_country=home.get("IdCountry") or "",
                away_team=_en(away.get("TeamName")) or "",
                away_id=str(away.get("IdTeam") or ""),
                away_country=away.get("IdCountry") or "",
                stadium_id=str(stadium.get("IdStadium") or "") or None,
                stadium_name=_en(stadium.get("Name")),
                group_label=_en(m.get("GroupName")),
                stage_label=_en(m.get("StageName")),
                match_number=m.get("MatchNumber"),
                referee=ref, referee_country=ref_country,
                raw=m,
            ))
        out.sort(key=lambda x: (x.date_utc, x.match_number or 0))
        return out

    # ---- live / lineup ----
    async def lineup(self, id_competition: str, id_season: str,
                     id_stage: str, id_match: str) -> FifaLineup | None:
        d = await self._get(
            f"/live/football/{id_competition}/{id_season}/{id_stage}/{id_match}"
        )
        if not d:
            return None
        home = d.get("HomeTeam") or {}
        away = d.get("AwayTeam") or {}

        def _players(block) -> tuple[list[dict], list[dict]]:
            starters, subs = [], []
            for p in (block.get("Players") or []):
                nm = _en(p.get("PlayerName")) or _en(p.get("ShortName"))
                rec = {
                    "id": str(p.get("IdPlayer") or ""),
                    "name": nm, "shirt": p.get("ShirtNumber"),
                    "position": p.get("Position"),
                    "captain": bool(p.get("Captain")),
                    "x": p.get("LineupX"), "y": p.get("LineupY"),
                    "status": p.get("Status"),
                }
                if p.get("Status") == 1:
                    starters.append(rec)
                else:
                    subs.append(rec)
            return starters, subs

        hs, hsubs = _players(home)
        as_, asubs = _players(away)

        ref = None; ref_country = None; officials = []
        for o in d.get("Officials") or []:
            row = {
                "type": o.get("OfficialType"),
                "country": o.get("IdCountry"),
                "name": _en(o.get("NameShort")) or _en(o.get("Name")),
            }
            officials.append(row)
            if row["type"] == 1:
                ref = row["name"]; ref_country = row["country"]

        return FifaLineup(
            id_match=str(id_match),
            home_starters=hs, home_subs=hsubs,
            away_starters=as_, away_subs=asubs,
            referee=ref, referee_country=ref_country,
            officials=officials, match_status=d.get("MatchStatus"),
            raw=d,
        )

    async def player_profile(self, id_player: str) -> dict | None:
        return await self._get(f"/players/{id_player}")


# ---- quick CLI smoke test ----
if __name__ == "__main__":
    async def main():
        api = FifaApi()
        try:
            ms = await api.wc2026_matches()
            print(f"Fetched {len(ms)} WC2026 matches from FIFA")
            for m in ms[:5]:
                print(f"  M{m.match_number:>3}  {m.date_utc:%Y-%m-%d %H:%M} UTC  "
                      f"{m.home_team:>22s} v {m.away_team:<22s}  "
                      f"{m.stadium_name or '?':>18s}  ref={m.referee}")
            if ms:
                # Probe lineup for the opener (will be empty pre-kickoff).
                m1 = ms[0]
                lu = await api.lineup(WC_COMPETITION_ID, WC_SEASON_ID_2026,
                                      m1.id_stage, m1.id_match)
                if lu:
                    print(f"\nOpener lineup status: match_status={lu.match_status} "
                          f"home_starters={len(lu.home_starters)} away_starters={len(lu.away_starters)}")
                    print(f"  ref: {lu.referee} ({lu.referee_country})")
        finally:
            await api.close()
    asyncio.run(main())
