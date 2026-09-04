"""Async client for the FastAPI prediction server."""
from __future__ import annotations

from typing import Any

import httpx

from bot_config import API_BASE


class MLClient:
    def __init__(self, base: str = API_BASE) -> None:
        self.base = base.rstrip("/")
        self._client = httpx.AsyncClient(timeout=15.0)

    async def predict_prematch(
        self, home: str, away: str, date: str,
        city: str | None, neutral: bool, match_num_in_tournament: int,
        tournament_class: str = "wc_finals",
        home_starters: list[str] | None = None,
        away_starters: list[str] | None = None,
        referee: str | None = None,
    ) -> dict[str, Any]:
        r = await self._client.post(
            f"{self.base}/predict/prematch",
            json={
                "home": home, "away": away, "date": date,
                "tournament_class": tournament_class, "city": city,
                "neutral": neutral, "match_num_in_tournament": match_num_in_tournament,
                "home_starters": home_starters, "away_starters": away_starters,
                "referee": referee,
            },
        )
        r.raise_for_status()
        return r.json()

    async def predict_tip(
        self, home: str, away: str, date: str,
        city: str | None, neutral: bool, match_num_in_tournament: int,
        tournament_class: str = "wc_finals",
        home_starters: list[str] | None = None,
        away_starters: list[str] | None = None,
        referee: str | None = None,
    ) -> dict[str, Any]:
        r = await self._client.post(
            f"{self.base}/predict/tip",
            json={
                "home": home, "away": away, "date": date,
                "tournament_class": tournament_class, "city": city,
                "neutral": neutral, "match_num_in_tournament": match_num_in_tournament,
                "home_starters": home_starters, "away_starters": away_starters,
                "referee": referee,
            },
        )
        r.raise_for_status()
        return r.json()

    async def fetch_lineup(self, home: str, away: str, date: str) -> dict[str, Any]:
        r = await self._client.get(f"{self.base}/lineup",
                                   params={"home": home, "away": away, "date": date})
        r.raise_for_status()
        return r.json()

    async def set_manual_lineup(self, home: str, away: str, date: str,
                                 home_starters: list[str], away_starters: list[str],
                                 referee: str | None = None) -> dict[str, Any]:
        r = await self._client.post(
            f"{self.base}/lineup/manual",
            json={"home": home, "away": away, "date": date,
                  "home_starters": home_starters, "away_starters": away_starters,
                  "referee": referee},
        )
        r.raise_for_status()
        return r.json()

    async def tournament(self) -> dict[str, Any]:
        r = await self._client.get(f"{self.base}/tournament")
        r.raise_for_status()
        return r.json()

    async def forecast(self) -> dict[str, Any]:
        # The cascade does ~30 predictions sequentially; bump the timeout.
        r = await self._client.get(f"{self.base}/forecast", timeout=120.0)
        r.raise_for_status()
        return r.json()

    async def bonus(self) -> dict[str, Any]:
        r = await self._client.get(f"{self.base}/bonus")
        r.raise_for_status()
        return r.json()

    async def bonus_submit(self) -> dict[str, Any]:
        r = await self._client.get(f"{self.base}/bonus/submit")
        r.raise_for_status()
        return r.json()

    async def elo(self, top: int = 30) -> dict[str, Any]:
        r = await self._client.get(f"{self.base}/elo", params={"top": top})
        r.raise_for_status()
        return r.json()

    async def reload(self) -> dict[str, Any]:
        r = await self._client.post(f"{self.base}/reload-models")
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()
