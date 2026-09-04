"""Telegram send + command long-polling."""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from bot_config import TELEGRAM_API, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN


class Notifier:
    def __init__(self) -> None:
        self.token = TELEGRAM_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self._client = httpx.AsyncClient(timeout=20.0)
        self.commands: asyncio.Queue[dict] = asyncio.Queue()
        self._last_update_id = 0

    @property
    def base(self) -> str:
        return f"{TELEGRAM_API}/bot{self.token}"

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, text: str) -> None:
        if not self.configured:
            # Windows console uses cp1252 by default; coerce emojis safely.
            try:
                print(f"[telegram-stub] {text}")
            except UnicodeEncodeError:
                print(f"[telegram-stub] {text.encode('ascii', 'replace').decode('ascii')}")
            return
        try:
            await self._client.post(
                f"{self.base}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                      "disable_web_page_preview": True},
            )
        except Exception as e:
            print(f"[telegram] send failed: {e}")

    async def poll_commands(self) -> None:
        """Background long-poll loop; pushes recognised commands into self.commands.

        Resilient: on transient network/HTTP errors we back off and retry. On
        409 Conflict (another bot polling same token) we wait longer because
        that means a duplicate process is fighting us — the human will need to
        kill the duplicate."""
        if not self.configured:
            return
        backoff = 5
        while True:
            try:
                r = await self._client.get(
                    f"{self.base}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 25},
                    timeout=35.0,
                )
                if r.status_code == 409:
                    print("[telegram] 409 Conflict — another bot is polling this token. "
                          "Kill the duplicate process (taskkill /F /IM python.exe) and restart.")
                    await asyncio.sleep(30)
                    continue
                r.raise_for_status()
                data = r.json()
                if not data.get("ok"):
                    print(f"[telegram] response not ok: {data}")
                    await asyncio.sleep(backoff)
                    continue
                backoff = 5
                for u in data.get("result", []):
                    self._last_update_id = u["update_id"]
                    msg = u.get("message") or {}
                    text = (msg.get("text") or "").strip()
                    if text.startswith("/"):
                        cmd, *args = text.split()
                        await self.commands.put({"cmd": cmd.lower(), "args": args, "raw": text})
            except Exception as e:
                # Full type + str so empty errors are still identifiable.
                err_type = type(e).__name__
                err_msg = str(e) or repr(e)
                print(f"[telegram] poll error [{err_type}]: {err_msg}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def close(self) -> None:
        await self._client.aclose()


def format_prediction(home: str, away: str, group: str | None, city: str,
                      kickoff: str, p_home: float, p_draw: float, p_away: float,
                      predicted: str, confidence: float,
                      home_elo: float, away_elo: float,
                      signals_agree: int,
                      tip: str | None = None, tip_ev: float | None = None,
                      alt_tips: list[dict] | None = None,
                      refined: bool = False, referee: str | None = None,
                      mgr_info: dict | None = None) -> str:
    fav, p_fav = (
        (home, p_home) if predicted == "home_win"
        else (away, p_away) if predicted == "away_win"
        else ("Draw", p_draw)
    )
    group_txt = f"Group {group}" if group else "Group ?"
    header = "🔁 <b>REFINED (confirmed XI)</b>" if refined else "📋 <b>INITIAL (squad-based)</b>"
    lines = [
        header,
        f"<b>{home} vs {away}</b>",
        f"{group_txt} · {city} · kickoff {kickoff} UTC",
        f"Elo: {home} {home_elo:.0f}  |  {away} {away_elo:.0f}",
        f"<b>{home}</b> {p_home:.0%}  ·  Draw {p_draw:.0%}  ·  <b>{away}</b> {p_away:.0%}",
        f"Pick: <b>{fav}</b> ({p_fav:.0%})  ·  Confidence {confidence:.0%}  ·  {signals_agree}/8 signals agree",
    ]
    if tip is not None:
        alt_txt = ""
        if alt_tips:
            alt_txt = "  ·  alts: " + ", ".join(
                f"{t['tip']} ({t['ev']:.2f})" for t in alt_tips[:2]
            )
        lines.append(f"🎯 <b>Tip: {tip}</b>  ·  EV {tip_ev:.2f} pts{alt_txt}")
    if referee:
        lines.append(f"Referee: {referee}")
    if mgr_info and (mgr_info.get("home_manager") or mgr_info.get("away_manager")):
        hm = mgr_info.get("home_manager") or "?"
        am = mgr_info.get("away_manager") or "?"
        hs = mgr_info.get("home_style") or "?"
        as_ = mgr_info.get("away_style") or "?"
        hf = mgr_info.get("home_formation") or "?"
        af = mgr_info.get("away_formation") or "?"
        shift_pp = (mgr_info.get("shift_applied_pp") or 0) * 100
        tactical = (f"  ·  matchup {shift_pp:+.1f}pp"
                    if abs(shift_pp) >= 0.5 else "")
        lines.append(f"Managers: {hm} ({hf}/{hs}) vs {am} ({af}/{as_}){tactical}")
    return "\n".join(lines)


def format_result(home: str, away: str, home_score: int, away_score: int,
                  predicted: str, was_correct: bool,
                  tip: str | None = None, tip_points: int | None = None) -> str:
    tick = "✅" if was_correct else "❌"
    if home_score > away_score: actual = "home_win"
    elif home_score < away_score: actual = "away_win"
    else: actual = "draw"
    lines = [
        f"{tick} <b>{home} {home_score}-{away_score} {away}</b>",
        f"Predicted: {predicted}  ·  Actual: {actual}",
    ]
    if tip is not None and tip_points is not None:
        emoji = "🎯" if tip_points == 4 else ("✅" if tip_points >= 2 else "❌")
        lines.append(f"{emoji} Tip {tip} → <b>{tip_points} pts</b>")
    return "\n".join(lines)


def score_tip(tip: str, actual_h: int, actual_a: int) -> int:
    """Apply the WC Tip Game 2026 rubric: 4 / 3 / 2 / 0 points."""
    try:
        ht, at = (int(x) for x in tip.split("-"))
    except Exception:
        return 0
    tip_w = 1 if ht > at else (-1 if ht < at else 0)
    act_w = 1 if actual_h > actual_a else (-1 if actual_h < actual_a else 0)
    if tip_w != act_w:
        return 0
    if ht == actual_h and at == actual_a:
        return 4
    if tip_w == 0:
        return 3   # both predicted draw, score wrong
    if (ht - at) == (actual_h - actual_a):
        return 3   # correct goal difference
    return 2       # right winner, wrong margin
