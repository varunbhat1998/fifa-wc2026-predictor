"""Async phase handlers — one per state-machine transition.

Each handler reads MatchState, performs side-effects (Telegram, ML calls,
result fetch), advances the phase, and persists the new state.
"""
from __future__ import annotations

from datetime import datetime

import bot_log
from bot_data import fetch_recent_results, find_result
from bot_ml import MLClient
from bot_notify import Notifier, format_prediction, format_result, score_tip
from bot_state import MatchState, Phase


async def run_pre_match(state: MatchState, ml: MLClient, notify: Notifier,
                        *, refined: bool = False,
                        home_starters: list[str] | None = None,
                        away_starters: list[str] | None = None,
                        referee: str | None = None) -> None:
    """Initial (squad-based) prediction OR a refined re-prediction once the
    confirmed XI is in. `refined=True` changes the header to flag the update.
    Falls back to FIFA's pre-assigned referee on the initial prediction so the
    ref-bias adjustment fires from T-180min onward."""
    if referee is None:
        referee = getattr(state, "referee", None)
    pred = await ml.predict_prematch(
        home=state.home, away=state.away,
        date=state.kickoff.strftime("%Y-%m-%d"),
        city=state.city, neutral=state.neutral,
        match_num_in_tournament=state.match_no,
        home_starters=home_starters, away_starters=away_starters, referee=referee,
    )
    tip = await ml.predict_tip(
        home=state.home, away=state.away,
        date=state.kickoff.strftime("%Y-%m-%d"),
        city=state.city, neutral=state.neutral,
        match_num_in_tournament=state.match_no,
        home_starters=home_starters, away_starters=away_starters, referee=referee,
    )
    state.prediction = {**pred, "tip": tip, "refined": refined,
                        "referee": referee,
                        "starters_known": bool(home_starters and away_starters)}
    if refined:
        state.phase = Phase.LINEUP_REFINED
    else:
        state.phase = Phase.PREDICTED
    state.save()

    msg = format_prediction(
        home=state.home, away=state.away, group=state.group, city=state.city,
        kickoff=state.kickoff.strftime("%Y-%m-%d %H:%M"),
        p_home=pred["p_home"], p_draw=pred["p_draw"], p_away=pred["p_away"],
        predicted=pred["predicted"], confidence=pred["confidence"],
        home_elo=pred["home_elo"], away_elo=pred["away_elo"],
        signals_agree=pred["signals_agree"],
        tip=tip["best_tip"], tip_ev=tip["best_tip_ev"], alt_tips=tip["alt_tips"],
        refined=refined, referee=referee,
        mgr_info=tip.get("manager_info"),
    )
    await notify.send(msg)

    bot_log.log_prediction({
        "match_no": state.match_no, "date": state.kickoff.strftime("%Y-%m-%d"),
        "group": state.group, "home": state.home, "away": state.away, "city": state.city,
        "p_home": pred["p_home"], "p_draw": pred["p_draw"], "p_away": pred["p_away"],
        "predicted": pred["predicted"], "confidence": pred["confidence"],
        "home_elo": pred["home_elo"], "away_elo": pred["away_elo"],
        "tip": tip["best_tip"], "tip_ev": tip["best_tip_ev"],
    })

    # On initial squad prediction we move to LINEUP_POLLING; on refined we go
    # straight to AWAITING_RESULT.
    state.phase = Phase.AWAITING_RESULT if refined else Phase.LINEUP_POLLING
    state.save()


async def try_lineup_refresh(state: MatchState, ml: MLClient, notify: Notifier) -> bool:
    """Poll for confirmed XI; if found, send the refined prediction. Returns True
    when refined prediction was successfully posted."""
    try:
        info = await ml.fetch_lineup(state.home, state.away,
                                     state.kickoff.strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"[bot] lineup fetch error for M{state.match_no}: {e}")
        return False
    if not info.get("found"):
        return False
    await run_pre_match(
        state, ml, notify,
        refined=True,
        home_starters=info.get("home_starters"),
        away_starters=info.get("away_starters"),
        referee=info.get("referee"),
    )
    return True


async def poll_result(state: MatchState, notify: Notifier) -> bool:
    """Returns True if the result was found and applied; False otherwise."""
    df = await fetch_recent_results()
    res = find_result(df, state.kickoff, state.home, state.away)
    if res is None:
        return False
    state.final_home_score, state.final_away_score = res
    state.phase = Phase.RESULT_IN
    state.save()

    pred = state.prediction or {}
    if res[0] > res[1]: actual = "home_win"
    elif res[0] < res[1]: actual = "away_win"
    else: actual = "draw"
    was_correct = (pred.get("predicted") == actual)

    tip = (pred.get("tip") or {}).get("best_tip")
    tip_pts = score_tip(tip, res[0], res[1]) if tip else None

    await notify.send(format_result(
        home=state.home, away=state.away,
        home_score=res[0], away_score=res[1],
        predicted=pred.get("predicted", "?"),
        was_correct=was_correct,
        tip=tip, tip_points=tip_pts,
    ))
    return True
