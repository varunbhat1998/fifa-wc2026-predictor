"""
Async entry point for the FIFA WC 2026 Telegram bot.

Loop responsibilities:
  - Hydrate MatchState objects from the hard-coded WC2026 fixture list, picking up
    any checkpoints already saved on disk.
  - For each upcoming fixture, fire `run_pre_match` exactly once at T-60min.
  - For each AWAITING_RESULT fixture, periodically poll martj42 results.csv,
    apply the result, and (separately) kick off the auto-retrain pipeline.
  - Long-poll Telegram for slash commands:
      /predict <home> <away>     ad-hoc prediction
      /bracket                   current Monte Carlo standings (top 12)
      /elo                       top-20 current Elo
      /status                    counts by phase
      /upcoming                  next 5 fixtures with kickoff times
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bot_phases
from bot_config import (
    DEFAULT_KICKOFF_HOUR_UTC,
    LINEUP_FINAL_REFRESH_MIN,
    LINEUP_POLL_INTERVAL_SEC,
    LINEUP_POLL_START_MIN,
    PREDICT_LEAD_MIN,
    RESULT_POLL_SEC,
    RESULTS_ACTUAL_CSV,
    SCHEDULER_TICK_SEC,
)
from bot_ml import MLClient
from bot_notify import Notifier, format_prediction
from bot_state import MatchState, Phase, all_checkpoints
from wc2026_schedule import all_group_fixtures


def _utc_kickoff(date: datetime) -> datetime:
    # FIFA dates already carry UTC; everything else is fed in as naive local.
    if date.tzinfo is None:
        # Hardcoded fallback case: pick a sensible UTC hour.
        return date.replace(hour=DEFAULT_KICKOFF_HOUR_UTC, minute=0,
                            second=0, microsecond=0, tzinfo=timezone.utc)
    return date.astimezone(timezone.utc)


def hydrate_states() -> dict[int, MatchState]:
    existing = {s.match_no: s for s in all_checkpoints()}
    for fx in all_group_fixtures():
        if fx.match_no in existing:
            continue
        kickoff = _utc_kickoff(fx.date)
        s = MatchState(
            match_no=fx.match_no, home=fx.home, away=fx.away,
            city=fx.city, group=fx.group, stage=fx.stage,
            neutral=fx.neutral, kickoff=kickoff,
            id_match=fx.id_match, id_stage=fx.id_stage,
            referee=fx.referee, referee_country=fx.referee_country,
            stadium_name=fx.stadium_name,
        )
        s.save()
        existing[fx.match_no] = s
    return existing


async def trigger_retrain(state: MatchState) -> None:
    """Append the result, run the retrain script as a subprocess, hot-reload API."""
    RESULTS_ACTUAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_header = not RESULTS_ACTUAL_CSV.exists()
    with open(RESULTS_ACTUAL_CSV, "a", encoding="utf-8") as f:
        if new_header:
            f.write("match_no,date,home,away,home_score,away_score\n")
        f.write(
            f"{state.match_no},{state.kickoff.strftime('%Y-%m-%d')},"
            f"{state.home},{state.away},{state.final_home_score},{state.final_away_score}\n"
        )
    # Fire-and-forget retrain. The script is responsible for hot-reloading.
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(Path(__file__).parent / "07_retrain_after_match.py"),
        str(state.match_no),
    )
    await proc.wait()
    state.phase = Phase.RETRAINED
    state.save()


async def scheduler(states: dict[int, MatchState], ml: MLClient, notify: Notifier) -> None:
    last_lineup_try: dict[int, datetime] = {}
    while True:
        now = datetime.now(timezone.utc)
        for state in list(states.values()):
            if state.phase == Phase.SCHEDULED:
                if now >= state.kickoff - timedelta(minutes=PREDICT_LEAD_MIN):
                    try:
                        await bot_phases.run_pre_match(state, ml, notify)
                    except Exception as e:
                        print(f"[scheduler] prematch {state.match_no} failed: {e}")
            elif state.phase in (Phase.PREDICTED, Phase.LINEUP_POLLING):
                # Start polling for confirmed XI at T-LINEUP_POLL_START_MIN.
                start_at = state.kickoff - timedelta(minutes=LINEUP_POLL_START_MIN)
                kickoff = state.kickoff
                if now < start_at:
                    continue
                last = last_lineup_try.get(state.match_no)
                due = (last is None) or (
                    (now - last).total_seconds() >= LINEUP_POLL_INTERVAL_SEC
                )
                if due:
                    last_lineup_try[state.match_no] = now
                    try:
                        ok = await bot_phases.try_lineup_refresh(state, ml, notify)
                    except Exception as e:
                        print(f"[scheduler] lineup refresh M{state.match_no}: {e}")
                        ok = False
                    if not ok and now >= kickoff - timedelta(minutes=LINEUP_FINAL_REFRESH_MIN):
                        # Last-chance: advance to AWAITING_RESULT to stop polling.
                        state.phase = Phase.AWAITING_RESULT
                        state.save()
            elif state.phase == Phase.LINEUP_REFINED:
                if now >= state.kickoff:
                    state.phase = Phase.AWAITING_RESULT
                    state.save()
            elif state.phase == Phase.AWAITING_RESULT:
                if now >= state.kickoff + timedelta(hours=2):
                    last = last_lineup_try.get(("result", state.match_no))
                    # Adaptive cadence: poll every 5 min in the first hour after
                    # kickoff+2h, then every 30 min until found. This avoids
                    # waiting 30 min after a fresh restart.
                    minutes_since_window = (now - (state.kickoff + timedelta(hours=2))).total_seconds() / 60
                    poll_interval = 300 if minutes_since_window < 60 else RESULT_POLL_SEC
                    due = (last is None) or ((now - last).total_seconds() >= poll_interval)
                    if due:
                        last_lineup_try[("result", state.match_no)] = now
                        try:
                            ok = await bot_phases.poll_result(state, notify)
                            if ok:
                                asyncio.create_task(trigger_retrain(state))
                        except Exception as e:
                            print(f"[scheduler] result poll {state.match_no} failed: {e}")
            elif state.phase == Phase.RETRAINED:
                state.phase = Phase.DONE
                state.save()
        await asyncio.sleep(SCHEDULER_TICK_SEC)


async def command_loop(states: dict[int, MatchState], ml: MLClient, notify: Notifier) -> None:
    while True:
        cmd = await notify.commands.get()
        c = cmd["cmd"]
        args = cmd["args"]
        try:
            if c == "/elo":
                top = (await ml.elo(20)).get("top", [])
                lines = [f"{r['team']:20s} {r['rating']:.0f}" for r in top]
                await notify.send("<b>Top 20 Elo</b>\n<pre>" + "\n".join(lines) + "</pre>")
            elif c == "/bracket":
                t = (await ml.tournament()).get("standings", [])[:12]
                lines = [f"{r['team']:18s} G{r['group']}  champ {r['p_champion']:.1%}" for r in t]
                await notify.send("<b>WC2026 Top 12</b>\n<pre>" + "\n".join(lines) + "</pre>")
            elif c == "/status":
                from collections import Counter
                cnt = Counter(s.phase.value for s in states.values())
                await notify.send("<b>Status</b>\n" + "\n".join(f"{k}: {v}" for k, v in cnt.items()))
            elif c == "/upcoming" or c == "/next":
                # Optional arg: number of days to look ahead. Default 7.
                try:
                    days_ahead = int(args[0]) if args else 7
                except ValueError:
                    days_ahead = 7
                # Use the cascade (already has predicted scores per match) and
                # filter to group-stage matches that haven't been played yet.
                f = await ml.forecast()
                matches = f.get("matches", [])
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                cutoff = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                upcoming = [
                    m for m in matches
                    if not m.get("from_actual")            # exclude played
                    and m.get("date", "")[:10] >= today_str
                    and m.get("date", "")[:10] <= cutoff
                ]
                if not upcoming:
                    await notify.send(
                        f"No upcoming matches in the next {days_ahead} days."
                    )
                    continue
                # Group by date, one Telegram message per date. Stage label is
                # shown for knockout matches so you know it's a KO prediction.
                from itertools import groupby
                upcoming.sort(key=lambda m: (m["date"], m["match_no"]))
                stage_label = {"group": "", "R32": " [R32]", "R16": " [R16]",
                               "QF": " [QF]", "SF": " [SF]", "3rd": " [3rd]",
                               "Final": " [Final]"}
                for date_key, grp in groupby(upcoming, key=lambda m: m["date"][:10]):
                    rows = list(grp)
                    lines = [
                        f"M{m['match_no']:>3}{stage_label.get(m.get('stage', ''), '')} "
                        f"{m['home']:>22s} {m['best_tip']:>4s} {m['away']:<22s}"
                        for m in rows
                    ]
                    await notify.send(
                        f"<b>{date_key}</b> ({len(rows)} match{'es' if len(rows) > 1 else ''})\n"
                        f"<pre>" + "\n".join(lines) + "</pre>"
                    )
            elif c == "/forecast":
                await notify.send("Running full tournament cascade — this takes ~60s ...")
                f = await ml.forecast()
                s = f["summary"]
                # 1) summary
                await notify.send(
                    f"🏆 <b>Predicted champion: {s['champion']}</b>\n"
                    f"Final: {s['final']}\n"
                    f"3rd place: {s['third_place']}\n"
                    f"Runner-up: {s.get('runner_up')}\n"
                    f"4th: {s.get('fourth_place')}"
                )
                # 2) group stage — one message per group (12 messages, 6 matches each)
                matches = f["matches"]
                for g in sorted({m.get("group") for m in matches if m.get("stage") == "group" and m.get("group")}):
                    rows = [m for m in matches if m.get("stage") == "group" and m.get("group") == g]
                    lines = [
                        f"{m['date'][5:10]}  {m['home']:>22s} {m['predicted_score']:>4s} {m['away']:<22s}"
                        + ("  ✓played" if m.get("from_actual") else "")
                        for m in rows
                    ]
                    await notify.send(f"<b>Group {g}</b>\n<pre>" + "\n".join(lines) + "</pre>")
                # 3) knockouts — one message per round
                for stage_label in ("R32", "R16", "QF", "SF", "3rd", "Final"):
                    rows = [m for m in matches if m.get("stage") == stage_label]
                    if not rows:
                        continue
                    lines = [
                        f"{m['date'][5:10]}  {m['home']:>22s} {m['predicted_score']:>4s} {m['away']:<22s}"
                        + ("  ✓played" if m.get("from_actual") else "")
                        for m in rows
                    ]
                    label = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
                             "SF": "Semi-finals", "3rd": "3rd-place play-off", "Final": "Final"}[stage_label]
                    await notify.send(f"<b>{label}</b>\n<pre>" + "\n".join(lines) + "</pre>")
            elif c == "/champion":
                f = await ml.forecast()
                s = f["summary"]
                await notify.send(f"🏆 <b>{s['champion']}</b> — final: {s['final']}")
            elif c == "/refresh":
                # Data-only refresh: pull fresh FIFA schedule + latest martj42
                # results, rebuild cascade, hot-reload API. Skips the slow model
                # retrain — use /run <mn> or wait for auto-retrain for that.
                # Takes ~90s total.
                await notify.send(
                    "🔄 <b>Refreshing schedule + results</b>\n"
                    "Pulling FIFA schedule, martj42 results, rebuilding cascade..."
                )
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(Path(__file__).parent / "18_refresh.py"),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                )
                out, _ = await proc.communicate()
                tail = (out.decode(errors="replace") or "").splitlines()
                summary_lines = [l for l in tail if any(
                    k in l for k in ("[refresh]", "loaded_at", "Traceback", "Error")
                )][-15:]
                emoji = "✅" if proc.returncode == 0 else "❌"
                await notify.send(
                    f"{emoji} <b>Refresh {'complete' if proc.returncode == 0 else 'failed'}</b>\n"
                    "<pre>" + "\n".join(summary_lines) + "</pre>"
                )
            elif c == "/bonus":
                # Sub-commands: /bonus  (digest) | /bonus submit (copy-paste lines)
                if args and args[0].lower() == "submit":
                    res = await ml.bonus_submit()
                    txt = res.get("text", "")
                    await notify.send(f"<b>Copy-paste these into the Bonus form:</b>\n<pre>{txt}</pre>")
                else:
                    picks = await ml.bonus()
                    lines = ["<b>🏆 WC2026 Bonus Picks</b>", ""]
                    lines.append(f"<b>Champion:</b> {picks['champion']} (top 5 by P)")
                    for r in picks["champion_top5"]:
                        lines.append(f"  {r['team']}  P={r['p_champion']:.1%}")
                    lines.append("")
                    lines.append(f"<b>Top scorer team:</b> {picks['top_scorer_team']}")
                    for r in picks["top_scorer_team_top5"]:
                        lines.append(f"  {r['team']}  score={r['top_scorer_score']:.0f}")
                    lines.append("")
                    lines.append("<b>Group winners:</b>")
                    for g, t in sorted(picks["group_winners"].items()):
                        rows = picks["group_winners_detailed"][g]
                        p = rows[0]["p_1st_in_group"]
                        lines.append(f"  {g}: {t}  P={p:.0%}")
                    lines.append("")
                    lines.append("<b>Semi-finalists:</b> " + ", ".join(picks["semifinalists"]))
                    lines.append("")
                    lines.append("Type /bonus submit for copy-paste form.")
                    await notify.send("\n".join(lines))
            elif c == "/lineup":
                # Two forms:
                #   /lineup <match_no>           -> fetch current best XI from sources
                #   /lineup <match_no> | A,B,C... | X,Y,Z...   -> manual override
                raw = cmd["raw"]
                if "|" in raw:
                    # manual override path
                    try:
                        head, hs, as_ = [s.strip() for s in raw.split("|", 2)]
                        mn = int(head.split()[1])
                        st = states[mn]
                        home_xi = [x.strip() for x in hs.split(",") if x.strip()]
                        away_xi = [x.strip() for x in as_.split(",") if x.strip()]
                        await ml.set_manual_lineup(
                            st.home, st.away,
                            st.kickoff.strftime("%Y-%m-%d"),
                            home_xi, away_xi,
                        )
                        await notify.send(
                            f"Manual XI saved for M{mn} ({st.home} vs {st.away}).\n"
                            f"Home: {len(home_xi)}  ·  Away: {len(away_xi)}"
                        )
                    except Exception as e:
                        await notify.send(
                            f"Manual /lineup format:\n"
                            f"/lineup <match_no> | name1,name2,... | name1,name2,...\n"
                            f"Error: {e}"
                        )
                elif len(args) >= 1 and args[0].isdigit():
                    mn = int(args[0])
                    st = states[mn]
                    info = await ml.fetch_lineup(
                        st.home, st.away, st.kickoff.strftime("%Y-%m-%d"),
                    )
                    if info.get("found"):
                        await notify.send(
                            f"Lineup for M{mn} ({info['source']}, "
                            f"{'confirmed' if info.get('confirmed') else 'probable'}):\n"
                            f"<b>{st.home}</b>: {', '.join(info['home_starters'])}\n"
                            f"<b>{st.away}</b>: {', '.join(info['away_starters'])}"
                            + (f"\nReferee: {info['referee']}" if info.get('referee') else "")
                        )
                    else:
                        await notify.send(
                            f"No lineup found yet for M{mn}. "
                            f"Use /lineup {mn} | name1,name2,... | name1,name2,... to set it manually."
                        )
                else:
                    await notify.send(
                        "Usage:\n"
                        "/lineup <match_no>  — try to fetch confirmed XI now\n"
                        "/lineup <match_no> | h1,h2,... | a1,a2,...  — paste manual XI"
                    )
            elif c == "/result" and len(args) >= 2:
                # Manual result input when martj42 / FIFA feeds are lagging.
                # Usage: /result <match_no> <home_score>-<away_score>
                #   /result 9 7-1   -> Germany 7-1 Curacao
                try:
                    mn = int(args[0])
                    hs, as_ = (int(x) for x in args[1].split("-"))
                except (ValueError, KeyError):
                    await notify.send(
                        "Usage: <code>/result &lt;match_no&gt; &lt;home&gt;-&lt;away&gt;</code>\n"
                        "Example: <code>/result 9 7-1</code>"
                    )
                    continue
                if mn not in states:
                    await notify.send(f"No match {mn} in schedule.")
                    continue
                st = states[mn]
                # 1. Append to results_actual.csv
                import csv
                from bot_config import RESULTS_ACTUAL_CSV
                RESULTS_ACTUAL_CSV.parent.mkdir(parents=True, exist_ok=True)
                rows_existing = []
                if RESULTS_ACTUAL_CSV.exists():
                    import pandas as _pd
                    rows_existing = _pd.read_csv(RESULTS_ACTUAL_CSV).to_dict("records")
                rows_existing = [r for r in rows_existing if int(r["match_no"]) != mn]
                rows_existing.append({
                    "match_no": mn, "date": st.kickoff.strftime("%Y-%m-%d"),
                    "home": st.home, "away": st.away,
                    "home_score": hs, "away_score": as_,
                })
                rows_existing.sort(key=lambda r: int(r["match_no"]))
                with open(RESULTS_ACTUAL_CSV, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["match_no", "date", "home", "away",
                                                       "home_score", "away_score"])
                    w.writeheader(); w.writerows(rows_existing)
                # 2. Update bot state + kick off retrain
                st.final_home_score, st.final_away_score = hs, as_
                st.phase = Phase.RESULT_IN
                st.save()
                # Score the prediction if we have one logged
                from bot_notify import score_tip
                tip = (st.prediction or {}).get("tip", {}).get("best_tip") if st.prediction else None
                tip_pts = score_tip(tip, hs, as_) if tip else None
                pts_str = f"  ·  tip <b>{tip} → {tip_pts} pts</b>" if tip else ""
                await notify.send(
                    f"📥 Recorded M{mn}: <b>{st.home} {hs}-{as_} {st.away}</b>{pts_str}\n"
                    f"Triggering retrain..."
                )
                asyncio.create_task(trigger_retrain(st))
            elif c == "/run" and len(args) >= 1 and args[0].isdigit():
                # Force a re-prediction now (uses whatever lineup is available).
                mn = int(args[0])
                st = states[mn]
                info = await ml.fetch_lineup(st.home, st.away,
                                              st.kickoff.strftime("%Y-%m-%d"))
                hs = info.get("home_starters") if info.get("found") else None
                as_ = info.get("away_starters") if info.get("found") else None
                ref = info.get("referee") if info.get("found") else None
                await bot_phases.run_pre_match(
                    st, ml, notify, refined=bool(hs and as_),
                    home_starters=hs, away_starters=as_, referee=ref,
                )
            elif c == "/group" and len(args) >= 1:
                # Quick group predictions
                g = args[0].upper()
                f = await ml.forecast()
                matches = [m for m in f["matches"] if m["stage"] == "group" and m.get("group") == g]
                if not matches:
                    await notify.send(f"No group {g}.")
                    continue
                lines = [
                    f"{m['date']} {m['home']:>22s} {m['predicted_score']:>4s} {m['away']:<22s}"
                    + ("  (played)" if m["from_actual"] else "")
                    for m in matches
                ]
                await notify.send(f"<b>Group {g}</b>\n<pre>" + "\n".join(lines) + "</pre>")
            elif c == "/predict" and len(args) >= 2:
                # /predict <home> <away> — ad-hoc, treats as a wc_finals neutral fixture today.
                home = " ".join(args[: len(args) // 2])
                away = " ".join(args[len(args) // 2:])
                pred = await ml.predict_prematch(
                    home=home, away=away,
                    date=datetime.utcnow().strftime("%Y-%m-%d"),
                    city=None, neutral=True, match_num_in_tournament=1,
                )
                await notify.send(format_prediction(
                    home=home, away=away, group=None, city="(neutral)",
                    kickoff=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                    p_home=pred["p_home"], p_draw=pred["p_draw"], p_away=pred["p_away"],
                    predicted=pred["predicted"], confidence=pred["confidence"],
                    home_elo=pred["home_elo"], away_elo=pred["away_elo"],
                    signals_agree=pred["signals_agree"],
                ))
            else:
                await notify.send(
                    "Commands:\n"
                    "/bonus - all 18 Bonus-tab picks (digest)\n"
                    "/bonus submit - copy-paste form for the league\n"
                    "/forecast - full cascade prediction to the final\n"
                    "/refresh - re-pull FIFA schedule + results, rebuild cascade (~90s)\n"
                    "/champion - just the predicted winner\n"
                    "/group &lt;letter&gt; - predicted scores for one group\n"
                    "/upcoming [days] - chronological match picks (default 7d)\n"
                    "/result &lt;mn&gt; &lt;h&gt;-&lt;a&gt; - manually input a final score\n"
                    "/elo - top-20 current Elo\n"
                    "/bracket - Monte Carlo champion probabilities\n"
                    "/status - bot phase counts\n"
                    "/lineup &lt;match_no&gt; - get confirmed XI (or paste it manually)\n"
                    "/run &lt;match_no&gt; - re-run model on demand for that match\n"
                    "/predict &lt;home&gt; &lt;away&gt; - ad-hoc prediction"
                )
        except Exception as e:
            await notify.send(f"Error handling {c}: {e}")


async def catch_up_on_startup(states: dict[int, MatchState], notify: Notifier) -> None:
    """Find matches whose kickoff has passed but state hasn't reached RESULT_IN.
    Poll martj42 once for all of them so we don't miss results when the bot was
    down across multiple match days."""
    from bot_data import fetch_recent_results, find_result
    try:
        df = await fetch_recent_results()
    except Exception as e:
        print(f"[catch_up] result fetch failed: {e}")
        return
    now = datetime.now(timezone.utc)
    n_found = 0
    for state in states.values():
        if state.phase in (Phase.RETRAINED, Phase.DONE, Phase.RESULT_IN):
            continue
        if now < state.kickoff + timedelta(hours=2):
            continue   # match not finished yet
        res = find_result(df, state.kickoff, state.home, state.away)
        if res is None:
            continue
        state.final_home_score, state.final_away_score = res
        state.phase = Phase.RESULT_IN
        state.save()
        n_found += 1
        # Trigger retrain for each missed match (sequential — they'll queue up).
        asyncio.create_task(trigger_retrain(state))
    if n_found:
        await notify.send(
            f"⏪ <b>Catch-up</b>: detected {n_found} missed result(s) since last run. "
            f"Retraining in background — predictions will update."
        )


async def main() -> None:
    ml = MLClient()
    notify = Notifier()
    states = hydrate_states()
    print(f"[match_bot] hydrated {len(states)} match states "
          f"({sum(1 for s in states.values() if s.phase == Phase.SCHEDULED)} scheduled).")
    if not notify.configured:
        print("[match_bot] FIFA_BOT_TOKEN / FIFA_BOT_CHAT_ID not set — running in stub mode.")
    # On startup, sweep for any matches whose results we missed while the bot
    # was down. This is the fix for the M2/M4/M7/M8 gap we hit earlier.
    await catch_up_on_startup(states, notify)
    try:
        await asyncio.gather(
            scheduler(states, ml, notify),
            notify.poll_commands(),
            command_loop(states, ml, notify),
        )
    finally:
        await ml.close()
        await notify.close()


if __name__ == "__main__":
    asyncio.run(main())
