"""
Step 8 - Auto-retrain after a WC2026 match result comes in.

Invocation:
  python 07_retrain_after_match.py <match_no>

What it does:
  1. Appends the result (read from results_actual.csv) to data/matches.csv as a
     new row using the same schema (tournament='FIFA World Cup', tournament_class='wc_finals').
  2. Incrementally updates Elo for the two teams (the rest of the rating table
     is unchanged because Elo is causal).
  3. Re-builds features for the new row only — appended to match_features.csv.
  4. Retrains the pre-match ensemble (full retrain, ~2-5 min on this dataset).
  5. Re-runs the Monte Carlo simulator.
  6. POSTs /reload-models to the live API.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

# Load .env so FIFA_BOT_TOKEN + FIFA_BOT_CHAT_ID are visible.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DATA_DIR = Path(__file__).parent / "data"
ROOT = Path(__file__).parent

# ---- Telegram notification helper ----
_TELEGRAM_TOKEN = os.environ.get("FIFA_BOT_TOKEN", "")
_TELEGRAM_CHAT  = os.environ.get("FIFA_BOT_CHAT_ID", "")
_STARTED_AT: float | None = None


def notify(text: str) -> None:
    """Best-effort Telegram send. Never raises; never blocks retrain on failure.
    Also echoes to stdout so the cmd window shows the same status."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Windows cmd cp1252 chokes on emojis; strip them for local echo only.
        print(text.encode("ascii", "replace").decode("ascii"))
    if not (_TELEGRAM_TOKEN and _TELEGRAM_CHAT):
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": _TELEGRAM_CHAT, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10.0,
        )
    except Exception as e:
        print(f"[notify] telegram send failed: {e}")


def _elapsed_str() -> str:
    if _STARTED_AT is None:
        return ""
    s = int(time.time() - _STARTED_AT)
    m, s = divmod(s, 60)
    return f"{m}m{s:02d}s"


def append_to_matches(match_no: int) -> dict | None:
    actual = pd.read_csv(DATA_DIR / "results_actual.csv")
    row = actual[actual["match_no"] == match_no]
    if row.empty:
        print(f"[retrain] no actual result for match {match_no}; skip")
        return None
    r = row.iloc[0]
    home, away = r["home"], r["away"]
    hs, as_ = int(r["home_score"]), int(r["away_score"])
    date = pd.to_datetime(r["date"])

    matches = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"])
    # No-op guard: don't double-append.
    dup = matches[
        (matches["date"] == date)
        & (matches["home_team"] == home)
        & (matches["away_team"] == away)
        & (matches["home_score"].notna())
    ]
    if not dup.empty:
        print(f"[retrain] match already present in matches.csv; skip append")
        return None

    new = {col: pd.NA for col in matches.columns}
    new.update({
        "date": date, "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_,
        "tournament": "FIFA World Cup",
        "city": "", "country": "",  # not load-bearing for retrain
        "neutral": False,
        "home_canonical": home, "away_canonical": away,
        "tournament_class": "wc_finals", "is_friendly": False, "is_wc": True,
        "goal_diff": hs - as_,
        "result": "home_win" if hs > as_ else ("away_win" if hs < as_ else "draw"),
        "shootout_winner": pd.NA,
    })
    matches = pd.concat([matches, pd.DataFrame([new])], ignore_index=True)
    matches.to_csv(DATA_DIR / "matches.csv", index=False)
    print(f"[retrain] appended match {match_no}: {home} {hs}-{as_} {away}")
    return new


def run(cmd: list[str]) -> None:
    print(f"[retrain] $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise SystemExit(f"step failed: {cmd}")


def step_notify(label: str, cmd: list[str], *, optional: bool = False) -> None:
    """Run one pipeline step and post a Telegram milestone with elapsed time."""
    t0 = time.time()
    try:
        run(cmd)
    except SystemExit as e:
        if optional:
            notify(f"⚠️ <b>{label}</b> failed (skipped) — {e}")
            return
        notify(f"❌ <b>{label}</b> failed — aborting retrain")
        raise
    notify(f"✓ <b>{label}</b> ({int(time.time() - t0)}s · total {_elapsed_str()})")


def reload_api() -> None:
    try:
        with httpx.Client(timeout=10) as c:
            c.post("http://127.0.0.1:8001/reload-models")
        print("[retrain] /reload-models posted")
    except Exception as e:
        print(f"[retrain] hot reload skipped (API down?): {e}")


def _snapshot_top(n: int = 5) -> list[dict]:
    """Read latest MC sim results — top n teams by P(champion)."""
    p = DATA_DIR / "tournament_sim_results.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return df.sort_values("p_champion", ascending=False).head(n).to_dict("records")


def _format_top(rows: list[dict]) -> str:
    if not rows:
        return "(no MC results yet)"
    return "\n".join(
        f"  {r['team']:>22s}  P={r['p_champion']*100:.1f}%  (SF {r['p_advance_SF']*100:.0f}%)"
        for r in rows
    )


def _summarise_injuries() -> str:
    p = DATA_DIR / "injuries_2026.csv"
    if not p.exists():
        return "no injury feed yet"
    df = pd.read_csv(p, encoding="utf-8")
    n_out = (df["status"] == "OUT").sum()
    n_doubt = (df["status"] == "DOUBT").sum()
    return f"{n_out} OUT · {n_doubt} DOUBT"


def main() -> int:
    global _STARTED_AT
    if len(sys.argv) < 2:
        print("usage: python 07_retrain_after_match.py <match_no>")
        return 2
    match_no = int(sys.argv[1])
    _STARTED_AT = time.time()

    pre_top = _snapshot_top(5)

    # Look up the team names so the start notification has match context.
    home = away = hs = as_ = None
    actual_path = DATA_DIR / "results_actual.csv"
    if actual_path.exists():
        df_a = pd.read_csv(actual_path)
        row = df_a[df_a["match_no"] == match_no]
        if not row.empty:
            r = row.iloc[0]
            home, away = r["home"], r["away"]
            hs, as_ = int(r["home_score"]), int(r["away_score"])

    header = (
        f"🔄 <b>Retraining after M{match_no}</b>"
        + (f": {home} {hs}-{as_} {away}" if home else "")
        + "\nThis takes ~3-5 minutes."
    )
    notify(header)

    if append_to_matches(match_no) is None:
        # Nothing to do, but still reload + re-sim so /tournament reflects the
        # latest played-results without a full retrain.
        notify("ℹ️ Result already recorded — running tournament sim only.")
        step_notify("Monte Carlo (10k)", [sys.executable, "05_simulate_tournament.py"])
        reload_api()
        notify(f"✅ <b>Refresh complete</b> ({_elapsed_str()})")
        return 0

    # Full retrain pipeline. Each step posts a milestone to Telegram.
    step_notify("FIFA schedule + officials refreshed",
                [sys.executable, "15_fifa_schedule.py"])
    step_notify("Injury news (Claude extraction)",
                [sys.executable, "16_injury_fetcher.py"], optional=True)
    notify(f"  Injury status: {_summarise_injuries()}")
    step_notify("Elo recomputed", [sys.executable, "02_compute_elo.py"])
    step_notify("Team strength + chemistry (with injuries)",
                [sys.executable, "11_team_strength.py"])
    step_notify("Features rebuilt", [sys.executable, "03_features.py"])
    step_notify("Pre-match ensemble retrained",
                [sys.executable, "04_train_prematch.py"])
    step_notify("Poisson goal model retrained",
                [sys.executable, "08_train_goals.py"])
    step_notify("Monte Carlo (10k sims)",
                [sys.executable, "05_simulate_tournament.py"])
    step_notify("Live odds refreshed (the-odds-api)",
                [sys.executable, "17_odds_fetcher.py"], optional=True)
    step_notify("Auto-tune check",
                [sys.executable, "17_auto_tune.py"], optional=True)
    step_notify("Cascade bracket refreshed",
                [sys.executable, "14_cascade_forecast.py"])
    reload_api()
    notify("✓ <b>API hot-reloaded</b>")

    # Summary — what changed in the top of the bracket.
    post_top = _snapshot_top(5)
    changes = _compare_tops(pre_top, post_top)
    final_msg = [
        f"✅ <b>Retrain complete</b> in {_elapsed_str()}",
        "",
        "<b>Top 5 by P(champion):</b>",
        _format_top(post_top),
    ]
    if changes:
        final_msg += ["", "<b>What changed:</b>"] + changes
    notify("\n".join(final_msg))

    print(f"[retrain] done at {datetime.utcnow().isoformat()}")
    return 0


def _compare_tops(before: list[dict], after: list[dict]) -> list[str]:
    if not before or not after:
        return []
    before_p = {r["team"]: r["p_champion"] for r in before}
    after_p  = {r["team"]: r["p_champion"] for r in after}
    deltas = []
    for team in set(before_p) | set(after_p):
        b = before_p.get(team, 0.0)
        a = after_p.get(team, 0.0)
        d = (a - b) * 100  # percentage-point change
        if abs(d) >= 0.5:
            sign = "▲" if d > 0 else "▼"
            deltas.append((abs(d), f"  {sign} {team}: {b*100:.1f}% → {a*100:.1f}%"))
    deltas.sort(key=lambda x: -x[0])
    return [line for _, line in deltas[:6]]


if __name__ == "__main__":
    sys.exit(main())
