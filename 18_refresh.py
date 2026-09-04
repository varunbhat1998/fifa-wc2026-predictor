"""
Data-only refresh triggered by the bot's /refresh command.

Runs — in order — the cheap steps that fix scheduling / bracket issues without
touching the ML models:

  1. 15_fifa_schedule.py     — refresh FIFA bracket + officials (~5s)
  2. 01_fetch_history.py     — refresh martj42 latest results (~5s)
  3. Sync results_actual.csv from martj42                      (~2s)
  4. 14_cascade_forecast.py  — re-run cascade with new state   (~60s)
  5. POST /reload-models     — hot-reload API                   (~1s)

Total ~75s. Prints a compact status line for each step so the bot can render
a clean progress summary.
"""
from __future__ import annotations

import subprocess
import sys
import time
import unicodedata
from datetime import timedelta
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).parent
DATA = ROOT / "data"

NAME_MAP = {"Cape Verde": "Cabo Verde", "DR Congo": "Congo DR"}


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.strip().lower()


def step(label: str, cmd: list[str], *, optional: bool = False) -> None:
    t0 = time.time()
    print(f"[refresh] -> {label}...")
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    dt = int(time.time() - t0)
    if r.returncode != 0 and not optional:
        print(f"[refresh] {label}: FAILED after {dt}s")
        raise SystemExit(r.returncode)
    print(f"[refresh] {label}: ok ({dt}s)")


def sync_results_from_martj42() -> None:
    t0 = time.time()
    print("[refresh] -> sync results_actual.csv from martj42...")
    r = requests.get(
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        timeout=20,
    )
    mart = pd.read_csv(StringIO(r.text))
    mart["date"] = pd.to_datetime(mart["date"])
    wc = mart[(mart["date"] >= "2026-06-11") & (mart["tournament"] == "FIFA World Cup")]
    wc = wc.dropna(subset=["home_score", "away_score"]).copy()

    sch = pd.read_csv(DATA / "wc2026_fifa_schedule.csv")
    sch["date_only"] = sch["date_utc"].str[:10]
    sch["home_n"] = sch["home"].map(_norm)
    sch["away_n"] = sch["away"].map(_norm)

    matched = []
    for _, m in wc.iterrows():
        hN = _norm(NAME_MAP.get(m["home_team"], m["home_team"]))
        aN = _norm(NAME_MAP.get(m["away_team"], m["away_team"]))
        hit = None
        for delta in (0, 1, -1):
            d = (m["date"] + timedelta(days=delta)).strftime("%Y-%m-%d")
            cand = sch[sch["date_only"] == d]
            same = cand[
                ((cand["home_n"] == hN) & (cand["away_n"] == aN))
                | ((cand["home_n"] == aN) & (cand["away_n"] == hN))
            ]
            if not same.empty:
                hit = same.iloc[0]
                break
        if hit is None:
            continue
        if _norm(hit["home"]) == hN:
            hs, as_ = int(m["home_score"]), int(m["away_score"])
        else:
            hs, as_ = int(m["away_score"]), int(m["home_score"])
        matched.append({
            "match_no": int(hit["match_no"]),
            "date": hit["date_only"],
            "home": hit["home"], "away": hit["away"],
            "home_score": hs, "away_score": as_,
        })

    df = (pd.DataFrame(matched)
          .drop_duplicates(subset=["match_no"], keep="last")
          .sort_values("match_no")
          .reset_index(drop=True))
    df.to_csv(DATA / "results_actual.csv", index=False)
    dt = int(time.time() - t0)
    print(f"[refresh] sync results: {len(df)} entries ({dt}s)")


def reload_api() -> None:
    try:
        with httpx.Client(timeout=10) as c:
            r = c.post("http://127.0.0.1:8001/reload-models")
        print(f'[refresh] reload-models: {r.json().get("status", r.status_code)}')
    except Exception as e:
        print(f"[refresh] reload-models skipped (API down?): {e}")


def main() -> int:
    step("FIFA schedule refresh",  [sys.executable, "15_fifa_schedule.py"])
    step("martj42 pull",           [sys.executable, "01_fetch_history.py"])
    sync_results_from_martj42()
    step("Cascade rebuild",        [sys.executable, "14_cascade_forecast.py"])
    reload_api()
    print("[refresh] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
