"""
Auto-retune the tip-optimiser hyperparameters when enough new results are in.

Reads:
  - data/results_actual.csv          (which matches have been played)
  - data/last_auto_tune.json         (the count at last tune; empty if never)
  - .env                             (AUTO_TUNE_INTERVAL, default 10)

If matches_played - last_tuned_count >= AUTO_TUNE_INTERVAL, runs the 3D grid
search from tune_draw_boost.run_3d_grid() and writes the optimal
{ODDS_BLEND, DC_RHO, DRAW_BOOST, CAL_BLEND} to data/auto_tune_config.json.

tip_optimizer.py reads that JSON if present (overriding env vars), so the new
config takes effect on the next predict_tip() call after the API hot-reload.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DATA = Path(__file__).parent / "data"
LAST_TUNE_PATH = DATA / "last_auto_tune.json"
CONFIG_PATH = DATA / "auto_tune_config.json"
INTERVAL = int(os.environ.get("AUTO_TUNE_INTERVAL", "1"))


def _played_count() -> int:
    p = DATA / "results_actual.csv"
    if not p.exists():
        return 0
    return len(pd.read_csv(p))


def _last_tuned() -> int:
    if not LAST_TUNE_PATH.exists():
        return 0
    try:
        return int(json.loads(LAST_TUNE_PATH.read_text()).get("matches", 0))
    except Exception:
        return 0


def _notify(msg: str) -> None:
    """Best-effort Telegram notification."""
    try:
        import httpx
        tok = os.environ.get("FIFA_BOT_TOKEN", "")
        chat = os.environ.get("FIFA_BOT_CHAT_ID", "")
        print(msg.encode("ascii", "replace").decode("ascii"))
        if tok and chat:
            httpx.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
                timeout=10,
            )
    except Exception as e:
        print(f"[auto_tune] notify failed: {e}")


def run_tune() -> dict | None:
    """Execute the 3D grid search and return the best config dict."""
    import importlib
    sys.path.insert(0, str(Path(__file__).parent))
    tune = importlib.import_module("tune_draw_boost")
    actual = pd.read_csv(DATA / "results_actual.csv")
    schedule = pd.read_csv(DATA / "wc2026_fifa_schedule.csv")
    by_mn = schedule.set_index("match_no").to_dict("index")

    blends = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rhos = [-0.05, -0.10, -0.15]
    boosts = [1.0, 1.25, 1.5, 1.75, 2.0]

    best = None
    n_errors = 0
    first_err: str | None = None
    for blend in blends:
        for rho in rhos:
            for boost in boosts:
                os.environ["ODDS_BLEND"] = str(blend)
                os.environ["DC_RHO"] = str(rho)
                os.environ["DRAW_BOOST"] = str(boost)
                os.environ["CAL_BLEND"] = "0.35"
                # The auto-tune mechanism (data/auto_tune_config.json) overrides
                # env vars in tip_optimizer — temporarily disable it during
                # tuning so each combo actually uses the env values being swept.
                os.environ["AUTO_TUNE_DISABLE"] = "1"
                if "tip_optimizer" in sys.modules:
                    del sys.modules["tip_optimizer"]
                tip_opt = importlib.import_module("tip_optimizer")
                tip_opt.invalidate_cache()
                total = 0
                for _, r in actual.sort_values("match_no").iterrows():
                    mn = int(r["match_no"])
                    sch = by_mn.get(mn, {})
                    try:
                        res = tip_opt.predict_tip(
                            home=r["home"], away=r["away"],
                            date=datetime.strptime(r["date"], "%Y-%m-%d"),
                            tournament_class="wc_finals",
                            city=str(sch.get("stadium_name", "")),
                            neutral=True,
                            match_num_in_tournament=mn,
                            referee=str(sch.get("referee") or ""),
                            is_knockout=False,
                        )
                        total += tune.score_tip(
                            res["best_tip"], int(r["home_score"]), int(r["away_score"]),
                        )
                    except Exception as e:
                        n_errors += 1
                        if first_err is None:
                            import traceback
                            first_err = f"M{mn} ({r['home']} vs {r['away']}): {type(e).__name__}: {e}\n" + traceback.format_exc()
                if best is None or total > best["pts"]:
                    best = {
                        "ODDS_BLEND": blend, "DC_RHO": rho,
                        "DRAW_BOOST": boost, "CAL_BLEND": 0.35,
                        "pts": total, "matches": len(actual),
                    }
    if n_errors > 0:
        print(f"[auto_tune] WARNING: {n_errors} predict_tip exceptions swallowed.")
        print(f"[auto_tune] First error:\n{first_err}")
    return best


def main() -> int:
    played = _played_count()
    last = _last_tuned()
    delta = played - last
    print(f"[auto_tune] played={played}  last_tuned_at={last}  interval={INTERVAL}  delta={delta}")
    if delta < INTERVAL:
        print(f"[auto_tune] skip — need {INTERVAL - delta} more results before next tune")
        return 0

    _notify(
        f"\U0001f527 <b>Auto-tuning</b> on {played} played matches "
        f"(last tune at {last}). This takes ~2-3 minutes..."
    )
    best = run_tune()
    if best is None:
        print("[auto_tune] no valid config found")
        return 1

    CONFIG_PATH.write_text(json.dumps(best, indent=2))
    LAST_TUNE_PATH.write_text(json.dumps({
        "matches": played, "tuned_at": datetime.utcnow().isoformat(),
        "best": best,
    }, indent=2))

    avg = best["pts"] / max(best["matches"], 1)
    _notify(
        f"✅ <b>Auto-tune complete</b>\n"
        f"ODDS_BLEND = {best['ODDS_BLEND']:.2f}\n"
        f"DC_RHO     = {best['DC_RHO']:.3f}\n"
        f"DRAW_BOOST = {best['DRAW_BOOST']:.2f}\n"
        f"Empirical: {best['pts']}/{best['matches']} pts "
        f"({avg:.2f}/match)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
