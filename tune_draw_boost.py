"""
Grid-search the DRAW_BOOST parameter against the matches we've already played.

For each candidate boost in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]:
  - Set DRAW_BOOST env var
  - Re-import tip_optimizer
  - Run predict_tip() for each played match
  - Score the resulting tip against the actual result using the rubric
Pick the boost that maximises total points.
"""
from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DATA_DIR = Path(__file__).parent / "data"


def score_tip(tip: str, h: int, a: int) -> int:
    """4 / 3 / 2 / 0 rubric (also 3 for correct-draw-tendency, 4 for exact draw)."""
    try:
        ht, at = (int(x) for x in tip.split("-"))
    except Exception:
        return 0
    tip_w = 1 if ht > at else (-1 if ht < at else 0)
    act_w = 1 if h > a else (-1 if h < a else 0)
    if tip_w != act_w:
        return 0
    if ht == h and at == a:
        return 4
    if tip_w == 0:
        return 3       # both predicted draw, score different
    if (ht - at) == (h - a):
        return 3       # right winner + right margin
    return 2           # right winner only


def run_grid(rhos: list[float]) -> None:
    actual = pd.read_csv(DATA_DIR / "results_actual.csv")
    schedule = pd.read_csv(DATA_DIR / "wc2026_fifa_schedule.csv")
    by_mn = schedule.set_index("match_no").to_dict("index")

    print(f"{'DC_RHO':>7}  {'pts':>4}  per-match (tip -> actual -> pts)")
    print("-" * 80)

    results = []
    for rho in rhos:
        os.environ["DC_RHO"] = str(rho)
        os.environ["DRAW_BOOST"] = "1.0"  # disable old mechanism so tuning is clean
        if "tip_optimizer" in sys.modules:
            del sys.modules["tip_optimizer"]
        tip_opt = importlib.import_module("tip_optimizer")
        tip_opt.invalidate_cache()

        total = 0
        per_match = []
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
                    is_knockout=False,  # all M1-M11 are group-stage
                )
                tip = res["best_tip"]
                pts = score_tip(tip, int(r["home_score"]), int(r["away_score"]))
                total += pts
                per_match.append(f"M{mn}({tip}->{int(r['home_score'])}-{int(r['away_score'])}={pts})")
            except Exception as e:
                per_match.append(f"M{mn}(err:{type(e).__name__})")

        results.append((rho, total, per_match))
        print(f"{rho:>7.3f}  {total:>4d}  " + " ".join(per_match))

    print()
    best = max(results, key=lambda x: x[1])
    print(f"BEST: DC_RHO={best[0]:.3f} -> {best[1]} pts on {len(actual)} matches "
          f"({best[1] / len(actual):.2f} pts/match avg)")


def run_2d_grid(rhos: list[float], boosts: list[float]) -> None:
    """Joint sweep over DC_RHO and DRAW_BOOST. Reports the best combo."""
    actual = pd.read_csv(DATA_DIR / "results_actual.csv")
    schedule = pd.read_csv(DATA_DIR / "wc2026_fifa_schedule.csv")
    by_mn = schedule.set_index("match_no").to_dict("index")

    print(f"{'DC_RHO':>7}  {'BOOST':>5}  {'pts':>4}  picks summary")
    print("-" * 90)

    best = (-1, -1, -1, "")
    for rho in rhos:
        for boost in boosts:
            os.environ["DC_RHO"] = str(rho)
            os.environ["DRAW_BOOST"] = str(boost)
            os.environ["CAL_BLEND"] = "0.35"
            if "tip_optimizer" in sys.modules:
                del sys.modules["tip_optimizer"]
            tip_opt = importlib.import_module("tip_optimizer")
            tip_opt.invalidate_cache()

            total = 0
            picks = []
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
                    tip = res["best_tip"]
                    pts = score_tip(tip, int(r["home_score"]), int(r["away_score"]))
                    total += pts
                    picks.append(f"M{mn}:{tip}={pts}")
                except Exception:
                    picks.append(f"M{mn}:err")
            summary = " ".join(picks)
            print(f"{rho:>7.3f}  {boost:>5.2f}  {total:>4d}  {summary[:80]}")
            if total > best[2]:
                best = (rho, boost, total, summary)

    print()
    print(f"BEST: DC_RHO={best[0]:.3f}  DRAW_BOOST={best[1]:.2f}  "
          f"-> {best[2]} pts on {len(actual)} matches ({best[2]/len(actual):.2f} pts/match)")
    print(f"Full picks: {best[3]}")


def run_3d_grid(rhos: list[float], boosts: list[float], blends: list[float]) -> None:
    """Joint sweep over DC_RHO, DRAW_BOOST, and ODDS_BLEND. Reports best combo."""
    actual = pd.read_csv(DATA_DIR / "results_actual.csv")
    schedule = pd.read_csv(DATA_DIR / "wc2026_fifa_schedule.csv")
    by_mn = schedule.set_index("match_no").to_dict("index")

    print(f"{'BLEND':>5}  {'RHO':>7}  {'BOOST':>5}  {'pts':>4}  {'avg':>5}")
    print("-" * 50)

    best = (-1, -1, -1, -1, "")
    all_results = []
    for blend in blends:
        for rho in rhos:
            for boost in boosts:
                os.environ["DC_RHO"] = str(rho)
                os.environ["DRAW_BOOST"] = str(boost)
                os.environ["ODDS_BLEND"] = str(blend)
                os.environ["CAL_BLEND"] = "0.35"
                if "tip_optimizer" in sys.modules:
                    del sys.modules["tip_optimizer"]
                tip_opt = importlib.import_module("tip_optimizer")
                tip_opt.invalidate_cache()

                total = 0
                picks = []
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
                        tip = res["best_tip"]
                        pts = score_tip(tip, int(r["home_score"]), int(r["away_score"]))
                        total += pts
                        picks.append(f"M{mn}:{tip}={pts}")
                    except Exception:
                        picks.append(f"M{mn}:err")
                avg = total / len(actual)
                summary = " ".join(picks)
                print(f"{blend:>5.2f}  {rho:>7.3f}  {boost:>5.2f}  {total:>4d}  {avg:>5.2f}")
                all_results.append((blend, rho, boost, total, summary))
                if total > best[3]:
                    best = (blend, rho, boost, total, summary)

    print()
    print(f"BEST: ODDS_BLEND={best[0]:.2f}  DC_RHO={best[1]:.3f}  DRAW_BOOST={best[2]:.2f}")
    print(f"      -> {best[3]} pts on {len(actual)} matches ({best[3]/len(actual):.2f} pts/match)")
    print()
    print(f"Picks at optimum: {best[4]}")
    print()
    # Top 10 results
    print("Top 10 configurations:")
    for r in sorted(all_results, key=lambda x: -x[3])[:10]:
        print(f"  BLEND={r[0]:.2f}  RHO={r[1]:>6.3f}  BOOST={r[2]:.2f}  -> {r[3]} pts ({r[3]/len(actual):.2f}/match)")


if __name__ == "__main__":
    # 3D sweep including ODDS_BLEND (bookmaker blend weight).
    run_3d_grid(
        blends=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        rhos=[-0.05, -0.10, -0.15],
        boosts=[1.0, 1.25, 1.5, 1.75, 2.0],
    )
