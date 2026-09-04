"""
Compute running points under the WC Tip Game 2026 rubric (4/3/2/0).

Reads:
  data/results_actual.csv     — match results captured by the bot
  logs/predictions.csv        — every prediction the bot has sent

Writes a per-match scoreboard + cumulative total to stdout.

Usage:
  python score_tracker.py            # summary + per-match table
  python score_tracker.py --json     # JSON for API consumption
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"


def score_tip(tip: str, actual_h: int, actual_a: int) -> int:
    """Same logic as bot_notify.score_tip — kept independent so this script
    works even if the bot module is missing."""
    if not isinstance(tip, str) or "-" not in tip:
        return 0
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
        return 3
    if (ht - at) == (actual_h - actual_a):
        return 3
    return 2


def points_breakdown(tip: str, h: int, a: int) -> str:
    p = score_tip(tip, h, a)
    return {
        4: "✓✓ EXACT",
        3: "✓ goal-diff" if (str(h) != str(a)) else "✓ draw",
        2: "✓ tendency only",
        0: "✗ wrong",
    }[p]


def main() -> int:
    json_out = "--json" in sys.argv

    actual_path = DATA_DIR / "results_actual.csv"
    preds_path = LOG_DIR / "predictions.csv"
    if not actual_path.exists():
        print("No results yet.")
        return 0

    actual = pd.read_csv(actual_path)
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()

    rows = []
    for _, r in actual.iterrows():
        mn = int(r["match_no"])
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        sub = preds[preds["match_no"] == mn] if not preds.empty else pd.DataFrame()
        if sub.empty:
            tip = None; conf = None; predicted = None
        else:
            # Take the LATEST prediction for that match (refined trumps initial)
            r2 = sub.sort_values("logged_at").iloc[-1]
            tip = r2.get("tip")
            conf = r2.get("confidence")
            predicted = r2.get("predicted")
        pts = score_tip(tip, hs, as_) if tip else 0
        rows.append({
            "match_no": mn,
            "home": r["home"], "away": r["away"],
            "actual": f"{hs}-{as_}",
            "tip": tip or "(none)",
            "points": pts,
            "result": points_breakdown(tip, hs, as_) if tip else "(no prediction logged)",
        })

    df = pd.DataFrame(rows)
    total = int(df["points"].sum())
    max_possible = 4 * len(df)
    if json_out:
        print(json.dumps({
            "total_points": total,
            "max_possible": max_possible,
            "matches_scored": len(df),
            "matches": rows,
        }, indent=2))
        return 0

    print(f"=== Per-match scorecard ({len(df)} matches) ===")
    for r in rows:
        marker = {4: "***", 3: "++", 2: "+", 0: "-"}[r["points"]]
        line = f"  M{r['match_no']:>3}  {marker:>3}  {r['home']:>22s} {r['actual']:>4s} {r['away']:<22s}  tip={r['tip']:>4s}  {r['points']} pts  ({r['result']})"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('ascii', 'replace').decode('ascii'))
    print()
    print(f"TOTAL: {total} / {max_possible} pts ({100*total/max(max_possible,1):.1f}%)")
    if rows:
        avg = total / len(rows)
        try:
            print(f"Average: {avg:.2f} pts/match  (perfect = 4.00, random ~0.7)")
        except UnicodeEncodeError:
            print(f"Average: {avg:.2f} pts/match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
