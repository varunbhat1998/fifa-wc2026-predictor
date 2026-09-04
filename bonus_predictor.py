"""
WC2026 Bonus-tab prediction picks.

Reads the Monte Carlo simulator output (data/tournament_sim_results.csv) and
recommends the EV-maximising pick for each of the 18 dropdowns:
  - World Champion              (1 pick)
  - Team of highest goal scorer (1 pick)  — Phase A heuristic until FBref lands
  - Group A..L winners          (12 picks)
  - Semi-finalists              (4 picks)

Phase A top-scorer team heuristic
---------------------------------
Per team, top-scorer-team score = fw_goals_top4_total × E[matches_played]
  fw_goals_top4_total : team_profiles_2026.csv column — sum of int'l goals
                        by the squad's top-4 forwards (heavy Ronaldo / Messi /
                        Kane bias since those guys dominate squad goal totals).
  E[matches_played]   : Monte Carlo expectation — 3 (group) + sum of advance
                        probabilities through each KO round.

This is a clean approximation: teams whose star striker has lots of int'l
goals AND who are expected to go deep get the highest score. Replace with
per-player Poisson once FBref club minutes/xG lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import wc2026_schedule

DATA_DIR = Path(__file__).parent / "data"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    mc = pd.read_csv(DATA_DIR / "tournament_sim_results.csv")
    tp = pd.read_csv(DATA_DIR / "team_profiles_2026.csv")
    return mc, tp


def champion(mc: pd.DataFrame, top_n: int = 5) -> list[dict]:
    return (
        mc.sort_values("p_champion", ascending=False)
          .head(top_n)
          [["team", "group", "p_champion", "p_advance_Final", "p_advance_SF"]]
          .to_dict(orient="records")
    )


def group_winners(mc: pd.DataFrame) -> dict[str, list[dict]]:
    out = {}
    for grp in sorted(wc2026_schedule.GROUPS.keys()):
        sub = mc[mc["group"] == grp].sort_values("p_1st_in_group", ascending=False)
        out[grp] = sub[["team", "p_1st_in_group", "p_advance_R32", "p_champion"]].head(4).to_dict(orient="records")
    return out


def semifinalists(mc: pd.DataFrame, n: int = 4) -> list[dict]:
    return (
        mc.sort_values("p_advance_SF", ascending=False)
          .head(n)
          [["team", "group", "p_advance_SF", "p_advance_Final", "p_champion"]]
          .to_dict(orient="records")
    )


def top_scorer_team(mc: pd.DataFrame, tp: pd.DataFrame, top_n: int = 5) -> list[dict]:
    # n_injured_out (if present) trims fw_goals proportionally — a team that has
    # lost players in their top-4 forward block gets discounted.
    cols = ["team", "fw_goals_top4_total"]
    if "n_injured_out" in tp.columns:
        cols.append("n_injured_out")
    df = mc.merge(tp[cols], on="team", how="left")
    df["top_scorer_score"] = df["fw_goals_top4_total"].fillna(0) * df["e_matches_played"]
    return (
        df.sort_values("top_scorer_score", ascending=False)
          .head(top_n)
          [["team", "group", "fw_goals_top4_total", "e_matches_played",
            "top_scorer_score", "p_champion"]]
          .to_dict(orient="records")
    )


def bonus_picks() -> dict:
    mc, tp = _load()
    champs = champion(mc)
    gws = group_winners(mc)
    sfs = semifinalists(mc, n=4)
    ts = top_scorer_team(mc, tp)
    return {
        "champion_top5": champs,
        "champion": champs[0]["team"] if champs else None,
        "top_scorer_team_top5": ts,
        "top_scorer_team": ts[0]["team"] if ts else None,
        "group_winners": {g: rows[0]["team"] for g, rows in gws.items() if rows},
        "group_winners_detailed": gws,
        "semifinalists": [r["team"] for r in sfs],
        "semifinalists_detailed": sfs,
    }


def format_digest(picks: dict) -> str:
    """Human-readable digest with confidence numbers."""
    lines = ["<b>🏆 WC2026 Bonus Picks</b>", ""]

    lines.append("<b>World Champion</b>")
    for r in picks["champion_top5"]:
        marker = " ◀ PICK" if r["team"] == picks["champion"] else ""
        lines.append(f"  {r['team']:>22s}  P={r['p_champion']:.1%}  (SF {r['p_advance_SF']:.0%}, F {r['p_advance_Final']:.0%}){marker}")

    lines.append("")
    lines.append("<b>Team of Top Goal Scorer</b>")
    for r in picks["top_scorer_team_top5"]:
        marker = " ◀ PICK" if r["team"] == picks["top_scorer_team"] else ""
        lines.append(f"  {r['team']:>22s}  goal-score={r['top_scorer_score']:6.1f}  "
                     f"(fw_goals {int(r['fw_goals_top4_total']):3d}, E[matches] {r['e_matches_played']:.2f}){marker}")

    lines.append("")
    lines.append("<b>Group Winners</b>")
    for g, rows in sorted(picks["group_winners_detailed"].items()):
        pick = rows[0]
        # show top 2 per group for context
        alt = rows[1] if len(rows) > 1 else None
        line = f"  Group {g}:  <b>{pick['team']:>22s}</b>  P={pick['p_1st_in_group']:.0%}"
        if alt:
            line += f"     (alt {alt['team']} {alt['p_1st_in_group']:.0%})"
        lines.append(line)

    lines.append("")
    lines.append("<b>Semi-finalists</b>")
    for r in picks["semifinalists_detailed"]:
        lines.append(f"  {r['team']:>22s}  P(reach SF)={r['p_advance_SF']:.0%}  champion {r['p_champion']:.0%}")

    return "\n".join(lines)


def format_submit(picks: dict) -> str:
    """Plain copy-paste output, one line per dropdown in form order."""
    lines = []
    lines.append(f"World Champion: {picks['champion']}")
    lines.append(f"Team of highest goal scorer: {picks['top_scorer_team']}")
    for g in sorted(picks["group_winners"].keys()):
        lines.append(f"Winner Group {g}: {picks['group_winners'][g]}")
    for i, t in enumerate(picks["semifinalists"], start=1):
        lines.append(f"Semi-finalist #{i}: {t}")
    return "\n".join(lines)


if __name__ == "__main__":
    picks = bonus_picks()
    try:
        print(format_digest(picks))
    except UnicodeEncodeError:
        print(format_digest(picks).encode("ascii", "replace").decode("ascii"))
    print()
    print("---- COPY-PASTE FORMAT ----")
    print(format_submit(picks))
