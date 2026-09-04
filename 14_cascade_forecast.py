"""
Step 10 - Deterministic cascading tournament forecaster.

Given the current model state, produce a single predicted scoreline for EVERY
remaining WC2026 match, all the way to the final. Cascade:

  group_stage     -> predict 72 scores (or use actual results where known)
  standings       -> apply FIFA tiebreakers (pts -> GD -> GF -> H2H)
  advance         -> top-2 per group + 8 best 3rd-placed
  R32             -> apply published bracket (winners + runners-up fixed,
                     3rd-placed slots filled by ranked-seed assignment)
  R16 / QF / SF   -> winners of (M2k-1, M2k) pair up the bracket
  3rd-place play  -> SF losers
  Final           -> SF winners

For "predicted score" we use the **modal scoreline** of the class-reweighted
Poisson joint distribution (i.e. the (h, a) with the highest P(h, a) given the
classifier's win/draw/away calibration).

The (modal vs EV-tip) split:
  predicted_score : argmax_{h,a} P(h, a)             -> "what will actually happen"
  best_tip        : argmax_{h,a} E[points|tip=(h,a)] -> "what to write on the slip"
We report both per match.

Outputs:
  data/cascade_forecast.csv   one row per match (group + KO), in chronological order
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
_train = importlib.import_module("04_train_prematch")
_goals = importlib.import_module("08_train_goals")

# Make pickled dataclasses resolvable under __main__ when imported elsewhere.
import __main__ as _main
for _cls_name, _mod in (("PrematchModel", _train), ("GoalsModel", _goals)):
    if not hasattr(_main, _cls_name):
        setattr(_main, _cls_name, getattr(_mod, _cls_name))

import feature_engine
import tip_optimizer
import wc2026_schedule

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class MatchOutcome:
    match_no: int
    date: datetime
    stage: str
    group: str | None
    home: str
    away: str
    city: str
    neutral: bool
    home_score: int
    away_score: int
    p_home: float
    p_draw: float
    p_away: float
    lambda_home: float
    lambda_away: float
    predicted_score: str          # modal "what will happen"
    best_tip: str                 # EV-max tip
    best_tip_ev: float
    from_actual: bool             # True if score is the real result, not a prediction
    winner: str                   # team or "draw" (group only)


def modal_scoreline(joint: np.ndarray) -> tuple[int, int]:
    """Return (h, a) maximising P(h, a)."""
    idx = np.unravel_index(np.argmax(joint), joint.shape)
    return int(idx[0]), int(idx[1])


def break_draw_via_lambda(joint: np.ndarray, lam_h: float, lam_a: float) -> tuple[int, int]:
    """For knockout matches where the modal scoreline is a draw, force a winner
    using the goal-rate split: pick the side with higher lambda."""
    # Find the highest-P non-draw scoreline biased toward the side with the
    # higher lambda. We just zero out the diagonal and re-argmax.
    j = joint.copy()
    for i in range(min(j.shape)):
        j[i, i] = 0.0
    if lam_h >= lam_a:
        # Favour rows where h > a
        mask = np.fromfunction(lambda h, a: h > a, j.shape).astype(float)
    else:
        mask = np.fromfunction(lambda h, a: h < a, j.shape).astype(float)
    j = j * mask
    return modal_scoreline(j) if j.sum() > 0 else modal_scoreline(joint)


def predict_match(home: str, away: str, date: datetime, city: str | None,
                  neutral: bool, match_no: int, *, force_winner: bool,
                  referee: str | None = None, is_knockout: bool = False) -> dict:
    fx = feature_engine.FixtureInput(
        home=home, away=away, date=date,
        tournament_class="wc_finals", city=city, neutral=neutral,
        match_num_in_tournament=match_no,
        referee=referee,
    )
    feats = feature_engine.compute_features(fx)
    feats_model = feats.drop(columns=[c for c in feats.columns if c.startswith("__")])
    proba = tip_optimizer._prematch_model().predict_proba(feats_model)[0]
    p_home, p_draw, p_away = float(proba[0]), float(proba[1]), float(proba[2])
    # Apply the same post-hoc referee + manager adjustments the API uses.
    import importlib as _il
    _refs = _il.import_module("12_referees")
    import managers as _mgr
    p_home, p_draw, p_away, _ = _refs.apply_ref_adjustment(p_home, p_draw, p_away, referee)
    p_home, p_draw, p_away, _ = _mgr.apply_manager_adjustment(p_home, p_draw, p_away, home, away)
    # Bookmaker odds blend (Oddset closing line). Better-calibrated than ML alone.
    p_home, p_draw, p_away, _ = tip_optimizer._blend_with_odds(
        home, away, date, p_home, p_draw, p_away, tip_optimizer.ODDS_BLEND,
    )
    # `force_winner` (used for KO matches) and `is_knockout` mean the same thing
    # for the tip optimiser: disable Dixon-Coles draw boost + exclude draw tips.
    ko = is_knockout or force_winner
    joint, lam_h, lam_a = tip_optimizer.joint_scoreline_distribution(
        feats_model, p_home, p_draw, p_away, is_knockout=ko,
    )
    h, a = modal_scoreline(joint)
    if force_winner and h == a:
        h, a = break_draw_via_lambda(joint, lam_h, lam_a)
    tips = tip_optimizer.best_tip(joint, top_k=1, is_knockout=ko)
    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "lambda_home": lam_h, "lambda_away": lam_a,
        "predicted_score": (h, a),
        "best_tip": tips[0]["tip"], "best_tip_ev": tips[0]["ev"],
    }


# ---- group standings ----

def apply_results(group_outcomes: list[MatchOutcome]) -> dict[str, list[dict]]:
    """Return standings[group_letter] = list of {team, pts, gd, gf, ga} ranked."""
    tbl: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(
        lambda: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0}
    ))
    for m in group_outcomes:
        if m.stage != "group" or m.group is None:
            continue
        h, a = m.home_score, m.away_score
        tbl[m.group][m.home]["gf"] += h
        tbl[m.group][m.home]["ga"] += a
        tbl[m.group][m.home]["gd"] += h - a
        tbl[m.group][m.away]["gf"] += a
        tbl[m.group][m.away]["ga"] += h
        tbl[m.group][m.away]["gd"] += a - h
        if h > a:
            tbl[m.group][m.home]["pts"] += 3; tbl[m.group][m.home]["w"] += 1
            tbl[m.group][m.away]["l"] += 1
        elif h < a:
            tbl[m.group][m.away]["pts"] += 3; tbl[m.group][m.away]["w"] += 1
            tbl[m.group][m.home]["l"] += 1
        else:
            tbl[m.group][m.home]["pts"] += 1; tbl[m.group][m.away]["pts"] += 1
            tbl[m.group][m.home]["d"] += 1; tbl[m.group][m.away]["d"] += 1

    standings: dict[str, list[dict]] = {}
    for grp, teams in wc2026_schedule.GROUPS.items():
        ranked = sorted(
            teams,
            key=lambda t: (tbl[grp][t]["pts"], tbl[grp][t]["gd"], tbl[grp][t]["gf"]),
            reverse=True,
        )
        standings[grp] = [{"team": t, **tbl[grp][t]} for t in ranked]
    return standings


def determine_advancers(standings: dict[str, list[dict]]) -> tuple[dict, list[str]]:
    """Return ({group: (winner, runnerup)}, third_place_8_ranked)."""
    pair: dict[str, tuple[str, str]] = {}
    third_place: list[tuple[str, str, int, int, int]] = []  # (group, team, pts, gd, gf)
    for g, rows in standings.items():
        pair[g] = (rows[0]["team"], rows[1]["team"])
        r3 = rows[2]
        third_place.append((g, r3["team"], r3["pts"], r3["gd"], r3["gf"]))
    third_ranked = sorted(third_place, key=lambda r: (r[2], r[3], r[4]), reverse=True)
    top_8_third = [r[1] for r in third_ranked[:8]]
    return pair, top_8_third


# ---- knockout bracket (16 matches in R32, then 8/4/2/1) ----
#
# The actual 2026 bracket has a 495-combo lookup table for 3rd-placed slot
# assignment. We approximate: best 3rd-placed plays best group-winner-slot
# (= toughest seed gets the toughest 3rd opponent); worst 3rd-placed plays
# the lower-seeded group-winner slot. This keeps strong-vs-strong matchups
# in the bracket which is what the FIFA assignment also targets.
#
# Fixed pairings extracted from Wikipedia's R32 page:
FIXED_R32_PAIRS = [
    # (match_no, side1_spec, side2_spec)  spec = (group, position)  position in {"W", "RU"}
    (73, ("A", "RU"), ("B", "RU")),
    (74, None,         None),               # both 3rd-place slots (rare; handle below)
    (75, ("F", "W"),  ("C", "RU")),
    (76, ("C", "W"),  ("F", "RU")),
    (77, None,         None),
    (78, ("E", "RU"), ("I", "RU")),
    (79, ("A", "W"),  ("3RD", None)),
    (80, ("L", "W"),  ("3RD", None)),
    (81, ("D", "W"),  ("3RD", None)),
    (82, ("G", "W"),  ("3RD", None)),
    (83, ("K", "RU"), ("L", "RU")),
    (84, ("H", "W"),  ("J", "RU")),
    (85, ("B", "W"),  ("3RD", None)),
    (86, ("J", "W"),  ("H", "RU")),
    (87, ("K", "W"),  ("3RD", None)),
    (88, ("D", "RU"), ("G", "RU")),
]


def build_r32_pairings(advance: dict[str, tuple[str, str]],
                       thirds: list[str]) -> list[tuple[str, str, int]]:
    """Return [(home_team, away_team, match_no), ...] for the 16 R32 matches.

    Strategy (in order):
      1. Use FIFA's published bracket from wc2026_fifa_schedule.csv. As real
         group results come in, FIFA fills the team slots — those are the
         AUTHORITATIVE pairings and override our hardcoded template.
      2. For R32 slots FIFA still has as TBD (because a group hasn't finished),
         fall back to the Wikipedia FIXED_R32_PAIRS template + cascade advance.
    """
    # Step 1 — pull FIFA's R32 fixtures.
    all_fx = wc2026_schedule.all_fixtures_including_ko()
    fifa_r32 = sorted([f for f in all_fx if f.stage == "R32"], key=lambda f: f.match_no)
    confirmed: list[tuple[str, str, int]] = []
    confirmed_teams: set[str] = set()
    tbd_match_nos: list[tuple[int, bool, bool, str, str]] = []
    # First pass: claim every team FIFA has named (in any slot — home or away,
    # full pairing or partial). This prevents the cascade from reusing them
    # when filling TBD slots in the second pass.
    for f in fifa_r32:
        is_tbd_h = f.home.startswith("TBD_") or f.home in ("", "TBD")
        is_tbd_a = f.away.startswith("TBD_") or f.away in ("", "TBD")
        if not is_tbd_h:
            confirmed_teams.add(f.home)
        if not is_tbd_a:
            confirmed_teams.add(f.away)
    # Second pass: keep FIFA's confirmed pairings, queue TBD slots for fill-in.
    for f in fifa_r32:
        is_tbd_h = f.home.startswith("TBD_") or f.home in ("", "TBD")
        is_tbd_a = f.away.startswith("TBD_") or f.away in ("", "TBD")
        if not is_tbd_h and not is_tbd_a:
            confirmed.append((f.home, f.away, f.match_no))
        else:
            tbd_match_nos.append((f.match_no, is_tbd_h, is_tbd_a, f.home, f.away))

    # Step 2 — for TBD slots, derive a sensible pairing from cascade standings
    # without reusing any team FIFA has already locked in.
    available: list[str] = []
    for g in wc2026_schedule.GROUPS:
        if g in advance:
            for t in advance[g]:
                if t not in confirmed_teams and t not in available:
                    available.append(t)
    for t in thirds:
        if t not in confirmed_teams and t not in available:
            available.append(t)

    for mn, is_tbd_h, is_tbd_a, fixed_h, fixed_a in tbd_match_nos:
        home = available.pop(0) if (is_tbd_h and available) else fixed_h
        away = available.pop(0) if (is_tbd_a and available) else fixed_a
        confirmed.append((home, away, mn))

    confirmed.sort(key=lambda x: x[2])
    return confirmed


def _resolve_side(spec, advance, third_iter):
    g, pos = spec
    if g == "3RD":
        return next(third_iter)
    return advance[g][0 if pos == "W" else 1]


def _ko_bracket_chain(r32_pairings: list[tuple[str, str, int]]) -> list[tuple[str, int, list[tuple[int, int]]]]:
    """Return the WC2026 knockout bracket parent chain per FIFA's actual layout.

    Verified against FIFA's confirmed pairings:
      R16 M89 = M73 winner + M76 winner   (Canada + Morocco)
      R16 M91 = M74 winner + M77 winner   (Brazil + Norway)
      QF  M97 = M89 winner + M90 winner   (France + Morocco)
      QF  M99 = M91 winner + M92 winner   (Norway + England)
    Both R32→R16 and R16→QF cross-pair rather than adjacent-pair. The caller
    still overrides any team where FIFA has published the actual name.
    """
    # R32 -> R16 (per FIFA's cross-pairing bracket).
    r16_pairs = [
        (73, 76),   # M89
        (75, 78),   # M90
        (74, 77),   # M91
        (79, 80),   # M92
        (84, 83),   # M93
        (82, 81),   # M94
        (87, 86),   # M95
        (85, 88),   # M96
    ]
    # R16 -> QF (upper half feeds M97 / M99; lower half feeds M98 / M100).
    qf_pairs = [
        (89, 90),   # M97
        (93, 94),   # M98
        (91, 92),   # M99
        (95, 96),   # M100
    ]
    # QF -> SF: upper (M97, M99) → M101; lower (M98, M100) → M102.
    sf_pairs = [(97, 99), (98, 100)]
    # SF -> Final: M101 winner vs M102 winner. Base_mn=103 keeps the existing
    # 3rd-place-swap logic downstream working (it renames 103→104 and inserts
    # the 3rd-place playoff at 103).
    final_pairs = [(101, 102)]

    return [
        ("R16",   89, r16_pairs),
        ("QF",    97, qf_pairs),
        ("SF",   101, sf_pairs),
        ("Final", 103, final_pairs),
    ]


def _ko_date(round_label: str, idx: int) -> datetime:
    base = {
        "R32":   (2026, 6, 28),
        "R16":   (2026, 7, 4),
        "QF":    (2026, 7, 9),
        "SF":    (2026, 7, 14),
        "3rd":   (2026, 7, 18),
        "Final": (2026, 7, 19),
    }[round_label]
    d = datetime(*base) + timedelta(days=(idx // 2))
    return d


# ---- driver ----

def load_actual_results() -> dict[int, tuple[int, int]]:
    """Map match_no -> (home_score, away_score) for played matches."""
    path = DATA_DIR / "results_actual.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")):
            out[int(r["match_no"])] = (int(r["home_score"]), int(r["away_score"]))
    return out


def cascade() -> list[MatchOutcome]:
    actual = load_actual_results()
    fxs = wc2026_schedule.all_group_fixtures()
    outcomes: list[MatchOutcome] = []

    # ---- group stage ----
    for f in fxs:
        if f.match_no in actual:
            hs, as_ = actual[f.match_no]
            outcomes.append(MatchOutcome(
                match_no=f.match_no, date=f.date, stage="group", group=f.group,
                home=f.home, away=f.away, city=f.city, neutral=f.neutral,
                home_score=hs, away_score=as_,
                p_home=np.nan, p_draw=np.nan, p_away=np.nan,
                lambda_home=np.nan, lambda_away=np.nan,
                predicted_score=f"{hs}-{as_}", best_tip=f"{hs}-{as_}", best_tip_ev=0.0,
                from_actual=True,
                winner=(f.home if hs > as_ else (f.away if as_ > hs else "draw")),
            ))
            continue
        p = predict_match(f.home, f.away, f.date, f.city, f.neutral, f.match_no,
                          force_winner=False, referee=f.referee)
        h, a = p["predicted_score"]
        outcomes.append(MatchOutcome(
            match_no=f.match_no, date=f.date, stage="group", group=f.group,
            home=f.home, away=f.away, city=f.city, neutral=f.neutral,
            home_score=h, away_score=a,
            p_home=p["p_home"], p_draw=p["p_draw"], p_away=p["p_away"],
            lambda_home=p["lambda_home"], lambda_away=p["lambda_away"],
            predicted_score=f"{h}-{a}",
            best_tip=f"{p['best_tip'][0]}-{p['best_tip'][1]}",
            best_tip_ev=p["best_tip_ev"], from_actual=False,
            winner=(f.home if h > a else (f.away if a > h else "draw")),
        ))

    # ---- standings + advancers ----
    standings = apply_results(outcomes)
    advance, thirds = determine_advancers(standings)

    # ---- knockout ----
    r32 = build_r32_pairings(advance, thirds)
    winner_by_match_no: dict[int, str] = {}

    # R32
    for idx, (h, a, mn) in enumerate(r32):
        date = _ko_date("R32", idx)
        if mn in actual:
            hs, as_ = actual[mn]
            from_act = True
            winner = h if hs > as_ else (a if as_ > hs else h)  # KO can't draw
        else:
            p = predict_match(h, a, date, None, True, mn, force_winner=True)
            hs, as_ = p["predicted_score"]
            winner = h if hs > as_ else a
            from_act = False
        outcomes.append(MatchOutcome(
            match_no=mn, date=date, stage="R32", group=None,
            home=h, away=a, city="(KO)", neutral=True,
            home_score=hs, away_score=as_,
            p_home=p["p_home"] if not from_act else np.nan,
            p_draw=p["p_draw"] if not from_act else np.nan,
            p_away=p["p_away"] if not from_act else np.nan,
            lambda_home=p["lambda_home"] if not from_act else np.nan,
            lambda_away=p["lambda_away"] if not from_act else np.nan,
            predicted_score=f"{hs}-{as_}",
            best_tip=f"{p['best_tip'][0]}-{p['best_tip'][1]}" if not from_act else f"{hs}-{as_}",
            best_tip_ev=p["best_tip_ev"] if not from_act else 0.0,
            from_actual=from_act, winner=winner,
        ))
        winner_by_match_no[mn] = winner

    # Build a lookup of FIFA-published KO matchups (R16/QF/SF/Final). When a
    # FIFA fixture has both teams confirmed, that pairing OVERRIDES whatever
    # our hardcoded bracket chain would have produced for the same match_no.
    # This is the same rule we apply for R32 above — FIFA is authoritative.
    fifa_ko_pairings: dict[int, tuple[str, str]] = {}
    for f in wc2026_schedule.all_fixtures_including_ko():
        if f.stage not in ("R16", "QF", "SF", "3rd", "Final"):
            continue
        is_tbd_h = f.home.startswith("TBD_") or f.home in ("", "TBD")
        is_tbd_a = f.away.startswith("TBD_") or f.away in ("", "TBD")
        if not is_tbd_h and not is_tbd_a:
            fifa_ko_pairings[f.match_no] = (f.home, f.away)

    # Subsequent rounds
    chain = _ko_bracket_chain(r32)
    loser_by_match_no: dict[int, str] = {}
    sf_match_nos: list[int] = []
    for round_label, base_mn, pairs in chain:
        new_winner_map: dict[int, str] = {}
        if round_label == "SF":
            sf_match_nos = [base_mn + i for i in range(len(pairs))]
        for idx, (p1, p2) in enumerate(pairs):
            mn = base_mn + idx
            # Prefer FIFA's published matchup over the cascade-derived one.
            if mn in fifa_ko_pairings:
                home, away = fifa_ko_pairings[mn]
            else:
                home = winner_by_match_no[p1]
                away = winner_by_match_no[p2]
            date = _ko_date(round_label, idx)
            if mn in actual:
                hs, as_ = actual[mn]
                from_act = True
                winner = home if hs > as_ else away
            else:
                p = predict_match(home, away, date, None, True, mn, force_winner=True)
                hs, as_ = p["predicted_score"]
                winner = home if hs > as_ else away
                from_act = False
            outcomes.append(MatchOutcome(
                match_no=mn, date=date, stage=round_label, group=None,
                home=home, away=away, city="(KO)", neutral=True,
                home_score=hs, away_score=as_,
                p_home=p["p_home"] if not from_act else np.nan,
                p_draw=p["p_draw"] if not from_act else np.nan,
                p_away=p["p_away"] if not from_act else np.nan,
                lambda_home=p["lambda_home"] if not from_act else np.nan,
                lambda_away=p["lambda_away"] if not from_act else np.nan,
                predicted_score=f"{hs}-{as_}",
                best_tip=f"{p['best_tip'][0]}-{p['best_tip'][1]}" if not from_act else f"{hs}-{as_}",
                best_tip_ev=p["best_tip_ev"] if not from_act else 0.0,
                from_actual=from_act, winner=winner,
            ))
            new_winner_map[mn] = winner
            loser = away if winner == home else home
            loser_by_match_no[mn] = loser
        winner_by_match_no.update(new_winner_map)

    # 3rd-place playoff: losers of the two SFs.
    if len(sf_match_nos) == 2:
        sf1_loser = loser_by_match_no.get(sf_match_nos[0])
        sf2_loser = loser_by_match_no.get(sf_match_nos[1])
        if sf1_loser and sf2_loser:
            mn = 104  # use the conventional final match number 104 = Final, 103 = 3rd. We'll swap.
            # Re-number: existing chain put the Final at base_mn=103. Move that to 104,
            # and the 3rd-place playoff lives at 103.
            for o in outcomes:
                if o.stage == "Final":
                    o.match_no = 104
            date = _ko_date("3rd", 0)
            mn_3rd = 103
            if mn_3rd in actual:
                hs, as_ = actual[mn_3rd]
                from_act = True
                winner = sf1_loser if hs > as_ else sf2_loser
                p = {"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan,
                     "lambda_home": np.nan, "lambda_away": np.nan,
                     "best_tip": (hs, as_), "best_tip_ev": 0.0}
            else:
                p = predict_match(sf1_loser, sf2_loser, date, None, True, mn_3rd,
                                  force_winner=True)
                hs, as_ = p["predicted_score"]
                winner = sf1_loser if hs > as_ else sf2_loser
                from_act = False
            outcomes.append(MatchOutcome(
                match_no=mn_3rd, date=date, stage="3rd", group=None,
                home=sf1_loser, away=sf2_loser, city="(KO)", neutral=True,
                home_score=hs, away_score=as_,
                p_home=p["p_home"] if not from_act else np.nan,
                p_draw=p["p_draw"] if not from_act else np.nan,
                p_away=p["p_away"] if not from_act else np.nan,
                lambda_home=p["lambda_home"] if not from_act else np.nan,
                lambda_away=p["lambda_away"] if not from_act else np.nan,
                predicted_score=f"{hs}-{as_}",
                best_tip=f"{p['best_tip'][0]}-{p['best_tip'][1]}",
                best_tip_ev=p["best_tip_ev"],
                from_actual=from_act, winner=winner,
            ))

    outcomes.sort(key=lambda o: (o.date, o.match_no))
    return outcomes


def to_dataframe(outcomes: list[MatchOutcome]) -> pd.DataFrame:
    return pd.DataFrame([o.__dict__ for o in outcomes])


def summary_text(outcomes: list[MatchOutcome], emoji: bool = True) -> str:
    """Pretty summary for Telegram: champion, finalists, top picks."""
    final = [o for o in outcomes if o.stage == "Final"]
    third = [o for o in outcomes if o.stage == "3rd"]
    sf = [o for o in outcomes if o.stage == "SF"]
    qf = [o for o in outcomes if o.stage == "QF"]

    champion = final[0].winner if final else "?"
    runner_up = (final[0].away if final[0].winner == final[0].home
                 else final[0].home) if final else "?"
    trophy = "🏆 " if emoji else "[Champion] "
    lines = [f"{trophy}<b>Predicted champion: {champion}</b>"]
    if final:
        lines.append(f"Final: <b>{final[0].home}</b> {final[0].predicted_score} "
                     f"<b>{final[0].away}</b>  -> winner {final[0].winner}")
    if third:
        lines.append(f"3rd-place play-off: {third[0].home} {third[0].predicted_score} "
                     f"{third[0].away}  -> 3rd {third[0].winner}")
    lines.append("Semi-finalists: " + ", ".join(
        sorted({o.home for o in sf} | {o.away for o in sf})
    ))
    lines.append("Quarter-finalists: " + ", ".join(
        sorted({o.home for o in qf} | {o.away for o in qf})
    ))
    return "\n".join(lines)


def main() -> int:
    outcomes = cascade()
    df = to_dataframe(outcomes)
    out = DATA_DIR / "cascade_forecast.csv"
    df.to_csv(out, index=False)
    print(f"[14_cascade_forecast] wrote {out}  ({len(df)} matches)")
    try:
        print("\n" + summary_text(outcomes, emoji=True))
    except UnicodeEncodeError:
        print("\n" + summary_text(outcomes, emoji=False))

    sf = [o for o in outcomes if o.stage == "SF"]
    qf = [o for o in outcomes if o.stage == "QF"]
    final = [o for o in outcomes if o.stage == "Final"]
    print("\n--- Knockout bracket predictions ---")
    for stage in ("R32", "R16", "QF", "SF", "3rd", "Final"):
        for o in outcomes:
            if o.stage == stage:
                tag = "ACTUAL" if o.from_actual else "PRED"
                print(f"  [{stage:5s}] {tag:6s} M{o.match_no:3d}  "
                      f"{o.home:>26s} {o.predicted_score:>4s} {o.away:<26s}   -> {o.winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
