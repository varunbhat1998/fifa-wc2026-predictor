"""
Tip Game 2026 optimiser.

Scoring rubric (from the user's screenshot):
  win, exact result           -> 4
  win, correct goal diff      -> 3
  win, right tendency only    -> 2
  draw, exact result          -> 4
  draw, correct tendency      -> 3
  wrong tendency              -> 0

Given a fixture we:
  1. Predict per-team goal rate (lambda_h, lambda_a) via the Poisson model.
  2. Build a joint scoreline distribution over (h, a) in {0..MAX_GOALS}^2.
     Using independent Poisson PMFs (a defensible default for international
     football; club correlations are stronger but we are not modelling them).
  3. Reweight that joint so its marginals over {home_win, draw, away_win}
     equal the classifier's calibrated 3-class probabilities. This grafts the
     win-side calibration of the ensemble onto the score distribution of the
     Poisson model — best of both.
  4. For each candidate tip (h_t, a_t), compute expected points.
  5. Return the EV-maximising tip plus the runner-up.
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).parent))
_train = importlib.import_module("04_train_prematch")
_goals = importlib.import_module("08_train_goals")
_refs = importlib.import_module("12_referees")
import managers as _managers_mod

# Models were pickled while their training scripts ran as __main__, so pickle
# resolved the dataclass to `__main__.PrematchModel` / `__main__.GoalsModel`.
# Re-publish those classes on whatever module is currently __main__ so
# pickle.load resolves them when this module is imported elsewhere.
import __main__ as _main
if not hasattr(_main, "PrematchModel"):
    _main.PrematchModel = _train.PrematchModel
if not hasattr(_main, "GoalsModel"):
    _main.GoalsModel = _goals.GoalsModel

import feature_engine

MODEL_DIR = Path(__file__).parent / "models"
MAX_GOALS = 7   # 8x8 grid covers >99.9% of probability mass for international football

# Encoded as integer matrices for the EV calculation.
POINTS_EXACT      = 4
POINTS_GOAL_DIFF  = 3   # winner-side only
POINTS_DRAW_TEND  = 3   # both tip and result are draws but score differs
POINTS_TENDENCY   = 2   # winner-side correct, wrong margin
POINTS_WRONG      = 0


@lru_cache(maxsize=1)
def _prematch_model():
    with open(MODEL_DIR / "prematch_model.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _goals_model():
    with open(MODEL_DIR / "goals_model.pkl", "rb") as f:
        return pickle.load(f)


def invalidate_cache() -> None:
    _prematch_model.cache_clear()
    _goals_model.cache_clear()
    _tournament_class_priors.cache_clear()
    _odds_lookup.cache_clear()
    feature_engine.invalidate_cache()
    _refs.invalidate_cache()
    _managers_mod.invalidate_cache()
    # Re-read the auto-tune config so a fresh tune lands without a process restart.
    _apply_auto_tune_config()


@lru_cache(maxsize=1)
def _points_grid() -> np.ndarray:
    """points[h_tip, a_tip, h_act, a_act] -> int points awarded."""
    n = MAX_GOALS + 1
    pts = np.zeros((n, n, n, n), dtype=np.int8)
    for ht in range(n):
        for at in range(n):
            tip_w = 1 if ht > at else (-1 if ht < at else 0)
            for h in range(n):
                for a in range(n):
                    act_w = 1 if h > a else (-1 if h < a else 0)
                    if tip_w != act_w:
                        continue
                    if ht == h and at == a:
                        pts[ht, at, h, a] = POINTS_EXACT
                    elif tip_w == 0:
                        pts[ht, at, h, a] = POINTS_DRAW_TEND
                    elif (ht - at) == (h - a):
                        pts[ht, at, h, a] = POINTS_GOAL_DIFF
                    else:
                        pts[ht, at, h, a] = POINTS_TENDENCY
    return pts


# --- Dixon-Coles bivariate Poisson correction ---
# Standard football literature model (Dixon & Coles, 1997). Independent
# Poisson under-predicts low-scoring draws because goals aren't actually
# independent at low scores (defensive structures, score-effect, etc.).
# The tau correction biases (0,0), (1,1), (0,1), (1,0) cells:
#
#   tau(0,0) = 1 - lam_h * lam_a * rho
#   tau(0,1) = 1 + lam_h * rho
#   tau(1,0) = 1 + lam_a * rho
#   tau(1,1) = 1 - rho
#   tau(other) = 1
#
# With rho < 0 the correction inflates (0,0) and (1,1) and deflates (0,1)/(1,0),
# matching the tournament-observed pattern (1-1 draws happening when the model
# predicted 1-0 home wins). Tunable via DC_RHO env var (negative for draw bias).
DC_RHO = float(os.environ.get("DC_RHO", "-0.18"))

# Soft Bayesian recalibration of classifier marginals against tournament-observed
# rates. With N played matches showing X% draws, blend the classifier's P(draw)
# towards X with weight CAL_BLEND (0 = classifier only, 1 = empirical only).
CAL_BLEND = float(os.environ.get("CAL_BLEND", "0.35"))

# Additional empirical draw boost layered on top of Dixon-Coles. The DC correction
# only structurally tweaks low-score cells; in a tournament where draws are
# running well above historical average, we want an extra empirical tilt. The
# tuning script (tune_draw_boost.py) finds the boost that maximises observed
# scoring vs played matches. Disabled in knockouts.
DRAW_BOOST = float(os.environ.get("DRAW_BOOST", "1.0"))

# Bookmaker odds blend weight. 0 = model only, 1 = market only. Bet365/Oddset
# closing odds are sourced from data/odds_2026.csv and merged at the
# (p_home, p_draw, p_away) level. The market is generally better-calibrated
# than pure ML, so a non-zero blend usually helps.
ODDS_BLEND = float(os.environ.get("ODDS_BLEND", "0.4"))


def _apply_auto_tune_config() -> None:
    """If 17_auto_tune.py has written an optimal config to disk, override the
    env-var defaults so the live API always uses the latest empirical best.

    Skipped when AUTO_TUNE_DISABLE=1 (used by the tuner itself so each grid
    combo actually exercises the env-var values being swept)."""
    if os.environ.get("AUTO_TUNE_DISABLE", "0") == "1":
        return
    global DC_RHO, CAL_BLEND, DRAW_BOOST, ODDS_BLEND
    path = Path(__file__).parent / "data" / "auto_tune_config.json"
    if not path.exists():
        return
    try:
        import json as _json
        cfg = _json.loads(path.read_text())
    except Exception:
        return
    if "DC_RHO" in cfg:     DC_RHO = float(cfg["DC_RHO"])
    if "CAL_BLEND" in cfg:  CAL_BLEND = float(cfg["CAL_BLEND"])
    if "DRAW_BOOST" in cfg: DRAW_BOOST = float(cfg["DRAW_BOOST"])
    if "ODDS_BLEND" in cfg: ODDS_BLEND = float(cfg["ODDS_BLEND"])


_apply_auto_tune_config()


def _dixon_coles_tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    """DC multiplicative correction for the bivariate Poisson PMF at (h, a)."""
    if h == 0 and a == 0:
        return max(0.01, 1.0 - lam_h * lam_a * rho)
    if h == 0 and a == 1:
        return max(0.01, 1.0 + lam_h * rho)
    if h == 1 and a == 0:
        return max(0.01, 1.0 + lam_a * rho)
    if h == 1 and a == 1:
        return max(0.01, 1.0 - rho)
    return 1.0


@lru_cache(maxsize=1)
def _odds_lookup() -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """Map (date YYYY-MM-DD, home, away) -> (p_home, p_draw, p_away) implied
    from Oddset closing odds (de-vigged). Both team orderings are stored so
    we can match either way."""
    p = Path(__file__).parent / "data" / "odds_2026.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for _, r in df.iterrows():
        ih, id_, ia = float(r["implied_home"]), float(r["implied_draw"]), float(r["implied_away"])
        out[(r["date"], r["home"], r["away"])] = (ih, id_, ia)
        out[(r["date"], r["away"], r["home"])] = (ia, id_, ih)
    return out


def _blend_with_odds(home: str, away: str, date,
                     p_home: float, p_draw: float, p_away: float,
                     weight: float) -> tuple[float, float, float, bool]:
    """Mix classifier probabilities with bookmaker implied probabilities.
    Returns (p_home, p_draw, p_away, blended_flag). blended_flag is True if
    we had odds data for this match, False if we left probs unchanged."""
    if weight <= 0:
        return p_home, p_draw, p_away, False
    lk = _odds_lookup()
    if not lk:
        return p_home, p_draw, p_away, False
    if hasattr(date, "strftime"):
        date_key = date.strftime("%Y-%m-%d")
    else:
        date_key = str(date)[:10]
    # Try the given date plus +/- 1 day (kicktipp uses Berlin local, FIFA uses UTC).
    from datetime import datetime as _dt, timedelta as _td
    for delta in (0, 1, -1):
        try:
            d_try = (_dt.strptime(date_key, "%Y-%m-%d") + _td(days=delta)).strftime("%Y-%m-%d")
        except ValueError:
            d_try = date_key
        hit = lk.get((d_try, home, away))
        if hit:
            break
    else:
        return p_home, p_draw, p_away, False
    oh, od, oa = hit
    w = weight
    ph = (1 - w) * p_home + w * oh
    pd_ = (1 - w) * p_draw + w * od
    pa = (1 - w) * p_away + w * oa
    s = ph + pd_ + pa
    return ph / s, pd_ / s, pa / s, True


@lru_cache(maxsize=1)
def _tournament_class_priors() -> tuple[float, float, float] | None:
    """Observed (home, draw, away) rates from results_actual.csv.
    Returns None if too few matches to be informative."""
    import os.path as _osp
    p = Path(__file__).parent / "data" / "results_actual.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if len(df) < 5:
        return None
    h = (df["home_score"] > df["away_score"]).sum()
    d = (df["home_score"] == df["away_score"]).sum()
    a = (df["home_score"] < df["away_score"]).sum()
    n = len(df)
    return (h / n, d / n, a / n)


def joint_scoreline_distribution(
    feats_row: pd.DataFrame,
    p_home: float, p_draw: float, p_away: float,
    *, dc_rho: float | None = None, is_knockout: bool = False,
) -> np.ndarray:
    """Joint P(home_goals=h, away_goals=a) using Dixon-Coles bivariate Poisson,
    then reweighted to match (recalibrated) classifier class probabilities.

    is_knockout=True disables the DC draw boost (rho=0). The caller is
    responsible for additionally stripping draw cells from the EV grid since
    knockouts always end with a winner.
    """
    gm = _goals_model()
    lam_h, lam_a = gm.predict_lambdas(feats_row)
    lam_h = float(lam_h[0]); lam_a = float(lam_a[0])

    grid = np.arange(MAX_GOALS + 1)
    pmf_h = poisson.pmf(grid, lam_h)
    pmf_a = poisson.pmf(grid, lam_a)
    pmf_h /= pmf_h.sum()
    pmf_a /= pmf_a.sum()
    joint = np.outer(pmf_h, pmf_a)  # independent baseline

    # Apply Dixon-Coles tau correction (only for non-KO; KO has a winner).
    rho = 0.0 if is_knockout else (dc_rho if dc_rho is not None else DC_RHO)
    if rho != 0.0:
        for h in (0, 1):
            for a in (0, 1):
                joint[h, a] *= _dixon_coles_tau(h, a, lam_h, lam_a, rho)
        joint = np.maximum(joint, 1e-9)
        joint /= joint.sum()

    # Soft empirical recalibration of classifier marginals using tournament-so-far
    # draw rate. Helps when the model's class probs are off.
    if not is_knockout and CAL_BLEND > 0.0:
        priors = _tournament_class_priors()
        if priors is not None:
            ph_emp, pd_emp, pa_emp = priors
            w = CAL_BLEND
            p_home = (1 - w) * p_home + w * ph_emp
            p_draw = (1 - w) * p_draw + w * pd_emp
            p_away = (1 - w) * p_away + w * pa_emp
            s = p_home + p_draw + p_away
            p_home /= s; p_draw /= s; p_away /= s

    # Additional empirical draw boost (off in KO since draws aren't a valid result).
    if not is_knockout and DRAW_BOOST > 1.0 and (p_home + p_away) > 0:
        boosted = min(0.95, p_draw * DRAW_BOOST)
        leftover = 1.0 - boosted
        scale = leftover / (p_home + p_away)
        p_home *= scale; p_away *= scale; p_draw = boosted

    # Reweight to match (recalibrated) class probabilities.
    idx_h, idx_a = np.indices(joint.shape)
    mask_home = idx_h > idx_a
    mask_draw = idx_h == idx_a
    mask_away = idx_h < idx_a

    s_home = joint[mask_home].sum()
    s_draw = joint[mask_draw].sum()
    s_away = joint[mask_away].sum()
    if s_home > 0: joint[mask_home] *= p_home / s_home
    if s_draw > 0: joint[mask_draw] *= p_draw / s_draw
    if s_away > 0: joint[mask_away] *= p_away / s_away

    joint /= joint.sum()
    return joint, lam_h, lam_a


@lru_cache(maxsize=1)
def _points_grid_no_draw() -> np.ndarray:
    """Points grid for knockout matches — zeros out all draw-tip rows since
    a draw can't be the official KO result (extra time + penalties decide it)."""
    pts = _points_grid().copy()
    n = MAX_GOALS + 1
    for ht in range(n):
        pts[ht, ht, :, :] = 0
    return pts


def best_tip(joint: np.ndarray, top_k: int = 3,
             *, is_knockout: bool = False) -> list[dict]:
    """Return the top-k tips ranked by expected points. In knockouts, draw
    tips are excluded since the official result must have a winner."""
    grid = _points_grid_no_draw() if is_knockout else _points_grid()
    pts = grid.astype(np.float32)
    ev = np.einsum("hxij,ij->hx", pts, joint.astype(np.float32))
    flat = ev.flatten()
    order = np.argsort(flat)[::-1]
    n = MAX_GOALS + 1
    out = []
    for idx in order[:top_k]:
        ht, at = divmod(int(idx), n)
        out.append({"tip": (ht, at), "ev": float(flat[idx]),
                    "p_exact": float(joint[ht, at])})
    return out


def predict_tip(home: str, away: str, date, tournament_class: str,
                city: str | None, neutral: bool,
                match_num_in_tournament: int = 1,
                home_starters: list[str] | None = None,
                away_starters: list[str] | None = None,
                referee: str | None = None,
                is_knockout: bool = False) -> dict:
    """Full pipeline: features -> classifier -> goal rates -> tip EV.
    If confirmed XI is provided, squad features come from those 11 starters.
    is_knockout=True disables Dixon-Coles draw boost and excludes draw tips."""
    fx = feature_engine.FixtureInput(
        home=home, away=away, date=date,
        tournament_class=tournament_class, city=city, neutral=neutral,
        match_num_in_tournament=match_num_in_tournament,
        home_starters=home_starters, away_starters=away_starters,
        referee=referee,
    )
    feats_all = feature_engine.compute_features(fx)
    feats = feats_all.drop(columns=[c for c in feats_all.columns if c.startswith("__")])

    proba = _prematch_model().predict_proba(feats)[0]
    p_home, p_draw, p_away = float(proba[0]), float(proba[1]), float(proba[2])
    # Apply post-hoc referee adjustment (capped ±5pp on home/away) if provided.
    p_home, p_draw, p_away, ref_info = _refs.apply_ref_adjustment(
        p_home, p_draw, p_away, referee,
    )
    # Apply post-hoc manager tactical-matchup adjustment (capped ±3pp). This is
    # the explicit "manager A vs manager B" tactical signal — extra/independent
    # of any squad-quality features the model has already learned.
    p_home, p_draw, p_away, mgr_info = _managers_mod.apply_manager_adjustment(
        p_home, p_draw, p_away, home, away,
    )
    # Blend in bookmaker (Oddset/Bet365) closing-line implied probabilities
    # where available. The market generally outperforms pure ML on calibration.
    p_home, p_draw, p_away, odds_blended = _blend_with_odds(
        home, away, date, p_home, p_draw, p_away, ODDS_BLEND,
    )

    joint, lam_h, lam_a = joint_scoreline_distribution(
        feats, p_home, p_draw, p_away, is_knockout=is_knockout,
    )
    tips = best_tip(joint, top_k=3, is_knockout=is_knockout)

    return {
        "home": home, "away": away,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "lambda_home": lam_h, "lambda_away": lam_a,
        "best_tip": f"{tips[0]['tip'][0]}-{tips[0]['tip'][1]}",
        "best_tip_ev": tips[0]["ev"],
        "best_tip_p_exact": tips[0]["p_exact"],
        "alt_tips": [
            {"tip": f"{t['tip'][0]}-{t['tip'][1]}",
             "ev": round(t["ev"], 3),
             "p_exact": round(t["p_exact"], 4)}
            for t in tips[1:]
        ],
        "home_elo": float(feats_all["__home_elo"].iloc[0]),
        "away_elo": float(feats_all["__away_elo"].iloc[0]),
        "referee": referee,
        "ref_bias_applied": float(ref_info.get("home_win_bias") or 0.0) if ref_info else 0.0,
        "manager_info": mgr_info,
    }
