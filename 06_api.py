"""
Step 6 - FastAPI prediction server.

Endpoints:
  POST /predict/prematch   -> 3-class probabilities for a single fixture
  GET  /predict/wc2026     -> predictions for every remaining WC2026 group match
  GET  /elo                -> top-N current Elo standings
  GET  /tournament         -> Monte Carlo bracket probabilities (read from disk)
  POST /reload-models      -> hot reload model + cached state after retrain
"""
from __future__ import annotations

import importlib
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make sibling modules importable when running via `uvicorn 06_api:app`.
sys.path.insert(0, str(Path(__file__).parent))

# 04_train_prematch.PrematchModel is needed for pickle.load to resolve the class.
sys.path.insert(0, str(Path(__file__).parent))
_train_module = importlib.import_module("04_train_prematch")
PrematchModel = _train_module.PrematchModel

import feature_engine
import tip_optimizer
import wc2026_schedule

_cascade_module = importlib.import_module("14_cascade_forecast")
_refs_module = importlib.import_module("12_referees")
import managers as _managers_module
import bonus_predictor as _bonus_module

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

CLASSES = ["home_win", "draw", "away_win"]


# ---- state holder so /reload-models can swap atomically ----
class State:
    def __init__(self) -> None:
        self.model: PrematchModel | None = None
        self.elo: pd.DataFrame | None = None
        self.tournament: pd.DataFrame | None = None

    def load(self) -> None:
        with open(MODEL_DIR / "prematch_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.elo = pd.read_csv(DATA_DIR / "team_elo_current.csv")
        tourney_path = DATA_DIR / "tournament_sim_results.csv"
        self.tournament = pd.read_csv(tourney_path) if tourney_path.exists() else None
        feature_engine.invalidate_cache()
        tip_optimizer.invalidate_cache()
        _refs_module.invalidate_cache()
        _managers_module.invalidate_cache()


state = State()
state.load()
app = FastAPI(title="FIFA WC 2026 Predictor", version="1.0")


# ---- schemas ----
class PrematchRequest(BaseModel):
    home: str
    away: str
    date: str = Field(..., description="YYYY-MM-DD")
    tournament_class: Literal[
        "wc_finals", "continental_finals", "qualifier",
        "minor_tourney", "friendly",
    ] = "wc_finals"
    city: str | None = None
    neutral: bool = True
    match_num_in_tournament: int = 1
    is_knockout: bool = False
    home_starters: list[str] | None = None
    away_starters: list[str] | None = None
    referee: str | None = None


class PrematchResponse(BaseModel):
    home: str
    away: str
    date: str
    p_home: float
    p_draw: float
    p_away: float
    predicted: str
    confidence: float
    signals_agree: int
    signals: list[str]
    home_elo: float
    away_elo: float


def _confidence(p: list[float], n_signals: int) -> float:
    """Combine max-class probability with how many independent signals agree."""
    base = max(p)
    boost = min(1.5, 1.0 + 0.06 * n_signals)
    return min(0.99, base * boost)


def _predict_one(req: PrematchRequest) -> PrematchResponse:
    if state.model is None:
        raise HTTPException(503, "model not loaded")
    fx = feature_engine.FixtureInput(
        home=req.home,
        away=req.away,
        date=datetime.strptime(req.date, "%Y-%m-%d"),
        tournament_class=req.tournament_class,
        city=req.city,
        neutral=req.neutral,
        match_num_in_tournament=req.match_num_in_tournament,
        home_starters=req.home_starters,
        away_starters=req.away_starters,
        referee=req.referee,
    )
    feats = feature_engine.compute_features(fx)
    extras = {c: feats[c].iloc[0] for c in feats.columns if c.startswith("__")}
    model_input = feats.drop(columns=[c for c in feats.columns if c.startswith("__")])
    proba = state.model.predict_proba(model_input)[0]
    # Referee post-hoc adjustment (±5pp shift based on home-win bias).
    p_h, p_d, p_a, ref_info = _refs_module.apply_ref_adjustment(
        float(proba[0]), float(proba[1]), float(proba[2]), req.referee,
    )
    # Manager tactical-matchup adjustment (±3pp).
    p_h, p_d, p_a, mgr_info = _managers_module.apply_manager_adjustment(
        p_h, p_d, p_a, req.home, req.away,
    )
    proba = np.array([p_h, p_d, p_a])
    predicted = CLASSES[int(proba.argmax())]
    n_agree, signal_names = feature_engine.signal_agreement(feats.iloc[0], predicted)
    return PrematchResponse(
        home=req.home, away=req.away, date=req.date,
        p_home=float(proba[0]), p_draw=float(proba[1]), p_away=float(proba[2]),
        predicted=predicted,
        confidence=_confidence(list(proba), n_agree),
        signals_agree=n_agree, signals=signal_names,
        home_elo=float(extras["__home_elo"]),
        away_elo=float(extras["__away_elo"]),
    )


# ---- endpoints ----
@app.post("/predict/prematch", response_model=PrematchResponse)
def predict_prematch(req: PrematchRequest):
    return _predict_one(req)


class TipResponse(BaseModel):
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    lambda_home: float
    lambda_away: float
    best_tip: str
    best_tip_ev: float
    best_tip_p_exact: float
    alt_tips: list[dict]
    home_elo: float
    away_elo: float


@app.post("/predict/tip", response_model=TipResponse)
def predict_tip(req: PrematchRequest):
    """Returns the expected-points-maximising scoreline for the WC Tip Game 2026 rubric (4/3/2/0)."""
    return tip_optimizer.predict_tip(
        home=req.home, away=req.away,
        date=datetime.strptime(req.date, "%Y-%m-%d"),
        tournament_class=req.tournament_class,
        city=req.city, neutral=req.neutral,
        match_num_in_tournament=req.match_num_in_tournament,
        is_knockout=req.is_knockout,
        home_starters=req.home_starters,
        away_starters=req.away_starters,
        referee=req.referee,
    )


# ---- lineup management ----
_lineup_module = importlib.import_module("13_lineup_fetcher")


class LineupPayload(BaseModel):
    home: str
    away: str
    date: str
    home_starters: list[str]
    away_starters: list[str]
    referee: str | None = None


@app.post("/lineup/manual")
def set_manual_lineup(p: LineupPayload):
    _lineup_module.set_manual_lineup(
        home=p.home, away=p.away, date=p.date,
        home_starters=p.home_starters, away_starters=p.away_starters,
        referee=p.referee,
    )
    return {"status": "ok", "key": _lineup_module._match_key(p.home, p.away, p.date)}


@app.get("/lineup")
async def get_lineup(home: str, away: str, date: str):
    res = await _lineup_module.fetch_lineup(home, away, date)
    if res is None:
        return {"found": False, "source": "none"}
    return {
        "found": True, "source": res.source, "confirmed": res.confirmed,
        "home_starters": res.home_starters, "away_starters": res.away_starters,
        "home_subs": res.home_subs, "away_subs": res.away_subs,
        "referee": res.referee,
    }


@app.get("/predict/tips/wc2026")
def predict_tips_wc2026():
    """Tip predictions for every group fixture in EV order — useful pre-tournament digest."""
    fxts = wc2026_schedule.all_group_fixtures()
    rows = []
    for f in fxts:
        try:
            r = tip_optimizer.predict_tip(
                home=f.home, away=f.away, date=f.date,
                tournament_class="wc_finals", city=f.city,
                neutral=f.neutral, match_num_in_tournament=f.match_no,
            )
            rows.append({
                "match_no": f.match_no, "date": f.date.strftime("%Y-%m-%d"),
                "group": f.group, "home": f.home, "away": f.away, "city": f.city,
                "tip": r["best_tip"], "ev": round(r["best_tip_ev"], 3),
                "lambda_home": round(r["lambda_home"], 2),
                "lambda_away": round(r["lambda_away"], 2),
                "p_home": round(r["p_home"], 3), "p_draw": round(r["p_draw"], 3),
                "p_away": round(r["p_away"], 3),
            })
        except Exception as e:
            rows.append({"match_no": f.match_no, "error": str(e)})
    return {"fixtures": rows}


@app.get("/predict/wc2026")
def predict_wc2026():
    fxts = wc2026_schedule.all_group_fixtures()
    rows = []
    for f in fxts:
        req = PrematchRequest(
            home=f.home, away=f.away, date=f.date.strftime("%Y-%m-%d"),
            tournament_class="wc_finals", city=f.city,
            neutral=f.neutral, match_num_in_tournament=f.match_no,
        )
        try:
            r = _predict_one(req)
            rows.append({
                "match_no": f.match_no, "date": req.date, "group": f.group,
                "home": f.home, "away": f.away, "city": f.city,
                "p_home": round(r.p_home, 4), "p_draw": round(r.p_draw, 4),
                "p_away": round(r.p_away, 4),
                "predicted": r.predicted, "confidence": round(r.confidence, 4),
                "home_elo": round(r.home_elo, 1), "away_elo": round(r.away_elo, 1),
            })
        except Exception as e:
            rows.append({"match_no": f.match_no, "error": str(e),
                         "home": f.home, "away": f.away})
    return {"fixtures": rows}


@app.get("/elo")
def elo(top: int = 30):
    if state.elo is None:
        raise HTTPException(503, "elo not loaded")
    return {"top": state.elo.head(top).to_dict(orient="records")}


@app.get("/tournament")
def tournament():
    if state.tournament is None:
        return {"note": "Monte Carlo simulation has not been run yet.",
                "hint": "POST /reload-models after running 05_simulate_tournament.py."}
    return {"standings": state.tournament.to_dict(orient="records")}


@app.get("/forecast")
def forecast(persist: bool = True):
    """Run the full cascading deterministic forecast and return all 104 predicted matches."""
    outcomes = _cascade_module.cascade()
    df = _cascade_module.to_dataframe(outcomes)
    if persist:
        df.to_csv(DATA_DIR / "cascade_forecast.csv", index=False)
    rows = []
    for o in outcomes:
        rows.append({
            "match_no": o.match_no, "date": o.date.strftime("%Y-%m-%d"),
            "stage": o.stage, "group": o.group,
            "home": o.home, "away": o.away,
            "predicted_score": o.predicted_score,
            "best_tip": o.best_tip,
            "best_tip_ev": round(o.best_tip_ev, 3) if not (o.best_tip_ev != o.best_tip_ev) else None,
            "winner": o.winner, "from_actual": o.from_actual,
        })
    final = next((o for o in outcomes if o.stage == "Final"), None)
    third = next((o for o in outcomes if o.stage == "3rd"), None)
    summary = {
        "champion": final.winner if final else None,
        "runner_up": (final.away if final and final.winner == final.home else (final.home if final else None)),
        "third_place": third.winner if third else None,
        "fourth_place": (third.away if third and third.winner == third.home else (third.home if third else None)),
        "final": f"{final.home} {final.predicted_score} {final.away}" if final else None,
    }
    return {"summary": summary, "matches": rows}


@app.get("/bonus")
def bonus():
    """All 18 picks for the WC Tip Game 2026 Bonus tab + confidence breakdown."""
    return _bonus_module.bonus_picks()


@app.get("/bonus/submit")
def bonus_submit():
    """Plain-text copy-paste output, one line per dropdown."""
    return {"text": _bonus_module.format_submit(_bonus_module.bonus_picks())}


@app.post("/reload-models")
def reload_models():
    state.load()
    return {"status": "ok", "loaded_at": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
