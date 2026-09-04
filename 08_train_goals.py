"""
Step 9 - Train Poisson regressors for goals scored, separately for home and away.

These feed `tip_optimizer.py`, which builds a P(home_goals, away_goals) joint
distribution and picks the scoreline that maximises expected tip-game points.

Same feature set as the win/draw/loss model (FEATURES from 04_train_prematch).
We fit two PoissonRegressor models from sklearn — they cap log-link rates
cleanly and handle the sample-weight + recency-weighting workflow.

Saves models/goals_model.pkl containing {home_model, away_model, scaler, medians,
feature_names, dispersion}.
"""
from __future__ import annotations

import importlib
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
_train = importlib.import_module("04_train_prematch")
FEATURES = _train.FEATURES
sample_weights = _train.sample_weights

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class GoalsModel:
    home_model: PoissonRegressor
    away_model: PoissonRegressor
    scaler: StandardScaler
    medians: dict
    feature_names: list[str]

    def predict_lambdas(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xf = X[self.feature_names].fillna(self.medians).values
        Xs = self.scaler.transform(Xf)
        lam_h = self.home_model.predict(Xs)
        lam_a = self.away_model.predict(Xs)
        return lam_h, lam_a


def main() -> int:
    feats = pd.read_csv(DATA_DIR / "match_features.csv", parse_dates=["date"])
    feats = feats.dropna(subset=["target", "home_score", "away_score"]).copy()

    train_mask = feats["date"] < "2024-01-01"
    test_mask = feats["date"] >= "2024-06-01"
    train_df = feats[train_mask]
    test_df = feats[test_mask]

    X_train = train_df[FEATURES]
    medians = X_train.median(numeric_only=True).to_dict()
    X_train_f = X_train.fillna(medians)
    X_test_f = test_df[FEATURES].fillna(medians)

    scaler = StandardScaler().fit(X_train_f)
    X_train_s = scaler.transform(X_train_f)
    X_test_s = scaler.transform(X_test_f)

    sw = sample_weights(train_df)

    print(f"[08_train_goals] training Poisson(home_goals) on {len(train_df):,} rows ...")
    home_m = PoissonRegressor(alpha=0.5, max_iter=500)
    home_m.fit(X_train_s, train_df["home_score"].astype(int).values, sample_weight=sw)

    print(f"[08_train_goals] training Poisson(away_goals) ...")
    away_m = PoissonRegressor(alpha=0.5, max_iter=500)
    away_m.fit(X_train_s, train_df["away_score"].astype(int).values, sample_weight=sw)

    # ---- evaluate ----
    pred_h_test = home_m.predict(X_test_s)
    pred_a_test = away_m.predict(X_test_s)
    actual_h = test_df["home_score"].astype(int).values
    actual_a = test_df["away_score"].astype(int).values

    print(f"  test home_goals  mean(predicted)={pred_h_test.mean():.3f}  mean(actual)={actual_h.mean():.3f}")
    print(f"  test away_goals  mean(predicted)={pred_a_test.mean():.3f}  mean(actual)={actual_a.mean():.3f}")
    print(f"  test MAE home: {np.abs(pred_h_test - actual_h).mean():.3f}")
    print(f"  test MAE away: {np.abs(pred_a_test - actual_a).mean():.3f}")

    model = GoalsModel(
        home_model=home_m, away_model=away_m, scaler=scaler,
        medians=medians, feature_names=FEATURES,
    )
    out = MODEL_DIR / "goals_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[08_train_goals] saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
