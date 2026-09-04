"""
Step 4 - Train a 3-class (home_win / draw / away_win) ensemble pre-match model.

Architecture mirrors the IPL pre-match ensemble:
  XGBoost (multi:softprob) + LightGBM (multiclass) + Multinomial LogReg
  Soft-voting (mean of class probabilities)
  Isotonic per-class calibration via out-of-fold cross-validated predictions
  Recency weighting (2024+ -> 3x, 2010-2023 -> 1x, older -> 0.5x)
  Tournament weighting (WC + continental finals -> 2x)

Optuna tuning is skipped here (default hyperparams) to ship the MVP before
the WC2026 opener on 2026-06-11. A `--tune` flag can be added later.

Held-out test: 2025+ matches (qualifiers + Euro/Copa 2024 fall in 2024, so
include 2024 finals tournaments in the test set explicitly).
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

CLASSES = ["home_win", "draw", "away_win"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

FEATURES = [
    # Elo
    "elo_diff", "elo_diff_with_ha",
    # Form diffs
    "ppg_diff_last5", "ppg_diff_last10", "ppg_diff_last20",
    "gf_diff_last5", "gf_diff_last10", "gf_diff_last20",
    "ga_diff_last5", "ga_diff_last10", "ga_diff_last20",
    "form_weighted_diff", "days_rest_diff",
    # H2H
    "h2h_n", "h2h_home_win_rate", "h2h_draw_rate", "h2h_avg_gd",
    # Tournament class
    "is_wc_finals", "is_continental_finals", "is_qualifier",
    "is_minor_tourney", "is_friendly",
    # Venue
    "is_neutral", "home_in_own_country", "venue_altitude_m", "venue_high_altitude",
    # Confederation
    "same_confederation",
    "home_conf_UEFA", "home_conf_CONMEBOL", "home_conf_CONCACAF",
    "home_conf_AFC", "home_conf_CAF", "home_conf_OFC", "home_conf_OTHER",
    "away_conf_UEFA", "away_conf_CONMEBOL", "away_conf_CONCACAF",
    "away_conf_AFC", "away_conf_CAF", "away_conf_OFC", "away_conf_OTHER",
    # Match index
    "match_num_in_tournament", "early_tournament",
    # Squad profile diffs (2026 snapshot — constant per team across history,
    # acts as a team-level quality prior).
    "squad_diff_caps_top22", "squad_diff_caps_top11",
    "squad_diff_goal_rate_top4_fw", "squad_diff_fw_goals_top4_total",
    "squad_diff_age_top22_avg", "squad_diff_age_top22_std",
    "squad_diff_top5_league_pct", "squad_diff_big_club_pct",
    "squad_diff_chemistry_max_cluster", "squad_diff_chemistry_top3_sum",
    "squad_diff_unique_clubs", "squad_diff_gk_avg_caps",
    # Manager diffs + tactical style one-hot. Same fixed-snapshot caveat as
    # squad features — these act as manager-quality / tactical priors.
    "mgr_diff_tenure_months", "mgr_diff_matches_in_charge",
    "mgr_diff_career_win_pct", "mgr_diff_big_match_experience",
    "mgr_diff_age", "mgr_diff_attack_axis", "mgr_diff_possession_axis",
    "home_mgr_is_high_press", "home_mgr_is_possession", "home_mgr_is_counter",
    "home_mgr_is_low_block", "home_mgr_is_balanced",
    "away_mgr_is_high_press", "away_mgr_is_possession", "away_mgr_is_counter",
    "away_mgr_is_low_block", "away_mgr_is_balanced",
]


@dataclass
class PrematchModel:
    """Picklable ensemble: XGB + LGB + LogReg, soft-voted, isotonic-calibrated."""
    xgb_model: xgb.XGBClassifier
    lgb_model: lgb.LGBMClassifier
    lr_model: LogisticRegression
    scaler: StandardScaler
    medians: dict
    feature_names: list[str]
    isotonics: list[IsotonicRegression]  # one per class
    classes: list[str]

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.feature_names].fillna(self.medians).values
        Xs = self.scaler.transform(Xf)
        p_xgb = self.xgb_model.predict_proba(Xf)
        p_lgb = self.lgb_model.predict_proba(Xf)
        p_lr = self.lr_model.predict_proba(Xs)
        return (p_xgb + p_lgb + p_lr) / 3.0

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self._raw_proba(X)
        out = np.empty_like(p)
        for i, iso in enumerate(self.isotonics):
            out[:, i] = iso.transform(p[:, i])
        out = out / out.sum(axis=1, keepdims=True)
        return out


def sample_weights(df: pd.DataFrame) -> np.ndarray:
    years = pd.to_datetime(df["date"]).dt.year.values
    w = np.where(years >= 2024, 3.0, np.where(years >= 2010, 1.0, 0.5))
    boost = np.where(
        df["tournament_class"].isin(["wc_finals", "continental_finals"]).values, 2.0, 1.0
    )
    return w * boost


def main() -> int:
    feats = pd.read_csv(DATA_DIR / "match_features.csv", parse_dates=["date"])
    feats = feats.dropna(subset=["target"]).copy()
    feats["target_idx"] = feats["target"].map(CLASS_TO_IDX)

    # ---- split: train <= 2023, val = 2024 finals + qualifiers, test = 2025+ ----
    train_mask = feats["date"] < "2024-01-01"
    test_mask = feats["date"] >= "2024-06-01"  # Euro 2024 finals + Copa 2024 + 2025+ + early 2026
    val_mask = (~train_mask) & (~test_mask)

    train_df, val_df, test_df = feats[train_mask], feats[val_mask], feats[test_mask]
    print(f"[04_train] sizes — train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,}")

    X_train = train_df[FEATURES]
    y_train = train_df["target_idx"].values
    X_val = val_df[FEATURES]
    y_val = val_df["target_idx"].values
    X_test = test_df[FEATURES]
    y_test = test_df["target_idx"].values

    medians = X_train.median(numeric_only=True).to_dict()
    X_train_f = X_train.fillna(medians)
    X_val_f = X_val.fillna(medians)
    X_test_f = X_test.fillna(medians)

    sw_train = sample_weights(train_df)

    # ---- base models (sensible defaults, no Optuna) ----
    print("[04_train] training XGBoost ...")
    xgbm = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgbm.fit(X_train_f, y_train, sample_weight=sw_train)

    print("[04_train] training LightGBM ...")
    lgbm = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=600, max_depth=-1, num_leaves=63,
        learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_train_f, y_train, sample_weight=sw_train)

    print("[04_train] training LogReg (scaled) ...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    lr = LogisticRegression(
        max_iter=2000, C=0.5,
        solver="lbfgs",
        random_state=42, n_jobs=-1,
    )
    lr.fit(X_train_s, y_train, sample_weight=sw_train)

    # ---- raw soft-voting probabilities ----
    def raw_proba(Xf):
        Xs = scaler.transform(Xf)
        return (xgbm.predict_proba(Xf) + lgbm.predict_proba(Xf) + lr.predict_proba(Xs)) / 3.0

    p_val_raw = raw_proba(X_val_f)
    print(f"[04_train] val (raw) — log_loss={log_loss(y_val, p_val_raw):.4f} "
          f"acc={accuracy_score(y_val, p_val_raw.argmax(axis=1)):.4f}")

    # ---- isotonic calibration via OOF on the train set ----
    print("[04_train] OOF calibration ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X_train_f), 3))
    Xf_arr = X_train_f.values
    for fold, (tr, te) in enumerate(skf.split(Xf_arr, y_train), 1):
        xgb_f = xgb.XGBClassifier(**xgbm.get_params()); xgb_f.fit(Xf_arr[tr], y_train[tr], sample_weight=sw_train[tr])
        lgb_f = lgb.LGBMClassifier(**lgbm.get_params()); lgb_f.fit(Xf_arr[tr], y_train[tr], sample_weight=sw_train[tr])
        sc_f = StandardScaler().fit(Xf_arr[tr])
        lr_f = LogisticRegression(**lr.get_params()); lr_f.fit(sc_f.transform(Xf_arr[tr]), y_train[tr], sample_weight=sw_train[tr])
        oof[te] = (xgb_f.predict_proba(Xf_arr[te]) + lgb_f.predict_proba(Xf_arr[te]) + lr_f.predict_proba(sc_f.transform(Xf_arr[te]))) / 3.0
        print(f"  fold {fold}/5 done")

    isotonics: list[IsotonicRegression] = []
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof[:, k], (y_train == k).astype(float))
        isotonics.append(iso)

    model = PrematchModel(
        xgb_model=xgbm, lgb_model=lgbm, lr_model=lr,
        scaler=scaler, medians=medians, feature_names=FEATURES,
        isotonics=isotonics, classes=CLASSES,
    )

    # ---- evaluate ----
    p_val = model.predict_proba(val_df)
    p_test = model.predict_proba(test_df)
    print(f"[04_train] val  (calibrated) — log_loss={log_loss(y_val, p_val):.4f} "
          f"acc={accuracy_score(y_val, p_val.argmax(axis=1)):.4f}")
    print(f"[04_train] test (calibrated) — log_loss={log_loss(y_test, p_test):.4f} "
          f"acc={accuracy_score(y_test, p_test.argmax(axis=1)):.4f}")
    print(f"  test class distribution: "
          f"home_win={(y_test==0).mean():.3f} draw={(y_test==1).mean():.3f} away_win={(y_test==2).mean():.3f}")

    out_path = MODEL_DIR / "prematch_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[04_train] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
