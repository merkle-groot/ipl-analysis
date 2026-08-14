"""Gradient-boosted baselines (LightGBM, XGBoost) for ball-level wicket prediction.

Same temporal split and same metrics as model_baseline.py, so the numbers are
directly comparable to the logistic regression.

    python3 model_trees.py

Deliberate choices:

* No class rebalancing. `scale_pos_weight` / SMOTE would wreck calibration, and
  at 14.6k positives there is plenty of signal without it. We want honest
  probabilities, not a balanced-looking confusion matrix.
* Shallow trees and heavy regularisation. The signal here is weak (a ~2% log
  loss gain over the base rate is the realistic ceiling), so an unconstrained
  booster spends its capacity memorising noise.
* Early stopping on the 2024 validation season, never on test.
* Raw features, no splines - trees find thresholds like `bat_runs >= 50`
  themselves, which is the whole reason the LR needed splines and this does not.
"""

import numpy as np
import polars as pl
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

from model_baseline import CATEGORICAL, LINEAR, SPLINE, TARGET

# Trees take every feature raw. `ground` joins as a native categorical - it was
# left out of the LR because 37 one-hot columns of a near-zero-signal feature is
# a bad trade there, but a tree can ignore it for free.
NUMERIC = SPLINE + LINEAR
CATS = CATEGORICAL + ["ground"]
FEATS = NUMERIC + CATS


def frames():
    df = pl.read_parquet("ball_data.parquet")
    # One vocabulary per categorical, shared by all three splits. Grounds that
    # debut after 2023 (Mullanpur, 2024) are absent from the training frame, and
    # XGBoost errors on a category it has never seen rather than treating it as
    # unknown. This shares only the *level names*, never any target information.
    levels = {c: sorted(df[c].unique().drop_nulls().to_list()) for c in CATS}
    return levels, (df.filter(pl.col("year") <= 2023),
                    df.filter(pl.col("year") == 2024),
                    df.filter(pl.col("year") >= 2025))


def to_pandas(d, levels):
    import pandas as pd

    X = d.select(FEATS).to_pandas()
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=levels[c])
    return X


def report(name, y, p, base_rate):
    null_ll = log_loss(y, np.full_like(p, base_rate))
    ll = log_loss(y, p)
    print(f"  {name:6s} logloss {ll:.5f}  vs null {null_ll:.5f} "
          f"({100 * (null_ll - ll) / null_ll:+.2f}%)   "
          f"ROC-AUC {roc_auc_score(y, p):.4f}   "
          f"PR-AUC {average_precision_score(y, p):.4f}   "
          f"Brier {brier_score_loss(y, p):.5f}")


def main() -> None:
    import lightgbm as lgb
    import xgboost as xgb

    levels, (train, val, test) = frames()
    Xtr, Xva, Xte = (to_pandas(d, levels) for d in (train, val, test))
    ytr, yva, yte = (d[TARGET].to_numpy() for d in (train, val, test))
    base = ytr.mean()
    print(f"train {len(ytr):,} · val {len(yva):,} · test {len(yte):,} · "
          f"base rate {base:.4f}\n")

    # ---------------- LightGBM ----------------
    lgbm = lgb.LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=0.02,
        num_leaves=15, min_child_samples=400, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, random_state=0,
    )
    lgbm.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="binary_logloss",
             callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"LightGBM  (best iter {lgbm.best_iteration_})")
    for n, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        report(n, y, lgbm.predict_proba(X)[:, 1], base)

    # ---------------- XGBoost ----------------
    xgbm = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=3000, learning_rate=0.02,
        max_depth=4, min_child_weight=50, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=10.0, enable_categorical=True, tree_method="hist",
        eval_metric="logloss", early_stopping_rounds=100, random_state=0,
    )
    xgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    print(f"\nXGBoost  (best iter {xgbm.best_iteration})")
    for n, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        report(n, y, xgbm.predict_proba(X)[:, 1], base)

    # ---------------- comparison ----------------
    p_l = lgbm.predict_proba(Xte)[:, 1]
    p_x = xgbm.predict_proba(Xte)[:, 1]
    print("\ntest-set summary")
    print(f"  {'model':10s} {'logloss':>9s} {'gain':>8s} {'ROC-AUC':>9s} {'PR-AUC':>8s}")
    null_ll = log_loss(yte, np.full_like(p_l, base))
    for name, p in [("null", np.full_like(p_l, base)), ("lightgbm", p_l),
                    ("xgboost", p_x), ("lgb+xgb", (p_l + p_x) / 2)]:
        ll = log_loss(yte, p)
        gain = 100 * (null_ll - ll) / null_ll
        auc = "n/a" if name == "null" else f"{roc_auc_score(yte, p):.4f}"
        ap = "n/a" if name == "null" else f"{average_precision_score(yte, p):.4f}"
        print(f"  {name:10s} {ll:9.5f} {gain:7.2f}% {auc:>9s} {ap:>8s}")

    imp = sorted(zip(FEATS, lgbm.booster_.feature_importance("gain")),
                 key=lambda t: -t[1])
    total = sum(v for _, v in imp)
    print("\nLightGBM top features (% of total gain):")
    for k, v in imp[:15]:
        print(f"  {k:32s} {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()
