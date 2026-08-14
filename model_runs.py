"""Ball-level run prediction: the distribution of runs off the bat.

Companion to model_baseline.py / model_trees.py, which predict wickets. Same
parquet, same temporal split, same "every feature is pre-ball state" rule.

    python3 model_runs.py

The target is a *shape* rather than a number. Runs off the bat take six values
in practice - 0, 1, 2, 3, 4, 6 - and the gaps matter: a ball is far more likely
to go for 4 than for 3, so anything that treats the outcome as a continuous
quantity between 0 and 6 is modelling a space where most of the mass cannot
exist. We fit a 6-way multinomial and read expected runs off the predicted
distribution when a point estimate is wanted.

Scope: legal deliveries only. Wides and no-balls are a bowler-error process
with a different outcome space (the extra is charged before the bat is
involved), and they are 3.8% of rows.
"""

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

from model_baseline import CATEGORICAL, LINEAR, SPLINE

TARGET = "runs_batter"

# The six outcomes that actually occur. 5 off the bat happens 74 times in
# 18 seasons (overthrows) and is folded into 4; 3 is kept - rare, but a real
# running outcome rather than a scoring quirk.
OUTCOMES = [0, 1, 2, 3, 4, 6]

# Trees take every feature raw, including `ground` as a native categorical.
NUMERIC = SPLINE + LINEAR
CATS = CATEGORICAL + ["ground"]
FEATS = NUMERIC + CATS


def label(d: pl.DataFrame) -> pl.DataFrame:
    """Legal deliveries only, with runs off the bat mapped to a class index."""
    mapping = pl.when(pl.col(TARGET) == 5).then(4).otherwise(pl.col(TARGET))
    return (d.filter(pl.col("is_legal") == 1)
             .with_columns(mapping.alias("_runs"))
             .with_columns(pl.col("_runs").replace_strict(
                 {v: i for i, v in enumerate(OUTCOMES)}).alias("y")))


def frames():
    df = label(pl.read_parquet("ball_data.parquet"))
    # One vocabulary per categorical shared across splits: grounds that debut
    # after 2023 are absent from the training frame and XGBoost errors on a
    # level it has never seen. Level names only, never target information.
    levels = {c: sorted(df[c].unique().drop_nulls().to_list()) for c in CATS}
    return levels, (df.filter(pl.col("year") <= 2023),
                    df.filter(pl.col("year") == 2024),
                    df.filter(pl.col("year") >= 2025))


def to_pandas(d, levels, feats=None):
    import pandas as pd

    feats = feats or FEATS
    X = d.select(feats).to_pandas()
    for c in CATS:
        if c in feats:
            X[c] = pd.Categorical(X[c], categories=levels[c])
    return X


def expected_runs(proba: np.ndarray) -> np.ndarray:
    """Point estimate implied by a predicted distribution."""
    return proba @ np.array(OUTCOMES, dtype=float)


def report(name, y, proba, prior):
    """Multiclass log loss against the train prior, plus expected-runs error."""
    labels = list(range(len(OUTCOMES)))
    null = np.tile(prior, (len(y), 1))
    ll, null_ll = (log_loss(y, p, labels=labels) for p in (proba, null))
    runs = np.array(OUTCOMES, dtype=float)[y]
    e, e_null = expected_runs(proba), expected_runs(null)
    print(f"  {name:6s} logloss {ll:.5f}  vs null {null_ll:.5f} "
          f"({100 * (null_ll - ll) / null_ll:+.2f}%)   "
          f"E[runs] RMSE {np.sqrt(np.mean((e - runs) ** 2)):.4f} "
          f"vs null {np.sqrt(np.mean((e_null - runs) ** 2)):.4f}   "
          f"MAE {np.mean(np.abs(e - runs)):.4f}")


def main() -> None:
    import lightgbm as lgb

    levels, (train, val, test) = frames()
    Xtr, Xva, Xte = (to_pandas(d, levels) for d in (train, val, test))
    ytr, yva, yte = (d["y"].to_numpy() for d in (train, val, test))
    prior = np.bincount(ytr, minlength=len(OUTCOMES)) / len(ytr)
    print(f"train {len(ytr):,} · val {len(yva):,} · test {len(yte):,}")
    print("train outcome mix: " + "  ".join(
        f"{v}:{p:.4f}" for v, p in zip(OUTCOMES, prior)))
    print(f"mean runs per ball: {expected_runs(prior[None, :])[0]:.4f}\n")

    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(OUTCOMES), n_estimators=3000,
        learning_rate=0.03, num_leaves=15, min_child_samples=400, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1,
        random_state=0,
    )
    m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"LightGBM  (best iter {m.best_iteration_})")
    for n, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        report(n, y, m.predict_proba(X), prior)

    imp = sorted(zip(FEATS, m.booster_.feature_importance("gain")),
                 key=lambda t: -t[1])
    total = sum(v for _, v in imp)
    print("\nLightGBM top features (% of total gain):")
    for k, v in imp[:15]:
        print(f"  {k:32s} {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()
