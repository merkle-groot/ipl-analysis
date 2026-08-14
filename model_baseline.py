"""Logistic-regression baseline for ball-level wicket prediction.

Splines on the features whose effect on wicket probability is curved rather
than linear - `bat_runs` above all, where the rate is flat from 0-40 then
steps up sharply past 50, something a plain linear term cannot express.

    python3 model_baseline.py

Split is temporal: train <=2023, validate 2024, test 2025-26. Never random -
two balls from the same over are near-duplicates and a random split lets the
model memorise the match.
"""

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

TARGET = "wicket"

# Curved effects -> spline basis. bat_runs/over/bat_balls are the ones where
# the shape is clearly non-monotone or kinked; the chase terms are ratios that
# blow up late in an innings.
SPLINE = ["bat_runs", "bat_balls", "over", "crr", "rrr", "balls_since_boundary",
          "bat_balls_since_boundary", "team_wickets", "bowl_balls"]

# Roughly linear / already-rate-shaped -> plain standardised terms.
LINEAR = ["team_runs", "legal_balls", "wickets_in_hand", "balls_since_wicket",
          "bat_dot_pct", "bat_sr", "bat_is_new", "bowl_runs", "bowl_wickets",
          "runs_required", "balls_left", "rrr_minus_crr", "is_powerplay",
          "is_chase", "innings", "bat_career_balls", "bat_career_dismissal_rate",
          "bat_career_sr", "bowl_career_balls", "bowl_career_wicket_rate",
          "bowl_career_runs_per_ball", "venue_wicket_rate", "venue_runs_per_ball",
          "h2h_dismissals", "h2h_balls", "toss_won"]

CATEGORICAL = ["phase"]


def split(df: pl.DataFrame):
    return (df.filter(pl.col("year") <= 2023),
            df.filter(pl.col("year") == 2024),
            df.filter(pl.col("year") >= 2025))


def report(name, y, p, base_rate):
    # Compare against always predicting the base rate: any model that cannot
    # beat that constant has learned nothing, whatever its accuracy looks like.
    null_ll = log_loss(y, np.full_like(p, base_rate))
    print(f"  {name:12s} logloss {log_loss(y, p):.5f} (null {null_ll:.5f}, "
          f"gain {100*(null_ll-log_loss(y,p))/null_ll:+.2f}%)  "
          f"ROC-AUC {roc_auc_score(y, p):.4f}  PR-AUC {average_precision_score(y, p):.4f}  "
          f"Brier {brier_score_loss(y, p):.5f}")


def main() -> None:
    df = pl.read_parquet("ball_data.parquet")
    train, val, test = split(df)
    print(f"train {train.height:,} ({train['year'].min()}-{train['year'].max()})  "
          f"val {val.height:,}  test {test.height:,}")

    feats = SPLINE + LINEAR + CATEGORICAL
    base_rate = train[TARGET].mean()
    print(f"train wicket rate: {base_rate:.4f}\n")

    pre = ColumnTransformer([
        ("spline", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
            ("scale", StandardScaler()),
        ]), SPLINE),
        ("linear", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), LINEAR),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
    ])

    model = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])

    X = lambda d: d.select(feats).to_pandas()
    y = lambda d: d[TARGET].to_numpy()

    model.fit(X(train), y(train))
    print(f"features after expansion: {model.named_steps['pre'].transform(X(val)).shape[1]}\n")

    for name, d in [("train", train), ("val", val), ("test", test)]:
        report(name, y(d), model.predict_proba(X(d))[:, 1], base_rate)

    # Calibration: are predicted 8% balls actually 8%? Matters more than AUC
    # for a probability model.
    p = model.predict_proba(X(test))[:, 1]
    cal = (pl.DataFrame({"p": p, "y": y(test)})
             .with_columns(pl.col("p").qcut(10, labels=[str(i) for i in range(10)]).alias("bin"))
             .group_by("bin").agg(pl.len().alias("n"), pl.col("p").mean().round(4).alias("predicted"),
                                  pl.col("y").mean().round(4).alias("actual")).sort("bin"))
    print("\ntest calibration by decile:")
    print(cal)

    # Which terms carry weight (splines summed across their basis functions).
    names = model.named_steps["pre"].get_feature_names_out()
    coefs = np.abs(model.named_steps["lr"].coef_[0])
    agg = {}
    for n, c in zip(names, coefs):
        key = n.split("__")[1].rsplit("_sp_", 1)[0]
        agg[key] = agg.get(key, 0.0) + c
    print("\ntop features by summed |coef|:")
    for k, v in sorted(agg.items(), key=lambda t: -t[1])[:15]:
        print(f"  {k:34s} {v:.3f}")


if __name__ == "__main__":
    main()
