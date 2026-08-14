"""Independently calibrated style/matchup model for runs off the bat.

The time windows have distinct jobs:

* 2008-2022 selects the model against 2023 (tree count only).
* The selected model is refit on 2008-2023.
* 2024 fits class-intercept calibration and nothing else.
* 2025-2026 is evaluated without further fitting.

Run directly for a concise report, or import ``fit_independent_style_model``
from ``runs.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

from evaluation import apply_logit_bias, clustered_log_loss_gain, fit_logit_bias
from model_runs import CATS, OUTCOMES, expected_runs, label


KEEP = ["is_powerplay", "team_runs", "team_wickets", "legal_balls", "bat_career_sr",
        "balls_since_wicket", "bat_runs", "bat_dot_pct", "bat_balls",
        "bat_career_dismissal_rate", "bowl_career_runs_per_ball", "rrr", "phase",
        "bat_sr", "bat_career_balls"]
MATCHUP = ["bowler_type", "bowler_arm", "bat_hand", "same_handed", "turn_into_batter"]
FEATURES = KEEP + MATCHUP
CATEGORICAL = CATS + ["bowler_type", "bowler_arm", "bat_hand"]
PARAMS = dict(
    objective="multiclass", num_class=len(OUTCOMES), n_estimators=4000,
    learning_rate=0.02, num_leaves=15, min_child_samples=1000, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.7, reg_lambda=30.0, verbose=-1,
)


@dataclass(frozen=True)
class CalibratedStyleResult:
    raw_calibration: np.ndarray
    raw_test: np.ndarray
    calibrated_test: np.ndarray
    bias: np.ndarray
    best_iterations: tuple[int, ...]
    y_calibration: np.ndarray
    y_test: np.ndarray
    test_match_ids: np.ndarray


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_independent_style_model(
    df: pl.DataFrame | None = None,
    *,
    n_seeds: int = 5,
    calibration_l2: float = 1.0,
) -> CalibratedStyleResult:
    """Fit the full matchup ensemble and an independent 2024 bias calibrator."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    df = label(pl.read_parquet("ball_data.parquet")) if df is None else df
    if "y" not in df.columns:
        df = label(df)

    selection_train = df.filter(pl.col("year") <= 2022)
    early_stop = df.filter(pl.col("year") == 2023)
    calibration = df.filter(pl.col("year") == 2024)
    test = df.filter(pl.col("year") >= 2025)
    final_train = pl.concat([selection_train, early_stop])

    levels = {c: sorted(df[c].unique().drop_nulls().to_list()) for c in CATEGORICAL}

    def frame(data: pl.DataFrame) -> pd.DataFrame:
        x = data.select(FEATURES).to_pandas()
        for column in CATEGORICAL:
            if column in FEATURES:
                x[column] = pd.Categorical(x[column], categories=levels[column])
        return x

    x_select, x_stop, x_final, x_cal, x_test = map(
        frame, (selection_train, early_stop, final_train, calibration, test)
    )
    y_select, y_stop, y_final, y_cal, y_test = (
        data["y"].to_numpy()
        for data in (selection_train, early_stop, final_train, calibration, test)
    )

    cal_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    best_iterations: list[int] = []
    for seed in range(n_seeds):
        selector = lgb.LGBMClassifier(**PARAMS, random_state=seed)
        selector.fit(
            x_select, y_select, eval_X=x_stop, eval_y=y_stop,
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        best_iterations.append(selector.best_iteration_)

        model = lgb.LGBMClassifier(
            **{**PARAMS, "n_estimators": selector.best_iteration_}, random_state=seed
        )
        model.fit(x_final, y_final)
        cal_predictions.append(_normalize(model.predict_proba(x_cal)))
        test_predictions.append(_normalize(model.predict_proba(x_test)))

    raw_calibration = _normalize(np.mean(cal_predictions, axis=0))
    raw_test = _normalize(np.mean(test_predictions, axis=0))
    bias = fit_logit_bias(raw_calibration, y_cal, l2=calibration_l2)
    calibrated_test = apply_logit_bias(raw_test, bias)
    return CalibratedStyleResult(
        raw_calibration=raw_calibration,
        raw_test=raw_test,
        calibrated_test=calibrated_test,
        bias=bias,
        best_iterations=tuple(best_iterations),
        y_calibration=y_cal,
        y_test=y_test,
        test_match_ids=test["match_id"].to_numpy(),
    )


def main() -> None:
    result = fit_independent_style_model()
    prior_df = label(pl.read_parquet("ball_data.parquet")).filter(pl.col("year") <= 2023)
    prior = np.bincount(prior_df["y"].to_numpy(), minlength=len(OUTCOMES)) / prior_df.height
    null = np.tile(prior, (len(result.y_test), 1))
    null_loss = log_loss(result.y_test, null)
    actual_runs = np.asarray(OUTCOMES, dtype=float)[result.y_test]

    print("selected rounds:", list(result.best_iterations))
    print("calibration bias:", np.round(result.bias, 4))
    for name, probabilities in [
        ("null", null),
        ("style raw", result.raw_test),
        ("style + independent bias", result.calibrated_test),
    ]:
        loss = log_loss(result.y_test, probabilities)
        expected = expected_runs(probabilities)
        print(
            f"{name:26s} loss {loss:.6f} gain {100*(null_loss-loss)/null_loss:.3f}% "
            f"mean {expected.mean():.4f} rmse {np.sqrt(np.mean((expected-actual_runs)**2)):.4f} "
            f"boundary_auc {roc_auc_score((result.y_test >= 4).astype(int), probabilities[:, 4:].sum(axis=1)):.4f}"
        )

    ci = clustered_log_loss_gain(
        result.y_test, result.raw_test, result.calibrated_test,
        result.test_match_ids, n_boot=5_000,
    )
    print("calibration gain and 95% CI:", ci)


if __name__ == "__main__":
    main()
