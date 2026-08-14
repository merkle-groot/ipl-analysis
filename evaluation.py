"""Evaluation helpers shared by the modelling notebooks.

Ball-level losses are correlated within a match.  Treating every delivery as
an independent observation therefore makes uncertainty look much smaller than
it is.  The paired bootstrap below resamples whole matches and keeps each
model's predictions paired on exactly the same deliveries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class BootstrapDifference:
    """Candidate improvement over a reference, measured in log loss."""

    estimate: float
    low: float
    high: float


def apply_logit_bias(probabilities: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply class-specific intercepts and return normalized probabilities."""
    p = np.asarray(probabilities, dtype=float)
    b = np.asarray(bias, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(b):
        raise ValueError("probabilities must have one column per bias value")
    logits = np.log(np.clip(p, np.finfo(float).tiny, 1.0)) + b
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def fit_logit_bias(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    l2: float = 1.0,
) -> np.ndarray:
    """Fit regularized multiclass intercepts without changing model rankings.

    The final class is fixed at zero because adding the same constant to every
    logit has no effect.  This is the optimized form of a prior correction: it
    learns only class prevalence shifts and cannot refit feature relationships.
    ``l2`` is applied to the summed negative log likelihood, so a value near one
    is weak for common classes while stabilizing rare outcomes.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("probabilities and y_true must have matching rows")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    n_classes = p.shape[1]
    if set(np.unique(y)) - set(range(n_classes)):
        raise ValueError("y_true contains a class without a probability column")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        bias = np.r_[theta, 0.0]
        calibrated = apply_logit_bias(p, bias)
        loss = -np.log(calibrated[np.arange(len(y)), y]).sum()
        loss += 0.5 * l2 * np.dot(theta, theta)
        residual = calibrated
        residual[np.arange(len(y)), y] -= 1.0
        gradient = residual[:, :-1].sum(axis=0) + l2 * theta
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(n_classes - 1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"calibration optimization failed: {result.message}")
    return np.r_[result.x, 0.0]


def _row_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Return one log-loss contribution per row for binary or multiclass input."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    eps = np.finfo(float).eps

    if p.ndim == 1:
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    if p.ndim == 2:
        if len(p) != len(y):
            raise ValueError("y_true and probabilities must have the same number of rows")
        p = np.clip(p, eps, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        return -np.log(p[np.arange(len(y)), y])
    raise ValueError("probabilities must be a 1-D binary vector or a 2-D class matrix")


def clustered_log_loss_gain(
    y_true: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2_000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> BootstrapDifference:
    """Paired match-cluster bootstrap for ``reference loss - candidate loss``.

    A positive result favours the candidate.  Resampling match-level sums and
    counts reproduces resampling every delivery in the selected matches while
    avoiding the memory cost of materialising those rows on every draw.
    """
    y = np.asarray(y_true)
    group = np.asarray(groups)
    if len(y) != len(group):
        raise ValueError("y_true and groups must have the same number of rows")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    row_gain = _row_log_loss(y, reference) - _row_log_loss(y, candidate)
    _, codes = np.unique(group, return_inverse=True)
    n_groups = int(codes.max()) + 1
    group_gain = np.bincount(codes, weights=row_gain, minlength=n_groups)
    group_size = np.bincount(codes, minlength=n_groups)

    rng = np.random.default_rng(random_state)
    draws = np.empty(n_boot, dtype=float)
    # Chunk the bootstrap so large n_boot values do not create a large matrix.
    chunk = 250
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        sample = rng.integers(0, n_groups, size=(stop - start, n_groups))
        draws[start:stop] = group_gain[sample].sum(axis=1) / group_size[sample].sum(axis=1)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return BootstrapDifference(float(row_gain.mean()), float(low), float(high))
