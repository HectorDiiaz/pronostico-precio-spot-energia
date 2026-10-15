"""Métricas de error y bootstrap por bloques móviles."""

from __future__ import annotations

import numpy as np


def regression_metrics(actual, predicted) -> dict[str, float]:
    y, y_hat = np.asarray(actual, float), np.asarray(predicted, float)
    errors = y - y_hat
    denominator = np.abs(y) + np.abs(y_hat)
    smape = np.divide(200 * np.abs(errors), denominator, out=np.zeros_like(errors), where=denominator != 0)
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "r2": float(1 - np.sum(errors ** 2) / np.sum((y - y.mean()) ** 2)), "smape": float(np.mean(smape))}


def block_bootstrap(difference: np.ndarray, block: int = 7, repetitions: int = 10_000, seed: int = 42) -> dict[str, float]:
    """IC del 95 % de la mejora media diaria; positivo favorece al modelo comparado."""
    rng = np.random.default_rng(seed)
    n = len(difference); starts = np.arange(n - block + 1); needed = int(np.ceil(n / block))
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        chosen = rng.choice(starts, size=needed, replace=True)
        estimates[i] = np.concatenate([difference[s:s + block] for s in chosen])[:n].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mejora_media": float(difference.mean()), "ic95_inf": float(low), "ic95_sup": float(high),
            "prob_mejora": float(np.mean(estimates > 0)), "dias_gana_pct": float(np.mean(difference > 0) * 100)}
