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


def block_bootstrap(difference: np.ndarray, mask: np.ndarray | None = None, block: int = 7,
                    repetitions: int = 10_000, seed: int = 42) -> dict[str, float]:
    """IC del 95 % (percentiles) de la diferencia media diaria; positivo favorece al modelo evaluado.

    Remuestrea bloques de `block` días consecutivos sobre todo el periodo; con `mask`, cada réplica
    promedia solo los días seleccionados (por ejemplo, días con publicaciones en la semana previa).
    """
    rng = np.random.default_rng(seed)
    n = len(difference); starts = np.arange(n - block + 1); needed = int(np.ceil(n / block))
    selected = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        index = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, size=needed)])[:n]
        keep = selected[index]
        estimates[i] = difference[index][keep].mean() if keep.any() else np.nan
    low, high = np.nanquantile(estimates, [0.025, 0.975])
    return {"media": float(difference[selected].mean()), "ic95_inf": float(low), "ic95_sup": float(high),
            "prob_mejora": float(np.nanmean(estimates > 0)), "n_dias": int(selected.sum())}
