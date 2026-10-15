"""Construcción causal de variables y de los conjuntos de variables oficiales.

Reproduce exactamente la ingeniería de variables del proyecto de grado: todos los
rezagos son positivos, las ventanas de X excluyen el día t (shift(1).rolling) y las
variables macroeconómicas usan solo el mes calendario anterior completo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DATE, TARGET, dataset_path, load_yaml


def _monthly_previous(df: pd.DataFrame, variable: str) -> tuple[pd.Series, pd.Series]:
    """Nivel y variación porcentual del mes completo anterior, repetidos en cada día del mes."""
    periods = df[DATE].dt.to_period("M")
    monthly = df.assign(_period=periods).groupby("_period", sort=True)[variable].agg(["first", "nunique"])
    if (monthly["nunique"] != 1).any():
        raise ValueError(f"{variable} tiene más de un valor dentro de un mismo mes.")
    level = monthly["first"].shift(1)
    variation = monthly["first"].pct_change(fill_method=None).mul(100).shift(1)
    return periods.map(level).astype(float), periods.map(variation).astype(float)


def _expand(definitions: dict[str, list[str]], groups: dict[str, list[str]]) -> dict[str, list[str]]:
    def expand(token: str, trail: tuple[str, ...]) -> list[str]:
        if token in groups:
            return groups[token]
        if token in definitions and token not in trail:
            return [column for child in definitions[token] for column in expand(child, (*trail, token))]
        raise KeyError(f"Token de conjunto de variables desconocido o circular: {token}")

    return {name: list(dict.fromkeys(c for token in tokens for c in expand(token, (name,))))
            for name, tokens in definitions.items()}


def build_features() -> dict[str, Any]:
    """Lee el dataset verificado y devuelve la matriz de variables y los conjuntos oficiales."""
    cfg = load_yaml("features.yaml")
    df = pd.read_csv(dataset_path(), parse_dates=[DATE], low_memory=False).sort_values(DATE).reset_index(drop=True)
    frame = df[[DATE, TARGET]].copy()
    groups: dict[str, list[str]] = {key: [] for key in (
        "price_lags", "fundamental_lags", "enso_default", "enso_alternative", "calendar", "twitter_basic_lag1",
        "twitter_lags", "twitter_rolling", "twitter_balance_rolling", "macro_causal")}

    for lag in cfg["price"]["lags"]:
        frame[f"price_lag_{lag}"] = df[TARGET].shift(lag); groups["price_lags"].append(f"price_lag_{lag}")

    for variable in cfg["fundamentals"]["lag_variables"]:
        for lag in cfg["fundamentals"]["lags"]:
            name = f"{variable}_lag_{lag}"
            frame[name] = df[variable].shift(lag); groups["fundamental_lags"].append(name)

    for source, group in (("oni", "enso_default"), ("enso_index", "enso_alternative")):
        frame[f"{source}_lag_1"] = df[source].shift(1); groups[group].append(f"{source}_lag_1")

    for variable in cfg["calendar"]["direct_candidates"]:
        frame[variable] = df[variable]; groups["calendar"].append(variable)

    for variable in cfg["twitter"]["lag_variables"]:
        for lag in cfg["twitter"]["lags"]:
            name = f"{variable}_lag_{lag}"
            frame[name] = df[variable].shift(lag); groups["twitter_lags"].append(name)
            if lag == 1:
                groups["twitter_basic_lag1"].append(name)

    shift = int(cfg["twitter"]["rolling_shift"])
    if shift < 1:
        raise ValueError("rolling_shift debe ser al menos 1 para excluir el día t.")
    for window in cfg["twitter"]["rolling_windows"]:
        for variable in cfg["twitter"]["rolling_variables"]:
            name = f"{variable}_roll_{window}"
            frame[name] = df[variable].shift(shift).rolling(window, min_periods=window).sum()
            groups["twitter_rolling"].append(name)
        balance = f"twitter_balance_roll_{window}"
        frame[balance] = frame[f"n_positivo_roll_{window}"] - frame[f"n_negativo_roll_{window}"]
        groups["twitter_balance_rolling"].append(balance)

    for variable in cfg["macro"]["level_candidates"]:
        prefix = "ipc" if variable == "ipc" else "ipp"
        level, variation = _monthly_previous(df, variable)
        frame[f"{prefix}_lag_1m"], frame[f"{prefix}_variacion_lag_1m"] = level, variation
        groups["macro_causal"] += [f"{prefix}_lag_1m", f"{prefix}_variacion_lag_1m"]

    feature_sets = _expand(cfg["feature_sets"], groups)
    excluded = set(cfg["excluded"]["control_variables"]) | set(cfg["excluded"]["target_derived_exogenous"])
    for name, columns in feature_sets.items():
        if set(columns) & excluded:
            raise ValueError(f"{name} contiene variables excluidas: {sorted(set(columns) & excluded)}")

    frame["persistence_prediction"] = frame["price_lag_1"]
    frame["target_change"] = frame[TARGET] - frame["persistence_prediction"]
    return {"frame": frame, "feature_sets": feature_sets}


def split(frame: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
    """Elimina filas sin historia suficiente y separa TRAIN, VALIDATION y TEST por fecha."""
    selected = frame[[DATE, TARGET, "persistence_prediction", "target_change", *features]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(DATE)
    bounds = load_yaml("split.yaml")
    return {name: selected[selected[DATE].between(pd.Timestamp(b["start"]), pd.Timestamp(b["end"]))].copy()
            for name, b in bounds.items()}
