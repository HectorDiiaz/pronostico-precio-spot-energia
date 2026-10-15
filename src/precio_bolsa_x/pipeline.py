"""Entrenamiento, evaluación, contraste de X, SHAP y verificación del modelo seleccionado."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATE, MODELS, OUTPUTS, REFERENCE, TARGET, load_yaml
from .features import build_features, split
from .metrics import block_bootstrap, regression_metrics

CFG = load_yaml("model.yaml")
SEED = int(CFG["seed"])
TWITTER_PREFIXES = ("n_total_tweets", "n_positivo", "n_negativo", "sentiment_index", "twitter_balance")
GROUP_LABELS = {"precio": "Precio rezagado", "calendario": "Calendario", "fundamentales": "Generación y embalses",
                "twitter": "Twitter / X", "macro": "Macro (IPC/IPP)", "enso": "ENSO (ONI)"}


def _estimator(algorithm: str) -> Pipeline:
    if algorithm == "gradient_boosting_huber":
        return Pipeline([("model", GradientBoostingRegressor(loss="huber", random_state=SEED))])
    if algorithm == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    if algorithm == "elastic_net":
        return Pipeline([("scale", StandardScaler()), ("model", ElasticNet(max_iter=20000, random_state=SEED))])
    raise ValueError(algorithm)


def _search(algorithm: str, train: pd.DataFrame, features: list[str], n_jobs: int) -> GridSearchCV:
    grid = {f"model__{k}": v for k, v in CFG["search"][algorithm].items()}
    search = GridSearchCV(_estimator(algorithm), grid, scoring="neg_mean_absolute_error",
                          cv=TimeSeriesSplit(n_splits=CFG["search"]["cv_splits"]), n_jobs=n_jobs, refit=True, error_score="raise")
    return search.fit(train[features], train["target_change"])


def _forecast(model, part: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    change = model.predict(part[features])
    return part["persistence_prediction"].to_numpy(float) + change, change


def _group(feature: str) -> str:
    if feature.startswith("price_lag_"): return "precio"
    if feature.startswith(TWITTER_PREFIXES): return "twitter"
    if feature.startswith(("nivel_embalses", "pct_")): return "fundamentales"
    if feature.startswith("oni"): return "enso"
    if feature.startswith(("ipc", "ipp")): return "macro"
    return "calendario"


def train(n_jobs: int = 2) -> dict:
    """Repite la búsqueda de hiperparámetros del ganador y del control sin X, y guarda ambos modelos."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    built = build_features(); rows = []
    for role, feature_set in (("ganador", CFG["winner"]["feature_set"]), ("control_sin_x", CFG["control_feature_set"])):
        features = built["feature_sets"][feature_set]; parts = split(built["frame"], features)
        started = time.perf_counter()
        search = _search("gradient_boosting_huber", parts["train"], features, n_jobs)
        params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        joblib.dump(search.best_estimator_, OUTPUTS / f"gb_huber_{feature_set}.joblib")
        row = {"rol": role, "feature_set": feature_set, "n_features": len(features), "parametros": json.dumps(params, sort_keys=True),
               "segundos": round(time.perf_counter() - started, 2)}
        for name, part in parts.items():
            prediction, _ = _forecast(search.best_estimator_, part, features)
            row |= {f"{name}_{k}": v for k, v in regression_metrics(part[TARGET], prediction).items()}
            row[f"n_{name}"] = len(part)
        rows.append(row)
        print(f"[train] {role:14s} {feature_set:20s} VAL MAE {row['validation_mae']:.4f}  TEST MAE {row['test_mae']:.4f}  {params}")
    result = pd.DataFrame(rows); result.to_csv(OUTPUTS / "modelos_entrenados.csv", index=False, encoding="utf-8-sig")
    if json.loads(result.parametros[0]) != CFG["winner"]["frozen_parameters"]:
        raise AssertionError("La búsqueda no recuperó los hiperparámetros congelados del ganador.")
    return {"models": result}


def evaluate() -> dict:
    """Métricas del ganador, comparación con persistencia y contraste confirmatorio de X con bootstrap por bloques."""
    built = build_features(); boot = CFG["bootstrap"]; out = {}
    winner_set, control_set = CFG["winner"]["feature_set"], CFG["control_feature_set"]
    model = joblib.load(OUTPUTS / f"gb_huber_{winner_set}.joblib"); control = joblib.load(OUTPUTS / f"gb_huber_{control_set}.joblib")
    features = built["feature_sets"][winner_set]; parts = split(built["frame"], features)

    rows, comparison, predictions = [], [], []
    for name in ("validation", "test"):
        part = parts[name]; prediction, change = _forecast(model, part, features)
        rows.append({"split": name, "modelo": "gradient_boosting_huber", **regression_metrics(part[TARGET], prediction)})
        rows.append({"split": name, "modelo": "persistencia_t1", **regression_metrics(part[TARGET], part["persistence_prediction"])})
        difference = (np.abs(part[TARGET] - part["persistence_prediction"]) - np.abs(part[TARGET] - prediction)).to_numpy(float)
        comparison.append({"split": name, **block_bootstrap(difference, boot["block_days"], boot["repetitions"], SEED)})
        predictions.append(part[[DATE, TARGET, "persistence_prediction"]].assign(predicted_change=change, prediction=prediction, split=name))
    pd.DataFrame(rows).to_csv(OUTPUTS / "metricas_ganador.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison).to_csv(OUTPUTS / "bootstrap_vs_persistencia.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions).rename(columns={TARGET: "actual"}).to_csv(OUTPUTS / "predicciones.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    test = parts["test"]; control_features = built["feature_sets"][control_set]
    control_parts = split(built["frame"], control_features)["test"]
    if not test[DATE].reset_index(drop=True).equals(control_parts[DATE].reset_index(drop=True)):
        raise AssertionError("Los dos modelos del contraste no cubren las mismas fechas de TEST.")
    with_x, _ = _forecast(model, test, features); without_x, _ = _forecast(control, control_parts, control_features)
    error_with, error_without = np.abs(test[TARGET].to_numpy() - with_x), np.abs(control_parts[TARGET].to_numpy() - without_x)
    confirm = {"mae_sin_x": float(error_without.mean()), "mae_con_x": float(error_with.mean()),
               **block_bootstrap(error_without - error_with, boot["block_days"], boot["repetitions"], SEED)}
    pd.DataFrame([confirm]).to_csv(OUTPUTS / "contraste_x_test.csv", index=False, encoding="utf-8-sig")
    out |= {"metricas": rows, "vs_persistencia": comparison, "contraste_x": confirm}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def explain() -> pd.DataFrame:
    """Participación SHAP de cada grupo de variables en VALIDATION y TEST."""
    import shap

    built = build_features(); feature_set = CFG["winner"]["feature_set"]; features = built["feature_sets"][feature_set]
    parts = split(built["frame"], features); model = joblib.load(OUTPUTS / f"gb_huber_{feature_set}.joblib")
    estimator = model.named_steps["model"]; explainer = shap.TreeExplainer(estimator)
    groups = pd.Series({f: _group(f) for f in features}); rows = []
    for name in ("validation", "test"):
        values = pd.DataFrame(explainer.shap_values(parts[name][features]), columns=features)
        by_group = values.abs().T.groupby(groups).sum().T.mean()
        rows += [{"split": name, "group": g, "label": GROUP_LABELS[g], "mean_abs_shap": float(v), "share_pct": float(v / by_group.sum() * 100)}
                 for g, v in by_group.items()]
    result = pd.DataFrame(rows).sort_values(["split", "share_pct"], ascending=[True, False])
    result.to_csv(OUTPUTS / "shap_grupos.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    return result


def select(n_jobs: int = 2) -> pd.DataFrame:
    """Opcional: repite la selección completa (3 algoritmos × 12 conjuntos) ordenando solo por VALIDATION."""
    built = build_features(); rows = []
    for feature_set, features in built["feature_sets"].items():
        parts = split(built["frame"], features)
        for algorithm in ("ridge", "elastic_net", "gradient_boosting_huber"):
            search = _search(algorithm, parts["train"], features, n_jobs)
            prediction, _ = _forecast(search.best_estimator_, parts["validation"], features)
            scores = regression_metrics(parts["validation"][TARGET], prediction)
            rows.append({"algorithm": algorithm, "feature_set": feature_set, "validation_mae": scores["mae"], "validation_rmse": scores["rmse"]})
            print(f"[select] {algorithm:24s} {feature_set:22s} VAL MAE {scores['mae']:.4f}")
    ranking = pd.DataFrame(rows).sort_values(["validation_mae", "validation_rmse"]).reset_index(drop=True)
    ranking.to_csv(OUTPUTS / "ranking_seleccion.csv", index=False, encoding="utf-8-sig")
    winner = ranking.iloc[0]
    if (winner.algorithm, winner.feature_set) != (CFG["winner"]["algorithm"], CFG["winner"]["feature_set"]):
        raise AssertionError(f"La selección no reprodujo el ganador oficial: {winner.algorithm} / {winner.feature_set}")
    print(f"[select] Ganador reproducido: {winner.algorithm} / {winner.feature_set}")
    return ranking


def verify() -> bool:
    """Compara los resultados regenerados con los artefactos oficiales de la tesis."""
    tol = float(CFG["tolerance"]); checks = []

    def check(name: str, got: float, expected: float, atol: float = tol) -> None:
        checks.append({"verificacion": name, "obtenido": got, "oficial": expected, "diferencia": abs(got - expected), "ok": abs(got - expected) <= atol})

    official = pd.read_csv(REFERENCE / "best_persistence_correction.csv", encoding="utf-8-sig").iloc[0]
    trained = pd.read_csv(OUTPUTS / "modelos_entrenados.csv", encoding="utf-8-sig").iloc[0]
    for split_name in ("train", "validation", "test"):
        for metric in ("mae", "rmse", "r2", "smape"):
            check(f"ganador_{split_name}_{metric}", float(trained[f"{split_name}_{metric}"]), float(official[f"{split_name}_{metric}"]))
    check("ganador_parametros", float(trained.parametros == json.dumps(dict(sorted(json.loads(official.best_parameters.replace("model__", "")).items())), sort_keys=True)), 1.0, 0)

    reference_pred = pd.read_csv(REFERENCE / "test_predictions.csv", encoding="utf-8-sig")
    produced = pd.read_csv(OUTPUTS / "predicciones.csv", encoding="utf-8-sig").query("split == 'test'")
    check("predicciones_test_max_dif", float(np.max(np.abs(produced.prediction.to_numpy() - reference_pred.prediction.to_numpy()))), 0.0)

    official_x = pd.read_csv(REFERENCE / "twitter_test_confirmatory_comparison.csv", encoding="utf-8-sig").iloc[0]
    produced_x = pd.read_csv(OUTPUTS / "contraste_x_test.csv", encoding="utf-8-sig").iloc[0]
    for got, ref in (("mae_sin_x", "mae_without_twitter"), ("mae_con_x", "mae_with_twitter"), ("ic95_inf", "ci95_low"),
                     ("ic95_sup", "ci95_high"), ("prob_mejora", "bootstrap_probability_improvement")):
        check(f"contraste_x_{got}", float(produced_x[got]), float(official_x[ref]))

    official_b = pd.read_csv(REFERENCE / "bootstrap_gb_vs_persistencia.csv", encoding="utf-8-sig")
    produced_b = pd.read_csv(OUTPUTS / "bootstrap_vs_persistencia.csv", encoding="utf-8-sig")
    for (_, o), (_, p) in zip(official_b.iterrows(), produced_b.iterrows()):
        for key in ("mejora_media", "ic95_inf", "ic95_sup", "prob_mejora"):
            check(f"vs_persistencia_{p.split}_{key}", float(p[key]), float(o[key]))

    if (OUTPUTS / "shap_grupos.csv").exists():
        official_s = pd.read_csv(REFERENCE / "group_importance.csv", encoding="utf-8-sig").set_index(["split", "group"])
        produced_s = pd.read_csv(OUTPUTS / "shap_grupos.csv", encoding="utf-8-sig").set_index(["split", "group"])
        for key in official_s.index:
            check(f"shap_{key[0]}_{key[1]}_pct", float(produced_s.loc[key, "share_pct"]), float(official_s.loc[key, "share_pct"]), 1e-4)

    result = pd.DataFrame(checks); result.to_csv(OUTPUTS / "verificacion.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False, formatters={"diferencia": "{:.2e}".format}))
    passed = bool(result.ok.all())
    print(f"\nRESULTADO: {'REPRODUCIDO' if passed else 'DIFERENCIAS ENCONTRADAS'} ({int(result.ok.sum())}/{len(result)} verificaciones)")
    return passed
