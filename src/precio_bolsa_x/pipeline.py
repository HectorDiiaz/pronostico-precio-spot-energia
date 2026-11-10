"""Entrenamiento, evaluación, SHAP y verificación de los dos modelos del trabajo de grado.

- Modelo de dos etapas (mejor modelo): la etapa 1 estima la probabilidad de un cambio extremo del precio
  (a la baja, normal o al alza) y la etapa 2, un boosting por gradiente con pérdida de Huber, pronostica el
  cambio diario con las variables más esas tres probabilidades.
- Boosting por gradiente con pérdida de Huber: modelo independiente, base de la segunda etapa.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, cross_val_predict
from sklearn.pipeline import Pipeline

from .config import DATE, OUTPUTS, REFERENCE, TARGET, load_yaml
from .features import build_features, split
from .metrics import block_bootstrap, regression_metrics

CFG = load_yaml("model.yaml")
SEED = int(CFG["seed"])
PROBS = ["p_baja", "p_normal", "p_alza"]
X_PREFIXES = ("n_total_tweets", "n_positivo", "n_negativo", "sentiment_index", "twitter_balance")
SPLITS = ("train", "validation", "test")


def group(feature: str) -> str:
    if feature.startswith("price_lag_"): return "Precio rezagado"
    if feature.startswith(X_PREFIXES): return "X"
    if feature.startswith(("nivel_embalses", "pct_")): return "Generación y embalses"
    if feature.startswith("oni"): return "ENSO (ONI)"
    if feature in PROBS: return "Probabilidades de la etapa 1"
    if feature.startswith(("ipc", "ipp")): return "Macroeconomía (IPC e IPP)"
    return "Calendario"


def _data(feature_set: str, built: dict | None = None) -> tuple[list[str], dict[str, pd.DataFrame]]:
    built = built or build_features()
    features = built["feature_sets"][feature_set]
    return features, split(built["frame"], features)


def _bootstrap(difference: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    return block_bootstrap(difference, mask, CFG["bootstrap"]["block_days"], CFG["bootstrap"]["repetitions"], SEED)


# --------------------------------------------------------------------------- boosting por gradiente con pérdida de Huber
def fit_gb_huber(parts: dict[str, pd.DataFrame], features: list[str], n_jobs: int = 2) -> tuple[Pipeline, dict]:
    grid = {f"model__{k}": v for k, v in CFG["gb_huber"]["busqueda"].items()}
    search = GridSearchCV(Pipeline([("model", GradientBoostingRegressor(loss="huber", random_state=SEED))]), grid,
                          scoring="neg_mean_absolute_error", cv=TimeSeriesSplit(n_splits=CFG["cv_splits"]),
                          n_jobs=n_jobs, refit=True, error_score="raise")
    search.fit(parts["train"][features], parts["train"]["target_change"])
    return search.best_estimator_, {k.removeprefix("model__"): v for k, v in search.best_params_.items()}


def predict_gb_huber(model: Pipeline, part: pd.DataFrame, features: list[str]) -> np.ndarray:
    return part["persistence_prediction"].to_numpy(float) + model.predict(part[features])


# --------------------------------------------------------------------------- modelo de dos etapas
def fit_two_stage(parts: dict[str, pd.DataFrame], features: list[str], params: dict | None = None, n_jobs: int = 2) -> dict:
    """Ajusta las dos etapas en TRAIN; con `params` fija los hiperparámetros del regresor en lugar de buscarlos."""
    spec = CFG["dos_etapas"]; train = parts["train"]; change = train["target_change"].to_numpy(float)
    low, high = np.percentile(change, spec["percentiles_cambio_extremo"])
    classes = np.where(change >= high, 2, np.where(change <= low, 0, 1))
    classifier = HistGradientBoostingClassifier(**spec["clasificador"], random_state=SEED)
    # Probabilidades de TRAIN fuera de pliegue, para que la etapa 2 no aprenda de probabilidades sobreajustadas.
    p_train = cross_val_predict(classifier, train[features], classes, cv=KFold(spec["pliegues_fuera_de_muestra"]), method="predict_proba")
    classifier.fit(train[features], classes)
    X_train = np.column_stack([train[features].to_numpy(float), p_train])
    if params is None:
        search = GridSearchCV(GradientBoostingRegressor(loss="huber", random_state=SEED), spec["busqueda_regresor"],
                              scoring="neg_mean_absolute_error", cv=TimeSeriesSplit(n_splits=CFG["cv_splits"]),
                              n_jobs=n_jobs, refit=True, error_score="raise").fit(X_train, change)
        regressor, params = search.best_estimator_, search.best_params_
    else:
        regressor = GradientBoostingRegressor(loss="huber", random_state=SEED, **params).fit(X_train, change)
    return {"clasificador": classifier, "regresor": regressor, "variables": list(features), "parametros": dict(params),
            "umbrales": [float(low), float(high)], "p_train": p_train}


def stage_inputs(model: dict, part: pd.DataFrame, probabilities: np.ndarray | None = None) -> np.ndarray:
    """Matriz de la etapa 2: variables de la combinación más las tres probabilidades de la etapa 1."""
    if probabilities is None:
        probabilities = model["clasificador"].predict_proba(part[model["variables"]])
    return np.column_stack([part[model["variables"]].to_numpy(float), probabilities])


def predict_two_stage(model: dict, part: pd.DataFrame, probabilities: np.ndarray | None = None) -> np.ndarray:
    return part["persistence_prediction"].to_numpy(float) + model["regresor"].predict(stage_inputs(model, part, probabilities))


def _two_stage_forecasts(model: dict, parts: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    # En TRAIN se usan las probabilidades fuera de pliegue con que se ajustó la etapa 2.
    return {name: predict_two_stage(model, part, model["p_train"] if name == "train" else None) for name, part in parts.items()}


# --------------------------------------------------------------------------- comandos
def train(n_jobs: int = 2) -> pd.DataFrame:
    """Repite la búsqueda de hiperparámetros de los dos modelos y guarda los modelos y sus pronósticos."""
    OUTPUTS.mkdir(parents=True, exist_ok=True); built = build_features(); rows, predictions = [], []

    def record(name: str, feature_set: str, n_features: int, params: dict, seconds: float, parts, forecasts) -> None:
        row = {"modelo": name, "feature_set": feature_set, "n_variables": n_features,
               "parametros": json.dumps(params, sort_keys=True), "segundos": round(seconds, 2)}
        for split_name, part in parts.items():
            row |= {f"{split_name}_{k}": v for k, v in regression_metrics(part[TARGET], forecasts[split_name]).items()}
            row[f"n_{split_name}"] = len(part)
            if split_name != "train":
                predictions.append(part[[DATE, TARGET, "persistence_prediction"]].assign(modelo=name, split=split_name, pronostico=forecasts[split_name]))
        rows.append(row)
        print(f"[train] {name:18s} VAL MAE {row['validation_mae']:.4f}  TEST MAE {row['test_mae']:.4f}  {params}", flush=True)

    spec = CFG["dos_etapas"]
    features, parts = _data(spec["feature_set"], built); started = time.perf_counter()
    model = fit_two_stage(parts, features, n_jobs=n_jobs)
    joblib.dump({k: v for k, v in model.items() if k != "p_train"}, OUTPUTS / "dos_etapas.joblib")
    record("dos_etapas", spec["feature_set"], len(features), model["parametros"], time.perf_counter() - started, parts,
           _two_stage_forecasts(model, parts))

    # Contraste con y sin X: misma combinación sin las variables de X, con los hiperparámetros del ganador.
    features_c, parts_c = _data(spec["control_feature_set"], built); started = time.perf_counter()
    control = fit_two_stage(parts_c, features_c, params=model["parametros"])
    joblib.dump({k: v for k, v in control.items() if k != "p_train"}, OUTPUTS / "dos_etapas_sin_x.joblib")
    record("dos_etapas_sin_x", spec["control_feature_set"], len(features_c), control["parametros"], time.perf_counter() - started,
           parts_c, _two_stage_forecasts(control, parts_c))

    spec = CFG["gb_huber"]
    features, parts = _data(spec["feature_set"], built); started = time.perf_counter()
    gb, params = fit_gb_huber(parts, features, n_jobs)
    joblib.dump(gb, OUTPUTS / "gb_huber.joblib")
    record("gb_huber", spec["feature_set"], len(features), params, time.perf_counter() - started, parts,
           {name: predict_gb_huber(gb, part, features) for name, part in parts.items()})

    result = pd.DataFrame(rows); result.to_csv(OUTPUTS / "modelos_entrenados.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions).rename(columns={TARGET: "real", "persistence_prediction": "persistencia"}).to_csv(
        OUTPUTS / "predicciones.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    return result


def evaluate() -> dict[str, pd.DataFrame]:
    """Análisis del mejor modelo: diferencia con la persistencia, contraste con y sin X y días con publicaciones."""
    built = build_features(); spec = CFG["dos_etapas"]
    features, parts = _data(spec["feature_set"], built)
    _, parts_c = _data(spec["control_feature_set"], built)
    model, control = joblib.load(OUTPUTS / "dos_etapas.joblib"), joblib.load(OUTPUTS / "dos_etapas_sin_x.joblib")
    vs_persistence, contrast, by_posts = [], [], []
    for name in ("validation", "test"):
        part, part_c = parts[name], parts_c[name]
        if not part[DATE].reset_index(drop=True).equals(part_c[DATE].reset_index(drop=True)):
            raise AssertionError("Las versiones con y sin X no cubren las mismas fechas.")
        y = part[TARGET].to_numpy(float); persistence = part["persistence_prediction"].to_numpy(float)
        with_x, without_x = predict_two_stage(model, part), predict_two_stage(control, part_c)

        d = np.abs(y - persistence) - np.abs(y - with_x)
        vs_persistence.append({"periodo": name, "mae_persistencia": float(np.abs(y - persistence).mean()),
                               "mae_modelo": float(np.abs(y - with_x).mean()), "dias_modelo_mejor": int((d > 0).sum()), **_bootstrap(d)})

        d = np.abs(y - without_x) - np.abs(y - with_x)
        mae_without, mae_with = float(np.abs(y - without_x).mean()), float(np.abs(y - with_x).mean())
        contrast.append({"periodo": name, "mae_sin_x": mae_without, "mae_con_x": mae_with,
                         "mejora_pct": 100 * (mae_without - mae_with) / mae_without, "dias_con_x_mejor": int((d > 0).sum()),
                         "dias_sin_x_mejor": int((d < 0).sum()), **_bootstrap(d)})

        posts = split(built["frame"], list(dict.fromkeys([*features, "n_total_tweets_roll_7"])))[name]
        if not posts[DATE].reset_index(drop=True).equals(part[DATE].reset_index(drop=True)):
            raise AssertionError("Las fechas del análisis por publicaciones no coinciden.")
        had_posts = posts["n_total_tweets_roll_7"].to_numpy(float) > 0
        for label, mask in (("con publicaciones", had_posts), ("sin publicaciones", ~had_posts)):
            by_posts.append({"periodo": name, "ventana": "7 días previos", "grupo": label, **_bootstrap(d, mask)})

    out = {"vs_persistencia": pd.DataFrame(vs_persistence), "contraste_con_sin_x": pd.DataFrame(contrast),
           "contraste_por_publicaciones": pd.DataFrame(by_posts)}
    for name, frame in out.items():
        frame.to_csv(OUTPUTS / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[{name}]\n{frame.round(4).to_string(index=False)}", flush=True)
    return out


def explain() -> pd.DataFrame:
    """Participación SHAP de cada grupo de variables en las dos etapas del mejor modelo."""
    import shap

    built = build_features(); features, parts = _data(CFG["dos_etapas"]["feature_set"], built)
    model = joblib.load(OUTPUTS / "dos_etapas.joblib"); rows = []
    names_2 = features + PROBS
    explainer_2 = shap.TreeExplainer(model["regresor"]); explainer_1 = shap.TreeExplainer(model["clasificador"])
    for name in ("validation", "test"):
        X2 = stage_inputs(model, parts[name]); values = explainer_2.shap_values(X2)
        error = float(np.max(np.abs(float(np.ravel(explainer_2.expected_value)[0]) + values.sum(1) - model["regresor"].predict(X2))))
        if error > 1e-6:
            raise AssertionError(f"SHAP no reproduce los pronósticos de la etapa 2 (error {error}).")
        importance = pd.Series(np.abs(values).mean(0), index=names_2)
        by_group = importance.groupby(pd.Series({f: group(f) for f in names_2})).sum()
        rows += [{"etapa": 2, "periodo": name, "grupo": g, "participacion_pct": float(100 * v / by_group.sum())} for g, v in by_group.items()]

        values = np.abs(np.asarray(explainer_1.shap_values(parts[name][features])))
        if values.ndim == 3:  # multiclase: promedio sobre las tres clases
            values = values.mean(axis=0) if values.shape[0] == 3 else values.mean(axis=2)
        importance = pd.Series(values.mean(0), index=features)
        by_group = importance.groupby(pd.Series({f: group(f) for f in features})).sum()
        rows += [{"etapa": 1, "periodo": name, "grupo": g, "participacion_pct": float(100 * v / by_group.sum())} for g, v in by_group.items()]
    result = pd.DataFrame(rows).sort_values(["etapa", "periodo", "participacion_pct"], ascending=[True, True, False])
    result.to_csv(OUTPUTS / "shap_por_grupo.csv", index=False, encoding="utf-8-sig")
    table = result.pivot_table(index="grupo", columns=["etapa", "periodo"], values="participacion_pct").round(2)
    print("\n[shap_por_grupo]\n" + table.to_string(), flush=True)
    return result


def verify() -> bool:
    """Compara los resultados regenerados con las cifras oficiales del trabajo de grado (carpeta reference/)."""
    tol = float(CFG["tolerance"]); checks = []

    def check(name: str, got: float, expected: float, atol: float = tol) -> None:
        checks.append({"verificacion": name, "obtenido": got, "oficial": expected, "diferencia": abs(got - expected),
                       "ok": abs(got - expected) <= atol})

    official = json.loads((REFERENCE / "metricas_oficiales.json").read_text(encoding="utf-8"))
    trained = pd.read_csv(OUTPUTS / "modelos_entrenados.csv", encoding="utf-8-sig").set_index("modelo")
    for model_name in ("dos_etapas", "gb_huber"):
        row, ref = trained.loc[model_name], official[model_name]
        for split_name in SPLITS:
            for metric in ("mae", "rmse", "r2", "smape"):
                check(f"{model_name}_{split_name}_{metric}", float(row[f"{split_name}_{metric}"]), float(ref[f"{split_name}_{metric}"]))
        check(f"{model_name}_hiperparametros", float(json.loads(row.parametros) == CFG[model_name]["frozen_parameters"]), 1.0, 0)
    model = joblib.load(OUTPUTS / "dos_etapas.joblib")
    for got, expected, label in zip(model["umbrales"], official["dos_etapas"]["umbrales"], ("baja", "alza")):
        check(f"dos_etapas_umbral_{label}", got, expected)

    reference = pd.read_csv(REFERENCE / "dos_etapas" / "pronostico_prueba.csv", encoding="utf-8-sig")
    produced = pd.read_csv(OUTPUTS / "predicciones.csv", encoding="utf-8-sig").query("modelo == 'dos_etapas' and split == 'test'")
    check("dos_etapas_pronosticos_prueba_max_dif",
          float(np.max(np.abs(produced.pronostico.to_numpy() - reference.pronostico.to_numpy()))), 0.0)

    for name, keys in (("vs_persistencia", ("mae_persistencia", "mae_modelo", "media", "ic95_inf", "ic95_sup", "prob_mejora")),
                       ("contraste_con_sin_x", ("mae_sin_x", "mae_con_x", "media", "ic95_inf", "ic95_sup", "prob_mejora")),
                       ("contraste_por_publicaciones", ("media", "ic95_inf", "ic95_sup", "prob_mejora", "n_dias"))):
        if not (OUTPUTS / f"{name}.csv").exists():
            continue
        ref = pd.read_csv(REFERENCE / "dos_etapas" / f"{name}.csv", encoding="utf-8-sig")
        got = pd.read_csv(OUTPUTS / f"{name}.csv", encoding="utf-8-sig")
        index = ["periodo", "grupo"] if "grupo" in ref else ["periodo"]
        ref, got = ref.set_index(index), got.set_index(index)
        for key in ref.index:
            label = "_".join(key) if isinstance(key, tuple) else key
            for column in keys:
                check(f"{name}_{label}_{column}".replace(" ", "_"), float(got.loc[key, column]), float(ref.loc[key, column]))

    if (OUTPUTS / "shap_por_grupo.csv").exists():
        ref = pd.read_csv(REFERENCE / "dos_etapas" / "shap_por_grupo.csv", encoding="utf-8-sig").set_index(["etapa", "periodo", "grupo"])
        got = pd.read_csv(OUTPUTS / "shap_por_grupo.csv", encoding="utf-8-sig").set_index(["etapa", "periodo", "grupo"])
        for key in ref.index:
            check(f"shap_etapa{key[0]}_{key[1]}_{key[2]}_pct".replace(" ", "_"),
                  float(got.loc[key, "participacion_pct"]), float(ref.loc[key, "participacion_pct"]), 1e-4)

    result = pd.DataFrame(checks); result.to_csv(OUTPUTS / "verificacion.csv", index=False, encoding="utf-8-sig")
    print("\n" + result.to_string(index=False, formatters={"diferencia": "{:.2e}".format}))
    passed = bool(result.ok.all())
    print(f"\nRESULTADO: {'REPRODUCIDO' if passed else 'DIFERENCIAS ENCONTRADAS'} ({int(result.ok.sum())}/{len(result)} verificaciones)")
    return passed
