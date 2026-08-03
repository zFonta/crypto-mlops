"""Entrenamiento de Prophet + registro y promocion en MLflow.

La promocion es incondicional: cada corrida registra una version nueva y le
asigna el alias `production`. El control no pasa por comparar metricas entre
modelos sino por *cuando* se reentrena: cada 24 h por schedule, o cuando el
precio se sale de la banda [yhat_lower, yhat_upper] (DAG `crypto_monitor_band`).
"""
import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from common.config import (
    BACKCAST_HOURS,
    FORECAST_HOURS,
    MAX_SAMPLE_WEIGHT,
    MLFLOW_EXPERIMENT,
    PROPHET_PARAMS,
    PRODUCTION_ALIAS,
    TRAIN_DAYS,
    WEIGHT_HALF_LIFE_DAYS,
    model_name,
)
from common.db import get_engine, load_prices, prune_predictions, save_predictions

log = logging.getLogger(__name__)

# Minimo de velas para que el ajuste tenga sentido (una semana horaria).
MIN_TRAIN_ROWS = 24 * 7


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el historico a formato Prophet (ds sin timezone, y = close)."""
    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(df["open_time"]).dt.tz_convert("UTC").dt.tz_localize(None),
            "y": df["close"].astype(float),
        }
    )
    return out.dropna().sort_values("ds").reset_index(drop=True)


def _apply_recency_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Repite las filas recientes para que pesen mas en el ajuste.

    Prophet no expone pesos por observacion; repetir una fila N veces le da
    peso N en la verosimilitud. El peso decae exponencialmente con la
    antiguedad (mitad cada WEIGHT_HALF_LIFE_DAYS) y nunca baja de 1.
    """
    age_days = (df["ds"].max() - df["ds"]).dt.total_seconds() / 86400
    weight = MAX_SAMPLE_WEIGHT * 0.5 ** (age_days / WEIGHT_HALF_LIFE_DAYS)
    reps = weight.round().clip(lower=1).astype(int).to_numpy()
    return df.loc[df.index.repeat(reps)].reset_index(drop=True)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _evaluate(model, df: pd.DataFrame) -> dict:
    """Metricas del ajuste sobre la propia ventana de entrenamiento.

    Son informativas: no deciden la promocion. La verificacion real del modelo
    la hace el monitor de banda hora a hora, sobre datos que todavia no existian
    cuando se entreno.
    """
    forecast = model.predict(df[["ds"]])
    y = df["y"].values
    lower, upper = forecast["yhat_lower"].values, forecast["yhat_upper"].values
    return {
        "mape": _mape(y, forecast["yhat"].values),
        "rmse": _rmse(y, forecast["yhat"].values),
        # Ancho medio de la banda como % del precio: si queda muy por debajo de
        # la volatilidad real, el monitor va a disparar reentrenos espurios.
        "band_width_pct": float(np.mean((upper - lower) / y) * 100),
        "coverage": float(np.mean((y >= lower) & (y <= upper)) * 100),
    }


def train_and_register(symbol: str, trigger_reason: str = "scheduled") -> dict:
    """Entrena Prophet para `symbol`, lo registra en MLflow y lo promueve.

    La version nueva siempre queda con el alias `production`. Ademas persiste en
    la tabla `predictions` la banda del modelo desde BACKCAST_HOURS atras hasta
    FORECAST_HOURS adelante, que es de donde leen el monitor y el dashboard.
    """
    from prophet import Prophet  # import perezoso: es costoso

    engine = get_engine()
    raw = load_prices(engine, symbol, days=TRAIN_DAYS)
    if len(raw) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"{symbol}: solo {len(raw)} filas en la BD. "
            "Ejecuta primero el DAG 'crypto_ingest'"
        )

    df = _prepare_dataframe(raw)
    train_df = _apply_recency_weights(df)

    log.info(
        "%s: entrenando con %d velas de %d dias (%d filas tras ponderar, "
        "peso maximo %d, semivida %d dias)",
        symbol, len(df), TRAIN_DAYS, len(train_df),
        MAX_SAMPLE_WEIGHT, WEIGHT_HALF_LIFE_DAYS,
    )
    model = Prophet(**PROPHET_PARAMS)
    model.fit(train_df)

    metrics = _evaluate(model, df)
    log.info("%s: MAPE=%.3f%% RMSE=%.2f banda=%.2f%% del precio (informativos)",
             symbol, metrics["mape"], metrics["rmse"], metrics["band_width_pct"])

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    name = model_name(symbol)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"{symbol}_train") as run:
        mlflow.log_params(
            {
                **PROPHET_PARAMS,
                "symbol": symbol,
                "train_days": TRAIN_DAYS,
                "train_rows": len(df),
                "weighted_rows": len(train_df),
                "max_sample_weight": MAX_SAMPLE_WEIGHT,
                "weight_half_life_days": WEIGHT_HALF_LIFE_DAYS,
                "trigger_reason": trigger_reason,
                "train_start": df["ds"].min(),
                "train_end": df["ds"].max(),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.prophet.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, name)
        new_version = str(registered_model.version)

    client.set_registered_model_alias(name, PRODUCTION_ALIAS, new_version)
    client.set_model_version_tag(name, new_version, "promotion_reason", trigger_reason)
    log.info("%s: version %s promovida a @%s (%s)", symbol, new_version,
             PRODUCTION_ALIAS, trigger_reason)

    # Banda del nuevo modelo: BACKCAST_HOURS hacia atras (para dibujarla sobre la
    # historia y comparar predicho vs real) y FORECAST_HOURS hacia adelante.
    horizon = pd.date_range(
        max(df["ds"].min(), df["ds"].max() - pd.Timedelta(hours=BACKCAST_HOURS)),
        df["ds"].max() + pd.Timedelta(hours=FORECAST_HOURS),
        freq="h",
    )
    forecast = model.predict(pd.DataFrame({"ds": horizon}))
    saved = save_predictions(
        engine, symbol,
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        int(new_version),
    )
    removed = prune_predictions(engine, symbol, int(new_version))
    log.info("%s: %d predicciones persistidas (v%s), %d de versiones viejas borradas",
             symbol, saved, new_version, removed)

    return {
        "symbol": symbol,
        "version": new_version,
        "trigger_reason": trigger_reason,
        **metrics,
    }
