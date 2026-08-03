"""API REST del pipeline crypto-mlops.

Expone en JSON la banda de prediccion del modelo en produccion. Es una capa de
solo lectura sobre la tabla `predictions`: el forecast ya lo calculo y persistio
el DAG `crypto_train_models` en cada promocion, asi que este servicio no carga
Prophet ni consulta MLflow. Comparte la configuracion con los DAGs y el
dashboard via el modulo `common/`.

Documentacion interactiva en /api/docs
"""
from datetime import datetime

from fastapi import FastAPI, HTTPException, Path, Query
from sqlalchemy.exc import SQLAlchemyError

from common.config import (
    FORECAST_DAYS,
    FORECAST_HOURS,
    LOCAL_TZ,
    PROPHET_PARAMS,
    SYMBOLS,
)
from common.db import get_engine, latest_vs_band, load_forecast

# root_path: nginx sirve la API bajo /api/ y recorta el prefijo, pero Swagger
# necesita saberlo para armar bien la URL del openapi.json.
app = FastAPI(
    title="Crypto MLOps API",
    description=__doc__,
    version="1.0.0",
    root_path="/api",
)

engine = get_engine()

INTERVAL_WIDTH = PROPHET_PARAMS["interval_width"]


def _iso(ts) -> str:
    """Timestamp de la BD (UTC) -> ISO 8601 en la zona del proyecto (GMT-3)."""
    return ts.astimezone(LOCAL_TZ).isoformat()


def _check_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(
            status_code=404,
            detail=f"Simbolo '{symbol}' no seguido. Disponibles: {SYMBOLS}",
        )
    return symbol


@app.get("/", summary="Indice de endpoints")
def index() -> dict:
    return {
        "servicio": "Crypto MLOps API",
        "docs": "/api/docs",
        "endpoints": [
            "/api/health",
            "/api/symbols",
            "/api/predictions/{symbol}?hours=24",
            "/api/status/{symbol}",
        ],
    }


@app.get("/health", summary="Estado del servicio y de la conexion a Postgres")
def health() -> dict:
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"Postgres no responde: {exc}")
    return {"status": "ok", "timestamp": datetime.now(LOCAL_TZ).isoformat()}


@app.get("/symbols", summary="Simbolos seguidos y parametros del forecast")
def symbols() -> dict:
    return {
        "symbols": SYMBOLS,
        "forecast_days": FORECAST_DAYS,
        "interval_width": INTERVAL_WIDTH,
        "granularity": "1h",
        "timezone": "GMT-3",
    }


@app.get(
    "/predictions/{symbol}",
    summary="Banda de prediccion hacia adelante, hora a hora",
)
def predictions(
    symbol: str = Path(description="Par de Binance, p. ej. BTCUSDT"),
    hours: int = Query(
        24, ge=1, le=FORECAST_HOURS, description="Horas de forecast a devolver"
    ),
) -> dict:
    symbol = _check_symbol(symbol)
    df = load_forecast(engine, symbol, hours)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No hay forecast vigente para {symbol}. "
                   "Ejecuta el DAG 'crypto_train_models' en Airflow.",
        )

    return {
        "symbol": symbol,
        "model_version": int(df["model_version"].iloc[0]),
        "interval_width": INTERVAL_WIDTH,
        "timezone": "GMT-3",
        "count": len(df),
        "predictions": [
            {
                "ds": _iso(r.ds),
                "yhat": round(float(r.yhat), 2),
                "yhat_lower": round(float(r.yhat_lower), 2),
                "yhat_upper": round(float(r.yhat_upper), 2),
            }
            for r in df.itertuples()
        ],
    }


@app.get(
    "/status/{symbol}",
    summary="Ultimo precio observado contra la banda del modelo",
)
def status(
    symbol: str = Path(description="Par de Binance, p. ej. BTCUSDT"),
) -> dict:
    symbol = _check_symbol(symbol)
    row = latest_vs_band(engine, symbol)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Todavia no hay precios con banda para {symbol}.",
        )

    dentro = row.yhat_lower <= row.close <= row.yhat_upper
    return {
        "symbol": symbol,
        "open_time": _iso(row.open_time),
        "price": round(float(row.close), 2),
        "yhat_lower": round(float(row.yhat_lower), 2),
        "yhat_upper": round(float(row.yhat_upper), 2),
        "dentro_de_banda": bool(dentro),
        "model_version": int(row.model_version),
        "interval_width": INTERVAL_WIDTH,
        "timezone": "GMT-3",
    }
