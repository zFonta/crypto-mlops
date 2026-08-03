"""Acceso a la base de datos de precios (Postgres, BD crypto)"""
import os
from datetime import datetime, timedelta, timezone

from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

from common.config import HISTORY_YEARS

load_dotenv(find_dotenv())


def get_crypto_db_uri() -> str:
    uri = os.environ.get("CRYPTO_DB_URI")
    if uri:
        return uri

    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("CRYPTO_DB_NAME", "crypto")

    if user and password:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    raise RuntimeError(
        "CRYPTO_DB_URI no definido. Establece CRYPTO_DB_URI en .env "
        "o define POSTGRES_USER/POSTGRES_PASSWORD/CRYPTO_DB_NAME."
    )


CRYPTO_DB_URI = get_crypto_db_uri()


def get_engine():
    return create_engine(CRYPTO_DB_URI, pool_pre_ping=True)


def upsert_klines (engine, symbol: str, klines: list) -> int:
    """Inserta velas de Binance (formato crudo del endpoint /klines). Idempotente""" 
    if not klines:
        return 0
    rows = [
        {
            "symbol": symbol,
            "open_time": datetime.fromtimestamp (k[0] / 1000, timezone.utc),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        }
        for k in klines
    ]
    stmt = text(
        """
        INSERT INTO prices (symbol, open_time, open, high, low, close, volume)
        VALUES (:symbol, :open_time, :open, :high, :low, :close, :volume)
        ON CONFLICT (symbol, open_time) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close = EXCLUDED.close, volume = EXCLUDED.volume
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)
    return len(rows)


def latest_open_time(engine, symbol: str):
    """Ultimo timestamp almacenado para un simbolo (o None)"""
    with engine.connect() as conn:
        res = conn.execute(
        text("SELECT MAX(open_time) FROM prices WHERE symbol = :s"), {"s": symbol}
        ).scalar()
    return res


def load_prices(engine, symbol: str, days: int | None = None) -> pd.DataFrame:
    """Historico horario del simbolo. Por defecto, toda la ventana rodante;
    el entrenamiento pide una ventana mas corta (TRAIN_DAYS)."""
    if days is None:
        days = 365 * HISTORY_YEARS
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = text(
        """
        SELECT open_time, close
        FROM prices
        WHERE symbol = :s AND open_time >= :since
        ORDER BY open_time
        """
    )   
    df = pd.read_sql(query, engine, params={"s": symbol, "since": since})
    return df


def load_ohlcv(engine, symbol: str, since) -> pd.DataFrame:
    """Velas completas (OHLCV) desde `since`. Lo usa el dashboard para el grafico
    de velas japonesas."""
    query = text(
        """
        SELECT open_time, open, high, low, close, volume
        FROM prices
        WHERE symbol = :s AND open_time >= :since
        ORDER BY open_time
        """
    )
    return pd.read_sql(query, engine, params={"s": symbol, "since": since})


def load_forecast(engine, symbol: str, hours: int) -> pd.DataFrame:
    """Banda del modelo en produccion desde la hora actual hacia adelante."""
    query = text(
        """
        SELECT ds, yhat, yhat_lower, yhat_upper, model_version
        FROM predictions
        WHERE symbol = :s
          AND ds >= date_trunc('hour', now())
          AND ds < date_trunc('hour', now()) + make_interval(hours => :h)
          AND model_version = (
              SELECT MAX(model_version) FROM predictions WHERE symbol = :s
          )
        ORDER BY ds
        """
    )
    return pd.read_sql(query, engine, params={"s": symbol, "h": hours})


def latest_vs_band(engine, symbol: str):
    """Ultima vela cerrada que tiene banda del modelo en produccion.

    Devuelve (open_time, close, yhat_lower, yhat_upper, model_version) o None.
    Es lo que necesita el monitor: si el precio actual quedo fuera, se reentrena.
    """
    query = text(
        """
        SELECT p.open_time, p.close, pr.yhat_lower, pr.yhat_upper, pr.model_version
        FROM prices p
        JOIN predictions pr
          ON pr.symbol = p.symbol AND pr.ds = p.open_time
        WHERE p.symbol = :s
          AND pr.model_version = (
              SELECT MAX(model_version) FROM predictions WHERE symbol = :s
          )
        ORDER BY p.open_time DESC
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        return conn.execute(query, {"s": symbol}).fetchone()


def prune_old(engine, years: int = HISTORY_YEARS) -> int:
    """Mantiene la ventana rodante eliminando datos mas viejos que `years`."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * years)
    with engine.begin() as conn:
        res = conn.execute(text("DELETE FROM prices WHERE open_time < :c"), {"c": cutoff})
    return res.rowcount


def save_predictions (engine, symbol: str, forecast: pd.DataFrame, model_version: int) -> int:
    """Persiste el forecast (columnas ds, yhat, yhat_lower, yhat_upper) en la tabla `predictions` """
    rows = [
        {
            "symbol": symbol,
            "ds": pd.Timestamp(r.ds).tz_localize("UTC").to_pydatetime(),
            "yhat": float(r.yhat),
            "yhat_lower": float(r.yhat_lower),
            "yhat_upper": float(r.yhat_upper),
            "model_version": int(model_version)
        }
        for r in forecast.itertuples()
    ]
    if not rows:
            return 0
    stmt = text(
        """
        INSERT INTO predictions (symbol, ds, yhat, yhat_lower, yhat_upper, model_version)
        VALUES (:symbol, :ds, :yhat, :yhat_lower, :yhat_upper, :model_version)
        ON CONFLICT (symbol, ds, model_version) DO UPDATE SET
            yhat = EXCLUDED.yhat, yhat_lower = EXCLUDED.yhat_lower, 
            yhat_upper = EXCLUDED.yhat_upper, predicted_at = NOW()
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)
    return len (rows)


def load_predictions(engine, symbol: str) -> pd.DataFrame:
    """Banda persistida del modelo en produccion (la version mas alta, que es la
    unica que sobrevive a `prune_predictions`)."""
    query = text(
        """
        SELECT ds, yhat, yhat_lower, yhat_upper, model_version
        FROM predictions
        WHERE symbol = :s
          AND model_version = (
              SELECT MAX(model_version) FROM predictions WHERE symbol = :s
          )
        ORDER BY ds
        """
    )
    return pd.read_sql(query, engine, params={"s": symbol})


def prune_predictions(engine, symbol: str, keep_version: int) -> int:
    """Borra las predicciones de versiones anteriores a `keep_version`.

    Cada promocion persiste miles de filas por simbolo y se promueve siempre;
    sin esto la tabla creceria sin limite. Solo queda vigente el forecast del
    modelo en produccion.
    """
    with engine.begin() as conn:
        res = conn.execute(
            text(
                "DELETE FROM predictions WHERE symbol = :s AND model_version < :v"
            ),
            {"s": symbol, "v": int(keep_version)},
        )
    return res.rowcount

