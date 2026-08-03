-- Se ejecuta una sola vez al crear el volumen de Postgres.
-- La BD "airflow" ya la crea la imagen via POSTGRES DB.

CREATE DATABASE mlflow;
CREATE DATABASE crypto;

\connect crypto

CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    open_time TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
PRIMARY KEY (symbol, open_time)
);

CREATE INDEX IF NOT EXISTS idx_prices_symbol_time ON prices (symbol, open_time DESC);

-- Forecast persistido del modelo en produccion (se escribe tras cada promocion)

CREATE TABLE IF NOT EXISTS predictions (
    symbol TEXT NOT NULL,
    ds TIMESTAMPTZ NOT NULL,
    yhat DOUBLE PRECISION,
    yhat_lower DOUBLE PRECISION,
    yhat_upper DOUBLE PRECISION,
    model_version INT NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ds, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ds ON predictions (symbol, ds DESC);