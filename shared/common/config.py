"""Configuracion compartida por los DAGs y el dashboard."""
import os
from datetime import timedelta, timezone

# Zona horaria del proyecto: GMT-3 (Argentina, sin horario de verano).
# Airflow la usa para los schedules y la UI; el dashboard, para mostrar fechas.
TIMEZONE_NAME = "America/Argentina/Buenos_Aires"
LOCAL_TZ = timezone(timedelta(hours=-3))

# Pares a seguir en Binance
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Ventana rodante de historico (anios). Es cuanto se GUARDA en la BD.
HISTORY_YEARS = int(os.environ.get("HISTORY_YEARS", "5"))

# Ventana de ENTRENAMIENTO (dias): un anio de velas horarias.
TRAIN_DAYS = int(os.environ.get("TRAIN_DAYS", "365"))

# Ponderacion por antiguedad. Dentro del anio de entrenamiento las horas
# recientes pesan mas que las viejas: el peso arranca en MAX_SAMPLE_WEIGHT para
# la ultima hora y se divide a la mitad cada WEIGHT_HALF_LIFE_DAYS.
# Prophet no acepta pesos por observacion, asi que se implementan repitiendo
# filas: repetir una fila N veces equivale a darle peso N en la verosimilitud.
WEIGHT_HALF_LIFE_DAYS = int(os.environ.get("WEIGHT_HALF_LIFE_DAYS", "120"))
MAX_SAMPLE_WEIGHT = int(os.environ.get("MAX_SAMPLE_WEIGHT", "3"))

# Hiperparametros de Prophet. Se loguean tal cual como parametros del run.
PROPHET_PARAMS = {
    "changepoint_prior_scale": 0.05,
    # El default (0.8) deja el ultimo 20% de la serie sin changepoints, o sea el
    # trend congelado justo en la zona que mira la banda de control.
    "changepoint_range": 0.9,
    "daily_seasonality": True,
    "weekly_seasonality": True,
    # No hay senial anual util en cripto a escala horaria.
    "yearly_seasonality": False,
    # La regla del proyecto habla de una banda de control del 95%.
    "interval_width": 0.95,
}

# Horizonte del forecast persistido en la tabla `predictions` (hacia adelante).
FORECAST_DAYS = int(os.environ.get("FORECAST_DAYS", "7"))
FORECAST_HOURS = FORECAST_DAYS * 24

# Banda persistida hacia atras: permite dibujar el intervalo sobre la historia
# en el dashboard. Nunca mas que la ventana de entrenamiento.
BACKCAST_DAYS = min(int(os.environ.get("BACKCAST_DAYS", "90")), TRAIN_DAYS)
BACKCAST_HOURS = BACKCAST_DAYS * 24

# Tiempo minimo entre reentrenamientos disparados por salida de banda (horas).
RETRAIN_COOLDOWN_HOURS = int(os.environ.get("RETRAIN_COOLDOWN_HOURS", "3"))

# Nombre del experimento y prefijo de los modelos registrados en MLflow
MLFLOW_EXPERIMENT = "crypto_forecasting"
MODEL_PREFIX = "crypto_prophet"
PRODUCTION_ALIAS = "production"


def model_name(symbol: str) -> str:
    return f"{MODEL_PREFIX}_{symbol}"
