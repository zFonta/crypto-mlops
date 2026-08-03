"""Dashboard Streamlit del pipeline crypto-mlops.

Muestra el historico en velas japonesas, el forecast del modelo en produccion y
la banda de prediccion del 95%. Las fechas se muestran en GMT-3. La
configuracion (simbolos, Postgres, MLflow) viene del modulo compartido
`common/`, el mismo que usan los DAGs.
"""
from datetime import datetime, timedelta, timezone

import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mlflow.tracking import MlflowClient

from common.config import (
    BACKCAST_DAYS,
    FORECAST_DAYS,
    LOCAL_TZ,
    PRODUCTION_ALIAS,
    SYMBOLS,
    model_name,
)
from common.db import get_engine as build_engine, load_ohlcv, load_predictions

st.set_page_config(page_title="Crypto MLOps", page_icon="📈", layout="wide")

#--- Paleta ----
# Un unico tono (indigo) para todo lo que produce el modelo: banda y forecast son
# la misma entidad, lo que cambia es el regimen (ajuste vs futuro) y eso lo marca
# el trazo. Verde/rojo quedan para la direccion de la vela.
MODEL_HUE = "#636EFA"
MODEL_HUE_SOFT = "rgba(99, 110, 250, 0.55)"
BAND_HIST = "rgba(99, 110, 250, 0.13)"
BAND_FUTURE = "rgba(99, 110, 250, 0.26)"
# paleta verificada: separacion OKLab >= 8 en los tres tipos de daltonismo y
# contraste >= 3:1 contra el fondo claro y el oscuro de Streamlit
CANDLE_UP = "#089981"
CANDLE_DOWN = "#EF5350"
GRID = "rgba(128, 128, 128, 0.18)"

RESAMPLE_RULES = {"1 hora": "1h", "4 horas": "4h", "1 dia": "1D"}

# El grafico no muestra mas historia que la que cubre la banda: una vela sin
# intervalo alrededor no dice nada sobre el modelo.
MAX_HISTORY_DAYS = BACKCAST_DAYS


#--- Acceso a datos ----
def to_local(s: pd.Series) -> pd.Series:
    """UTC de la base -> hora local (GMT-3) sin timezone, para graficar."""
    return pd.to_datetime(s, utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)


@st.cache_resource
def get_engine():
    return build_engine()


@st.cache_data(ttl=300)
def get_candles(symbol: str, days: int) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    df = load_ohlcv(get_engine(), symbol, since)
    if not df.empty:
        df["open_time"] = to_local(df["open_time"])
    return df


@st.cache_data(ttl=300)
def get_predictions(symbol: str) -> pd.DataFrame:
    df = load_predictions(get_engine(), symbol)
    if not df.empty:
        df["ds"] = to_local(df["ds"])
    return df


@st.cache_resource(ttl=600)
def load_production_model(symbol: str):
    return mlflow.prophet.load_model(f"models:/{model_name(symbol)}@{PRODUCTION_ALIAS}")


def fmt_ms(ms) -> str:
    """Epoch en milisegundos -> fecha legible en GMT-3."""
    if not ms:
        return "-"
    try:
        return datetime.fromtimestamp(ms / 1000, LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "-"


@st.cache_data(ttl=300)
def production_version_info(symbol: str) -> dict | None:
    """Version con alias `production` + metricas de su run."""
    client = MlflowClient()
    try:
        v = client.get_model_version_by_alias(model_name(symbol), PRODUCTION_ALIAS)
        run = client.get_run(v.run_id)
        return {
            "version": int(v.version),
            "creado": fmt_ms(v.creation_timestamp or run.info.start_time),
            "mape": run.data.metrics.get("mape"),
            "rmse": run.data.metrics.get("rmse"),
            "trigger": run.data.params.get("trigger_reason", ""),
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def model_inventory(symbol: str) -> pd.DataFrame:
    client = MlflowClient()
    name = model_name(symbol)
    rows = []
    for v in client.search_model_versions(f"name='{name}'"):
        run_metrics, run_params = {}, {}
        try:
            run = client.get_run(v.run_id)
            run_metrics = run.data.metrics
            run_params = run.data.params
        except Exception:
            pass
        rows.append(
            {
                "version": int(v.version),
                "creado": fmt_ms(v.creation_timestamp),
                "alias": ", ".join(v.aliases or []),
                "MAPE (%)": round(run_metrics.get("mape", float("nan")), 3),
                "RMSE": round(run_metrics.get("rmse", float("nan")), 2),
                "ancho banda (%)": round(
                    run_metrics.get("band_width_pct", float("nan")), 2
                ),
                "cobertura (%)": round(run_metrics.get("coverage", float("nan")), 1),
                "motivo": run_params.get("trigger_reason", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("version", ascending=False)


#--- Transformaciones ----
def auto_rule(days: int) -> str:
    """Granularidad por defecto segun el rango: velas legibles, no 8.760 barras."""
    if days <= 14:
        return "1h"
    if days <= 90:
        return "4h"
    return "1D"


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "1h" or df.empty:
        return df
    return (
        df.set_index("open_time")
        .resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open"])
        .reset_index()
    )


def join_real_vs_pred(prices: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Cruza precio real y banda hora a hora (solo donde hay ambos)."""
    if prices.empty or forecast is None or forecast.empty:
        return pd.DataFrame()
    merged = prices.merge(forecast, left_on="open_time", right_on="ds", how="inner")
    if merged.empty:
        return merged
    merged["error"] = merged["close"] - merged["yhat"]
    merged["dentro"] = (merged["close"] >= merged["yhat_lower"]) & (
        merged["close"] <= merged["yhat_upper"]
    )
    return merged


#--- Graficos ----
def _add_band(fig, df: pd.DataFrame, fillcolor: str, name: str):
    fig.add_trace(
        go.Scatter(
            x=df["ds"], y=df["yhat_upper"], mode="lines", line={"width": 0},
            showlegend=False, hoverinfo="skip", legendgroup=name,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["ds"], y=df["yhat_lower"], mode="lines", line={"width": 0},
            fill="tonexty", fillcolor=fillcolor, name=name, legendgroup=name,
            hoverinfo="skip",
        )
    )


#  cuanto puede excederse la banda por fuera del rango de las velas antes de
#  recortarse, como fraccion de ese rango. Sin este tope el cono del forecast
#  manda sobre el autoscale y las velas quedan aplastadas.
BAND_OVERSHOOT = 0.25


def _y_range(candles: pd.DataFrame, *bands: pd.DataFrame) -> list | None:
    """Rango del eje Y anclado a las velas, con la banda acotada."""
    if candles.empty:
        return None
    low, high = float(candles["low"].min()), float(candles["high"].max())
    span = high - low or high * 0.01
    lo, hi = low, high
    for band in bands:
        if band is not None and not band.empty:
            lo = min(lo, float(band["yhat_lower"].min()))
            hi = max(hi, float(band["yhat_upper"].max()))
    lo = max(lo, low - span * BAND_OVERSHOOT)
    hi = min(hi, high + span * BAND_OVERSHOOT)
    margin = (hi - lo) * 0.03
    return [lo - margin, hi + margin]


def price_figure(candles: pd.DataFrame, forecast: pd.DataFrame | None,
                 last_time: pd.Timestamp) -> go.Figure:
    fig = go.Figure()

    empty = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    if forecast is None or forecast.empty:
        hist = future = empty
    else:
        # el punto frontera entra en las dos mitades para que las bandas empalmen
        hist = forecast[forecast["ds"] <= last_time]
        future = forecast[forecast["ds"] >= last_time]

    if not hist.empty:
        _add_band(fig, hist, BAND_HIST, "Banda 95% (histórico)")
    if not future.empty:
        _add_band(fig, future, BAND_FUTURE, "Banda 95% (forecast)")

    fig.add_trace(
        go.Candlestick(
            x=candles["open_time"],
            open=candles["open"], high=candles["high"],
            low=candles["low"], close=candles["close"],
            name="Precio (OHLC)",
            increasing_line_color=CANDLE_UP, increasing_fillcolor=CANDLE_UP,
            decreasing_line_color=CANDLE_DOWN, decreasing_fillcolor=CANDLE_DOWN,
            line={"width": 1},
        )
    )

    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist["ds"], y=hist["yhat"], mode="lines",
                name="Ajuste del modelo",
                line={"color": MODEL_HUE_SOFT, "width": 1.5, "dash": "dash"},
            )
        )
    if not future.empty:
        fig.add_trace(
            go.Scatter(
                x=future["ds"], y=future["yhat"], mode="lines", name="Forecast",
                line={"color": MODEL_HUE, "width": 2.5, "dash": "dot"},
            )
        )

    # la anotacion va aparte: add_vline(annotation_text=...) promedia las
    # coordenadas x y revienta con timestamps (plotly 5.24 + pandas 2.x)
    fig.add_vline(x=last_time, line_dash="dot", line_color="rgba(128,128,128,0.9)")
    fig.add_annotation(
        x=last_time, y=1, yref="paper", yanchor="bottom", text="ahora",
        showarrow=False, font={"size": 11, "color": "rgba(128,128,128,0.9)"},
    )
    fig.update_layout(
        height=620,
        hovermode="x unified",
        margin={"t": 40, "r": 10, "b": 10, "l": 10},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"title": "Fecha (GMT-3)", "rangeslider": {"visible": False},
               "showgrid": False},
        yaxis={
            "title": "Precio (USDT)",
            "gridcolor": GRID,
            "zeroline": False,
            "range": _y_range(candles, hist, future),
        },
    )
    return fig


def error_figure(merged: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["open_time"], y=merged["error"], mode="lines",
            name="Precio real - yhat", line={"color": MODEL_HUE, "width": 1.5},
            fill="tozeroy", fillcolor=BAND_HIST,
        )
    )
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.6)", line_width=1)
    fig.update_layout(
        height=280,
        hovermode="x unified",
        margin={"t": 20, "r": 10, "b": 10, "l": 10},
        showlegend=False,
        xaxis={"showgrid": False},
        yaxis={"title": "Error (USDT)", "gridcolor": GRID, "zeroline": False},
    )
    return fig


#--- Sidebar ----
st.sidebar.title("Crypto MLOps")
symbol = st.sidebar.selectbox("Criptomoneda", SYMBOLS)
days = st.sidebar.slider("Dias de historico", 7, MAX_HISTORY_DAYS, 90)
forecast_days = st.sidebar.slider("Dias de forecast", 1, FORECAST_DAYS, 3)
granularity = st.sidebar.selectbox(
    "Granularidad de las velas", ["Auto"] + list(RESAMPLE_RULES)
)
forecast_hours = forecast_days * 24
rule = auto_rule(days) if granularity == "Auto" else RESAMPLE_RULES[granularity]
st.sidebar.caption(
    f"Velas de {rule}; el forecast siempre es horario. Horas en GMT-3.\n\n"
    "Airflow + MLflow + MinIO + Postgres + Prophet. Datos de Binance."
)

st.title(f"Forecast horario - {symbol}")

#--- Datos ----
prices = get_candles(symbol, days)
if prices.empty:
    st.warning(
        "No hay datos en Postgres todavia. Ejecuta el DAG 'crypto_ingest' en Airflow."
    )
    st.stop()

last_time = prices["open_time"].max()
last_price = float(prices.loc[prices["open_time"] == last_time, "close"].iloc[0])

prev_row = prices[prices["open_time"] <= last_time - timedelta(hours=24)]
delta_24h = (
    (last_price / float(prev_row["close"].iloc[-1]) - 1) * 100
    if not prev_row.empty
    else None
)

#--- Forecast del modelo en produccion ----
forecast = None
forecast_source = None

stored = get_predictions(symbol)
if not stored.empty:
    forecast = stored[
        (stored["ds"] >= prices["open_time"].min())
        & (stored["ds"] <= last_time + timedelta(hours=forecast_hours))
    ].reset_index(drop=True)
    forecast_source = f"modelo v{int(stored['model_version'].iloc[0])}"
else:
    # Fallback: predecir al vuelo con el modelo en produccion. Prophet trabaja en
    # UTC, asi que se le pide el horizonte en UTC y se reetiqueta a GMT-3.
    try:
        model = load_production_model(symbol)
        horizon = pd.date_range(
            prices["open_time"].min(),
            last_time + timedelta(hours=forecast_hours),
            freq="h",
        )
        forecast = model.predict(
            pd.DataFrame({"ds": horizon - pd.Timedelta(LOCAL_TZ.utcoffset(None))})
        )
        forecast["ds"] = horizon
        forecast_source = "calculado al vuelo"
    except Exception:
        st.info(
            "Aun no hay modelo en produccion para este simbolo. "
            "Ejecuta el DAG `crypto_train_models` en Airflow."
        )

# Solo se grafican velas que tengan banda alrededor.
prices_vis = prices
candles = resample_ohlc(prices_vis, rule)
if forecast is not None and not forecast.empty:
    banda_desde = forecast["ds"].min()
    prices_vis = prices[prices["open_time"] >= banda_desde]
    # el recorte va DESPUES del resample tambien: el bucket de 4h o 1D arranca
    # en su borde inferior, que puede caer antes del inicio de la banda
    candles = resample_ohlc(prices_vis, rule)
    candles = candles[candles["open_time"] >= banda_desde].reset_index(drop=True)

merged = join_real_vs_pred(prices_vis, forecast)
coverage = float(merged["dentro"].mean() * 100) if not merged.empty else None
prod = production_version_info(symbol)

#--- KPIs ----
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Ultimo precio (USDT)",
    f"{last_price:,.2f}",
    f"{delta_24h:+.2f}% (24h)" if delta_24h is not None else None,
)
k2.metric("Ultima vela (GMT-3)", last_time.strftime("%Y-%m-%d %H:%M"))
k3.metric("Modelo en produccion", f"v{prod['version']}" if prod else "-")
k4.metric(
    "MAPE del modelo",
    f"{prod['mape']:.2f} %" if prod and prod.get("mape") is not None else "-",
    help="Error porcentual medio del modelo en produccion sobre su ventana de "
         "entrenamiento. Es informativo: no decide la promocion.",
)
k5.metric(
    "Cobertura de la banda",
    f"{coverage:.1f} %" if coverage is not None else "-",
    help="Porcentaje de horas del historico visible en las que el precio real "
         "cayo dentro de la banda del 95%.",
)

#--- Grafico principal ----
# Deliberadamente FUERA de st.tabs: un plotly_chart con use_container_width
# dentro de una pestania se mide con ancho 0 en el primer render.
if forecast is not None and not forecast.empty:
    row_now = forecast.loc[forecast["ds"] == last_time]
    if not row_now.empty:
        lo = float(row_now["yhat_lower"].iloc[0])
        hi = float(row_now["yhat_upper"].iloc[0])
        if lo <= last_price <= hi:
            st.success(f"Precio dentro de la banda 95%: [{lo:,.2f} - {hi:,.2f}]")
        else:
            st.error(
                f"Precio FUERA de la banda 95%: [{lo:,.2f} - {hi:,.2f}]. "
                "El monitor de Airflow dispara un reentrenamiento."
            )

st.plotly_chart(price_figure(candles, forecast, last_time), use_container_width=True)
if forecast_source is not None:
    st.caption(
        f"Velas de {rule} sobre {days} dias, forecast a {forecast_days} dias. "
        f"Fuente: {forecast_source}."
    )

with st.expander("Diagnostico del modelo"):
    if merged.empty:
        st.info("Todavia no hay solape entre precios y predicciones.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Horas comparadas", f"{len(merged):,}")
        c2.metric("Cobertura de la banda", f"{coverage:.1f} %")
        c3.metric("Error absoluto medio", f"{merged['error'].abs().mean():,.2f} USDT")
        st.plotly_chart(error_figure(merged), use_container_width=True)
        st.caption(
            "Diferencia entre el precio real y el centro de la banda, hora a hora."
        )
        fuera = merged.loc[~merged["dentro"], ["open_time", "close", "yhat_lower", "yhat_upper"]]
        if not fuera.empty:
            st.markdown(f"**Horas fuera de banda** ({len(fuera)} en total)")
            st.dataframe(fuera.tail(50), use_container_width=True, hide_index=True)

with st.expander("Inventario de modelos (MLflow Model Registry)"):
    st.caption(
        "Cada reentrenamiento registra una version nueva y le pasa el alias "
        "`production`. Se reentrena cada 24 h o al salirse de la banda."
    )
    try:
        inv = model_inventory(symbol)
        if inv.empty:
            st.info("Sin versiones registradas todavia.")
        else:
            st.dataframe(inv, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"No se pudo consultar MLflow: {exc}")

st.caption("MLflow UI: /mlflow/ | Airflow UI: /airflow/ | MinIO Console: /minio/")
