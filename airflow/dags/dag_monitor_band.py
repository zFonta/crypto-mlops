"""DAG de monitoreo: cada hora compara el precio de la ultima vela cerrada con
la banda [yhat_lower, yhat_upper] del modelo en produccion. Si quedo fuera,
dispara el reentrenamiento *solo de los simbolos afectados*.

Junto con el schedule diario de `crypto_train_models` cubre la regla del
proyecto: se reentrena cada 24 h o cuando el activo se va fuera de la banda.
"""
from datetime import datetime, timezone

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from common.config import (
    LOCAL_TZ,
    PRODUCTION_ALIAS,
    RETRAIN_COOLDOWN_HOURS,
    SYMBOLS,
    model_name,
)


@dag(
    dag_id="crypto_monitor_band",
    schedule="20 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    # necesario para que el XCom con la lista de simbolos llegue al conf como
    # lista de Python y no como el string "['BTCUSDT']"
    render_template_as_native_obj=True,
    tags=["crypto", "monitoring"],
    doc_md=__doc__,
)
def monitor_band():

    @task
    def check_symbol(symbol: str) -> dict:
        from mlflow.tracking import MlflowClient

        from common.db import get_engine, latest_vs_band

        # Cooldown: si el modelo en produccion es muy reciente, no reentrenar.
        name = model_name(symbol)
        try:
            version = MlflowClient().get_model_version_by_alias(name, PRODUCTION_ALIAS)
            created = datetime.fromtimestamp(
                version.creation_timestamp / 1000, timezone.utc
            )
            age_h = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        except Exception as exc:
            print(f"{symbol}: no hay modelo en produccion todavia ({exc})")
            return {"symbol": symbol, "breach": False}

        if age_h < RETRAIN_COOLDOWN_HOURS:
            print(
                f"{symbol}: modelo v{version.version} tiene {age_h:.1f} h "
                f"(< {RETRAIN_COOLDOWN_HOURS} h), en cooldown"
            )
            return {"symbol": symbol, "breach": False}

        row = latest_vs_band(get_engine(), symbol)
        if row is None:
            print(f"{symbol}: la banda persistida no cubre ninguna vela todavia")
            return {"symbol": symbol, "breach": False}

        breach = row.close < row.yhat_lower or row.close > row.yhat_upper
        estado = "FUERA de banda" if breach else "dentro de banda"
        vela = row.open_time.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M GMT-3")
        print(
            f"{symbol}: vela {vela}, precio {row.close:,.2f} {estado} "
            f"[{row.yhat_lower:,.2f} - {row.yhat_upper:,.2f}]"
        )
        return {"symbol": symbol, "breach": bool(breach), "precio": float(row.close)}

    @task
    def breached_symbols(results: list[dict]) -> list[str]:
        breached = [r["symbol"] for r in results if r.get("breach")]
        if breached:
            print(f"Fuera de banda: {breached} -> se reentrenan solo estos")
        else:
            print("Todos los precios dentro de la banda de prediccion")
        return breached

    @task.short_circuit
    def has_breach(symbols: list[str]) -> bool:
        return bool(symbols)

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="crypto_train_models",
        conf={
            "symbols": "{{ ti.xcom_pull(task_ids='breached_symbols') }}",
            "trigger_reason": "band_breach",
        },
        wait_for_completion=False,
    )

    breached = breached_symbols(check_symbol.expand(symbol=SYMBOLS))
    has_breach(breached) >> trigger_retrain


monitor_band()
