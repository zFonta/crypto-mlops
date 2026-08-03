"""DAG de ingesta de velas horarias de Binance.

Regla unica, por simbolo:
- si la tabla `prices` ya tiene datos -> arrancar en MAX(open_time) + 1 h
- si no hay datos                     -> arrancar HISTORY_YEARS anios atras

La paginacion del limite de 1000 velas por request la resuelve `fetch_klines`.
Corre a los 5 minutos de cada hora, cuando la vela anterior ya cerro.
"""
from datetime import datetime, timedelta, timezone

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from common.config import HISTORY_YEARS, LOCAL_TZ, SYMBOLS


def _local(ts) -> str:
    """Timestamp -> texto legible en la zona horaria del proyecto (GMT-3)."""
    return ts.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M GMT-3")


@dag(
    dag_id="crypto_ingest",
    schedule="5 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["crypto", "ingest"],
    doc_md=__doc__,
)
def ingest():

    @task
    def ingest_symbol(symbol: str) -> dict:
        from common.binance import fetch_klines
        from common.db import get_engine, latest_open_time, upsert_klines

        engine = get_engine()
        last = latest_open_time(engine, symbol)
        cold_start = last is None

        if cold_start:
            start = datetime.now(timezone.utc) - timedelta(days=365 * HISTORY_YEARS)
            print(f"{symbol}: tabla vacia, backfill desde {_local(start)}")
        else:
            start = last + timedelta(hours=1)
            print(f"{symbol}: incremental desde {_local(start)} "
                  f"(ultimo dato {_local(last)})")

        klines = fetch_klines(symbol, start_ms=int(start.timestamp() * 1000))
        rows = upsert_klines(engine, symbol, klines)
        print(f"{symbol}: {rows} velas horarias insertadas/actualizadas")

        return {"symbol": symbol, "rows": rows, "cold_start": cold_start}

    @task
    def prune_window():
        """Mantiene la ventana rodante de HISTORY_YEARS anios."""
        from common.db import get_engine, prune_old

        deleted = prune_old(get_engine(), HISTORY_YEARS)
        print(f"{deleted} filas fuera de la ventana rodante eliminadas")
        return deleted

    @task.short_circuit
    def needs_bootstrap(results: list[dict]) -> bool:
        """Solo en el arranque en frio hay que entrenar desde aca.

        En regimen normal el (re)entrenamiento lo gobiernan el schedule diario de
        `crypto_train_models` y el DAG `crypto_monitor_band`.
        """
        cold = [r["symbol"] for r in results if r.get("cold_start") and r.get("rows")]
        if cold:
            print(f"Primera carga de {cold} -> se dispara el entrenamiento inicial")
            return True
        print("Ingesta incremental: no se dispara entrenamiento")
        return False

    trigger_train = TriggerDagRunOperator(
        task_id="trigger_initial_training",
        trigger_dag_id="crypto_train_models",
        conf={"trigger_reason": "initial_ingest"},
        wait_for_completion=False,
    )

    ingested = ingest_symbol.expand(symbol=SYMBOLS)
    ingested >> prune_window()
    needs_bootstrap(ingested) >> trigger_train


ingest()
