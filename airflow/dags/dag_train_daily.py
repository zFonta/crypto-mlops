"""DAG de (re)entrenamiento: entrena un Prophet por cripto, lo registra en MLflow
y lo promueve SIEMPRE al alias `production`.

Dos disparadores automaticos:
- cada 24 h por schedule (03:00 GMT-3)
- `crypto_monitor_band`, cuando el precio se sale de la banda [yhat_lower, yhat_upper]

Ademas `crypto_ingest` lo dispara una unica vez, en el arranque en frio.

`dag_run.conf` acepta:
- `symbols`: lista de simbolos a reentrenar (por defecto, todos)
- `trigger_reason`: motivo, se guarda como parametro del run y como tag de la version
"""
from datetime import datetime

from airflow.decorators import dag, task

from common.config import SYMBOLS


@dag(
    dag_id="crypto_train_models",
    schedule="0 3 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1, # entrena de a un modelo por vez para no saturar la CPU
    tags=["crypto", "training", "mlflow"],
    doc_md=__doc__,
)
def train_models():

    @task
    def symbols_to_train(**context) -> list[str]:
        """Simbolos pedidos en el conf (el monitor manda solo los que rompieron
        la banda); si no viene nada, se reentrenan todos."""
        conf = (context["dag_run"].conf or {}) if context.get("dag_run") else {}
        requested = conf.get("symbols") or SYMBOLS
        valid = [s for s in requested if s in SYMBOLS]
        if not valid:
            print(f"conf.symbols={requested} no coincide con {SYMBOLS}; se usan todos")
            valid = list(SYMBOLS)
        print(f"Se entrenan: {valid}")
        return valid

    @task
    def train_symbol(symbol: str, **context) -> dict:
        from common.training import train_and_register

        conf = (context["dag_run"].conf or {}) if context.get("dag_run") else {}
        trigger_reason = conf.get("trigger_reason", "scheduled")

        result = train_and_register(symbol, trigger_reason=trigger_reason)
        print(result)
        return result

    @task
    def summary(results: list[dict]):
        print("Resumen de entrenamiento:")
        for r in results:
            print(
                f" {r['symbol']}: v{r['version']} promovido a produccion "
                f"(motivo={r['trigger_reason']}, MAPE={r['mape']:.3f}%, "
                f"banda={r['band_width_pct']:.2f}% del precio, "
                f"cobertura={r['coverage']:.1f}%)"
            )

    summary(train_symbol.expand(symbol=symbols_to_train()))


train_models()
