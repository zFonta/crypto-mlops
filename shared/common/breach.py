"""
Decisión de reentrenamiento por salida de banda.

Esta es la lógica que hoy vive dentro de `dag_monitor_band.py` mezclada con la
lectura de Postgres y el TriggerDagRunOperator. Extraída acá:

  - no toca la base, no toca Airflow, no llama a datetime.now()
  - `ahora` y `ultimo_retrain` se inyectan como parámetros

El DAG queda como una cáscara:

    obs = leer_ultima_vela_y_banda(symbol)          # I/O
    d = evaluar_breach(symbol, **obs, ahora=..., ultimo_retrain=...)
    if d.reentrenar: ...                            # I/O

Reglas fijadas (ver tests/test_breach.py):
  - la banda es CERRADA: precio == yhat_upper está DENTRO
  - sin banda disponible nunca es breach (un stack recién levantado no debe
    entrar en loop de reentrenamiento)
  - el cooldown se evalúa después del breach, y `>=` cooldown ya habilita
  - todos los datetimes deben ser tz-aware
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

# Motivos
DENTRO = "dentro_de_banda"
FUERA_ARRIBA = "fuera_arriba"
FUERA_ABAJO = "fuera_abajo"
SIN_BANDA = "sin_banda"
EN_COOLDOWN = "en_cooldown"


@dataclass(frozen=True)
class Decision:
    symbol: str
    breach: bool
    reentrenar: bool
    motivo: str

    @property
    def trigger_reason(self) -> Optional[str]:
        """Lo que se guarda como parámetro del run y tag de la versión en MLflow."""
        return f"breach:{self.motivo}" if self.reentrenar else None


def _exigir_aware(nombre: str, dt: Optional[datetime]) -> None:
    if dt is not None and (dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None):
        raise ValueError(
            f"{nombre} tiene que ser tz-aware: los timestamps salen de Postgres "
            f"como TIMESTAMPTZ y compararlos con un naive tira TypeError."
        )


def evaluar_breach(
    symbol: str,
    precio: Optional[float],
    yhat_lower: Optional[float],
    yhat_upper: Optional[float],
    ahora: datetime,
    ultimo_retrain: Optional[datetime] = None,
    cooldown_horas: float = 3,
) -> Decision:
    _exigir_aware("ahora", ahora)
    _exigir_aware("ultimo_retrain", ultimo_retrain)

    if precio is None or yhat_lower is None or yhat_upper is None:
        return Decision(symbol, breach=False, reentrenar=False, motivo=SIN_BANDA)

    if precio > yhat_upper:
        motivo = FUERA_ARRIBA
    elif precio < yhat_lower:
        motivo = FUERA_ABAJO
    else:
        return Decision(symbol, breach=False, reentrenar=False, motivo=DENTRO)

    if ultimo_retrain is not None:
        transcurrido = ahora - ultimo_retrain
        if transcurrido < timedelta(hours=cooldown_horas):
            return Decision(symbol, breach=True, reentrenar=False, motivo=EN_COOLDOWN)

    return Decision(symbol, breach=True, reentrenar=True, motivo=motivo)


def simbolos_a_reentrenar(decisiones: Iterable[Decision]) -> list[str]:
    """
    Lo que va en `dag_run.conf["symbols"]`. Solo los que rompieron y no están en
    cooldown — reentrenar los tres siempre es caro y ensucia el Model Registry.
    """
    return sorted({d.symbol for d in decisiones if d.reentrenar})
