from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import dag_monitor_band
from common.config import RETRAIN_COOLDOWN_HOURS


def _get_callable(task):
    """Extrae la funcion original de una task de TaskFlow, sea simple o mapeada
    (`.expand(...)`, donde el callable puede vivir en `partial_kwargs` en vez de
    ser un atributo directo segun la version de Airflow)."""
    if hasattr(task, "python_callable"):
        return task.python_callable
    partial_kwargs = getattr(task, "partial_kwargs", None) or {}
    if "python_callable" in partial_kwargs:
        return partial_kwargs["python_callable"]
    raise AttributeError(f"No pude extraer el callable de {task!r}")


@pytest.fixture
def dag():
    return dag_monitor_band.monitor_band()


@pytest.fixture
def check_symbol(dag):
    return _get_callable(dag.get_task("check_symbol"))


@pytest.fixture
def breached_symbols(dag):
    return _get_callable(dag.get_task("breached_symbols"))


@pytest.fixture
def has_breach(dag):
    return _get_callable(dag.get_task("has_breach"))


@pytest.fixture
def mocked_mlflow_and_db():
    with patch("mlflow.tracking.MlflowClient") as mock_client_cls, \
         patch("common.db.get_engine") as mock_get_engine, \
         patch("common.db.latest_vs_band") as mock_latest_vs_band:
        yield mock_client_cls, mock_get_engine, mock_latest_vs_band


def _version(age_hours: float):
    created = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return SimpleNamespace(version="5", creation_timestamp=int(created.timestamp() * 1000))


def _row(close: float, lower: float, upper: float):
    return SimpleNamespace(
        open_time=datetime.now(timezone.utc),
        close=close,
        yhat_lower=lower,
        yhat_upper=upper,
        model_version=5,
    )


def test_no_production_model_means_no_breach(check_symbol, mocked_mlflow_and_db):
    mock_client_cls, _, mock_latest = mocked_mlflow_and_db
    mock_client_cls.return_value.get_model_version_by_alias.side_effect = Exception("no model")

    result = check_symbol("BTCUSDT")

    assert result == {"symbol": "BTCUSDT", "breach": False}
    mock_latest.assert_not_called()


def test_cooldown_skips_band_check(check_symbol, mocked_mlflow_and_db):
    mock_client_cls, _, mock_latest = mocked_mlflow_and_db
    mock_client_cls.return_value.get_model_version_by_alias.return_value = _version(
        age_hours=RETRAIN_COOLDOWN_HOURS / 2
    )

    result = check_symbol("BTCUSDT")

    assert result == {"symbol": "BTCUSDT", "breach": False}
    mock_latest.assert_not_called()


def test_no_persisted_band_means_no_breach(check_symbol, mocked_mlflow_and_db):
    mock_client_cls, _, mock_latest = mocked_mlflow_and_db
    mock_client_cls.return_value.get_model_version_by_alias.return_value = _version(
        age_hours=RETRAIN_COOLDOWN_HOURS * 5
    )
    mock_latest.return_value = None

    result = check_symbol("BTCUSDT")

    assert result == {"symbol": "BTCUSDT", "breach": False}


@pytest.mark.parametrize(
    "close, lower, upper, expected_breach",
    [
        (100.0, 90.0, 110.0, False),  # dentro de la banda
        (89.9, 90.0, 110.0, True),  # justo por debajo
        (110.1, 90.0, 110.0, True),  # justo por arriba
        (90.0, 90.0, 110.0, False),  # borde inferior exacto: comparacion estricta, no rompe
        (110.0, 90.0, 110.0, False),  # borde superior exacto: idem
    ],
)
def test_breach_comparison(
    check_symbol, mocked_mlflow_and_db, close, lower, upper, expected_breach
):
    mock_client_cls, _, mock_latest = mocked_mlflow_and_db
    mock_client_cls.return_value.get_model_version_by_alias.return_value = _version(
        age_hours=RETRAIN_COOLDOWN_HOURS * 5
    )
    mock_latest.return_value = _row(close, lower, upper)

    result = check_symbol("BTCUSDT")

    assert result["breach"] is expected_breach


def test_breached_symbols_filters_by_breach_flag(breached_symbols):
    results = [
        {"symbol": "BTCUSDT", "breach": True},
        {"symbol": "ETHUSDT", "breach": False},
        {"symbol": "SOLUSDT", "breach": True},
    ]
    assert breached_symbols(results) == ["BTCUSDT", "SOLUSDT"]


def test_has_breach(has_breach):
    assert has_breach([]) is False
    assert has_breach(["BTCUSDT"]) is True
