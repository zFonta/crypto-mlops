#--------------------------------------------------#
# Test para lógica de paginación de fetch_klines() #
#--------------------------------------------------#


from unittest.mock import MagicMock
from datetime import datetime, timezone
import time_machine
import requests
import pytest

from common import binance


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr(binance.time, "sleep", lambda _seconds: None)


def _kline(open_time_ms: int, close_time_ms: int) -> list:
    """Kline crudo de Binance: solo importan open_time (0) y close_time (6)."""
    return [open_time_ms, "0", "0", "0", "0", "0", close_time_ms, "0", "0", "0", "0", "0"]


def _resp(batch: list) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = batch
    resp.raise_for_status.return_value = None
    return resp


def _install_fake_session(monkeypatch, batches: list[list]) -> MagicMock:
    session = MagicMock()
    session.get.side_effect = [_resp(b) for b in batches]
    monkeypatch.setattr(binance.requests, "Session", lambda: session)
    return session


def test_pages_until_partial_batch_and_advances_cursor(monkeypatch):
    monkeypatch.setattr(binance, "MAX_LIMIT", 2)
    batch1 = [_kline(1000, 1000), _kline(2000, 2000)]
    batch2 = [_kline(3000, 3000), _kline(4000, 4000)]
    batch3 = [_kline(5000, 5000)]  # menos que MAX_LIMIT -> corta el loop
    session = _install_fake_session(monkeypatch, [batch1, batch2, batch3])

    result = binance.fetch_klines("BTCUSDT", start_ms=1000, end_ms=10**15)

    assert len(result) == 5
    assert session.get.call_count == 3
    calls = session.get.call_args_list
    assert calls[0].kwargs["params"]["startTime"] == 1000
    assert calls[1].kwargs["params"]["startTime"] == batch1[-1][0] + 1
    assert calls[2].kwargs["params"]["startTime"] == batch2[-1][0] + 1


def test_empty_first_batch_returns_immediately(monkeypatch):
    session = _install_fake_session(monkeypatch, [[]])

    result = binance.fetch_klines("BTCUSDT", start_ms=1000, end_ms=10**15)

    assert result == []
    assert session.get.call_count == 1


def test_still_open_candle_is_dropped(monkeypatch):
    closed = _kline(1000, 1000)
    still_open = _kline(2000, 10**15)  # close_time muy en el futuro
    _install_fake_session(monkeypatch, [[closed, still_open]])

    result = binance.fetch_klines("BTCUSDT", start_ms=1000, end_ms=10**16)

    assert result == [closed]


def test_cursor_past_end_ms_never_requests(monkeypatch):
    session = _install_fake_session(monkeypatch, [])

    result = binance.fetch_klines("BTCUSDT", start_ms=5000, end_ms=5000)

    assert result == []
    session.get.assert_not_called()

# ---------------------------------------------------------------------------
# Complementos: error HTTP, vela abierta con reloj congelado, y params enviados
# ---------------------------------------------------------------------------

def _failing_resp(status_exc: Exception) -> MagicMock:
    """Response cuyo raise_for_status() lanza, como haria un 429/503 real."""
    resp = MagicMock()
    resp.raise_for_status.side_effect = status_exc
    return resp


def test_http_error_is_propagated(monkeypatch):
    """Un 429/503 debe cortar con excepcion, nunca devolver lista parcial en
    silencio (un retorno parcial deja huecos invisibles en `prices`)."""
    session = MagicMock()
    session.get.side_effect = [_failing_resp(requests.HTTPError("429"))]
    monkeypatch.setattr(binance.requests, "Session", lambda: session)

    with pytest.raises(requests.HTTPError):
        binance.fetch_klines("BTCUSDT", start_ms=1000, end_ms=10**15)


def test_open_candle_dropped_against_frozen_now(monkeypatch):
    """El filtro final es `close_time < now`. Con el reloj congelado el borde es
    determinístico: la vela que cierra justo en `now` sigue abierta y se descarta;
    la que cerro 1 ms antes se conserva."""
    frozen = datetime(2026, 3, 1, 14, 20, tzinfo=timezone.utc)
    now_ms = int(frozen.timestamp() * 1000)

    just_closed = _kline(now_ms - 3_600_000, now_ms - 1)   # cerro 1 ms antes de now
    still_open = _kline(now_ms - 3_600_000 + 1, now_ms)    # cierra exactamente en now
    _install_fake_session(monkeypatch, [[just_closed, still_open]])

    with time_machine.travel(frozen, tick=False):
        result = binance.fetch_klines("BTCUSDT", start_ms=now_ms - 7_200_000, end_ms=now_ms + 10**9)

    assert result == [just_closed]


def test_request_params_are_correct(monkeypatch):
    """Ademas de startTime (ya cubierto), verificar symbol, interval, endTime y
    limit: si alguno se rompe, el backfill trae datos de mas o del simbolo/intervalo
    equivocado sin que ningun otro test lo note."""
    monkeypatch.setattr(binance, "MAX_LIMIT", 500)
    session = _install_fake_session(monkeypatch, [[_kline(1000, 1000)]])

    binance.fetch_klines("ETHUSDT", start_ms=1000, end_ms=999_999, interval="4h")

    params = session.get.call_args_list[0].kwargs["params"]
    assert params["symbol"] == "ETHUSDT"
    assert params["interval"] == "4h"
    assert params["endTime"] == 999_999
    assert params["limit"] == 500
