from unittest.mock import MagicMock

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
