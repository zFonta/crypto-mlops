"""Cliente minimo de la API publica de Binance (sin API key)."""
import logging
import os
import time
from datetime import datetime, timezone

import requests

from common.config import LOCAL_TZ

BASE_URL=os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
KLINES_ENDPOINT= "/api/v3/klines"
MAX_LIMIT= 1000 # maximo de velas por request

log = logging.getLogger(__name__)


def fetch_klines (symbol: str, start_ms: int, end_ms: int | None = None,
    interval: str = "1h") -> list:
    """Descarga velas paginando de a 1000 desde `start ms` hasta `end ms`.
    
    Devuelve la lista cruda de klines de Binance.
    """
    if end_ms is None:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_klines: list = []
    cursor = start_ms
    session = requests.Session()

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_LIMIT
        }
        resp = session.get(BASE_URL + KLINES_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_klines.extend(batch)
        # siguiente pagina: 1 ms despues del open_time de la ultima vela
        cursor = batch[-1][0] + 1
        log.info("%s: %d velas acumuladas (hasta %s GMT-3)", symbol, len(all_klines),
                 datetime.fromtimestamp(batch[-1][0] / 1000, LOCAL_TZ))
        if len(batch) < MAX_LIMIT:
            break
        time.sleep(0.3)  # respetar rate limit
    
    #descartar la ultima vela en curso (aun no cerrada)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_klines = [k for k in all_klines if k[6] < now_ms] # k[6] = close_time
    return all_klines
    