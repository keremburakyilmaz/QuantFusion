from __future__ import annotations

import logging

from app.services.data_service import DataService, LOOKBACK_DEFAULT


logger = logging.getLogger(__name__)

DEMO_TICKERS = ["JEPI", "JEPQ", "VOO", "QQQ", "SPY", "^VIX", "^TNX", "^IRX"]


async def sync_prices(data_service: DataService) -> None:
    logger.info("price_sync starting for %s", DEMO_TICKERS)
    try:
        df = await data_service.get_prices(DEMO_TICKERS, lookback_days=LOOKBACK_DEFAULT)
        logger.info("price_sync finished — %d rows in frame", len(df))
    except Exception:
        logger.exception("price_sync failed")
