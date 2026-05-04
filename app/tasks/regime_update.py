from __future__ import annotations

import asyncio
import logging

from app.services.regime_service import RegimeService


logger = logging.getLogger(__name__)


async def update_regime(regime_service: RegimeService) -> None:
    logger.info("regime_update starting")
    try:
        await regime_service.train()
        snapshot = await regime_service.predict_current()
        if snapshot is None:
            logger.warning("regime_update: prediction returned None")
        else:
            logger.info(
                "regime_update finished: %s (confidence %.2f)",
                snapshot.regime,
                snapshot.confidence,
            )
    except Exception:
        logger.exception("regime_update failed")
