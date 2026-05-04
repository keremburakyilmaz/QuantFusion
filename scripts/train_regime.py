from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache import create_redis
from app.database import SessionLocal
from app.services.data_service import DataService
from app.services.regime_service import MODEL_PATH, RegimeService


async def main() -> int:
    redis = create_redis()
    try:
        data = DataService(redis, SessionLocal)
        service = RegimeService(data, SessionLocal)
        print("Training HMM regime model...")
        await service.train()
        if MODEL_PATH.exists():
            print(f"Saved: {MODEL_PATH}")
            return 0
        print("Training did not produce a model (insufficient data?).")
        return 1
    finally:
        await redis.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
