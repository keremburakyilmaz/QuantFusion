import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.cache import create_redis
from app.config import settings
from app.database import SessionLocal
from app.routers import market as market_router
from app.routers import optimize as optimize_router
from app.routers import regime as regime_router
from app.routers import risk as risk_router
from app.services.data_service import DataService
from app.services.regime_service import MODEL_PATH, RegimeService
from app.tasks.price_sync import sync_prices
from app.tasks.regime_update import update_regime


scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = create_redis()
    app.state.data_service = DataService(app.state.redis, SessionLocal)
    app.state.regime_service = RegimeService(app.state.data_service, SessionLocal)

    scheduler.add_job(
        sync_prices,
        "interval",
        hours=1,
        args=[app.state.data_service],
        next_run_time=datetime.utcnow(),
        id="price_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        update_regime,
        "cron",
        hour=1,
        args=[app.state.regime_service],
        id="regime_update",
        replace_existing=True,
    )
    scheduler.start()

    if not MODEL_PATH.exists():
        asyncio.create_task(app.state.regime_service.train())

    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        await app.state.redis.aclose()


app = FastAPI(title="QuantFusion API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router.router, prefix="/api/market", tags=["market"])
app.include_router(risk_router.router, prefix="/api/risk", tags=["risk"])
app.include_router(optimize_router.router, prefix="/api/optimize", tags=["optimize"])
app.include_router(regime_router.router, prefix="/api/regime", tags=["regime"])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
