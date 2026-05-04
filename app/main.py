import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.cache import create_redis
from app.config import settings
from app.database import SessionLocal
from app.limiter import limiter
from app.routers import analyzer as analyzer_router
from app.routers import backtest as backtest_router
from app.routers import market as market_router
from app.routers import optimize as optimize_router
from app.routers import regime as regime_router
from app.routers import risk as risk_router
from app.services.analyzer_service import AnalyzerService
from app.services.backtester import VectorizedBacktester
from app.services.data_service import DataService
from app.services.optimizer import PortfolioOptimizer
from app.services.regime_service import MODEL_PATH, RegimeService
from app.services.risk_service import RiskService
from app.services.snapshot_service import SnapshotService
from app.tasks.price_sync import sync_prices
from app.tasks.regime_update import update_regime
from app.tasks.snapshot_cleanup import cleanup_snapshots


scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = create_redis()
    app.state.data_service = DataService(app.state.redis, SessionLocal)
    app.state.regime_service = RegimeService(app.state.data_service, SessionLocal)
    app.state.analyzer_service = AnalyzerService(
        data=app.state.data_service,
        risk=RiskService(),
        optimizer=PortfolioOptimizer(),
        backtester=VectorizedBacktester(),
        regime=app.state.regime_service,
    )
    app.state.snapshot_service = SnapshotService(SessionLocal)

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
    scheduler.add_job(
        cleanup_snapshots,
        "cron",
        hour=2,
        args=[app.state.snapshot_service],
        id="snapshot_cleanup",
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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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
app.include_router(backtest_router.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(analyzer_router.router, prefix="/api/analyzer", tags=["analyzer"])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
