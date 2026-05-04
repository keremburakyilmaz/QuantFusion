

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.limiter import limiter
from app.schemas.common import PortfolioInput
from app.schemas.optimization import EfficientFrontierResponse
from app.schemas.risk import RiskMetrics, VaRResult
from app.services.data_service import DataService, LOOKBACK_DEFAULT
from app.services.optimizer import PortfolioOptimizer
from app.services.portfolio_loader import load_holdings
from app.services.risk_service import BENCHMARK, RiskService


router = APIRouter()
risk_service = RiskService()
optimizer = PortfolioOptimizer()


def get_data_service(request: Request) -> DataService:
    return request.app.state.data_service


async def _fetch_returns_and_rf(
    data: DataService, tickers: list[str], lookback_days: int = LOOKBACK_DEFAULT
):
    universe = list({*tickers, BENCHMARK})
    returns = await data.get_returns(universe, lookback_days=lookback_days)
    rf = await data.get_risk_free_rate()
    return returns, rf


@router.get("/{portfolio_id}/metrics", response_model=RiskMetrics)
async def get_metrics(
    portfolio_id: uuid.UUID,
    db: Session = Depends(get_db),
    data: DataService = Depends(get_data_service),
) -> RiskMetrics:
    weights = load_holdings(db, portfolio_id)
    returns, rf = await _fetch_returns_and_rf(data, list(weights.keys()))
    return risk_service.compute_all(weights, returns, rf=rf)


@router.get("/{portfolio_id}/var", response_model=VaRResult)
async def get_var(
    portfolio_id: uuid.UUID,
    db: Session = Depends(get_db),
    data: DataService = Depends(get_data_service),
) -> VaRResult:
    weights = load_holdings(db, portfolio_id)
    returns, _ = await _fetch_returns_and_rf(data, list(weights.keys()))
    return risk_service.compute_var(weights, returns)


@router.get(
    "/{portfolio_id}/efficient_frontier", response_model=EfficientFrontierResponse
)
async def get_efficient_frontier(
    portfolio_id: uuid.UUID,
    db: Session = Depends(get_db),
    data: DataService = Depends(get_data_service),
) -> EfficientFrontierResponse:
    weights = load_holdings(db, portfolio_id)
    tickers = list(weights.keys())
    returns = await data.get_returns(tickers, lookback_days=LOOKBACK_DEFAULT)
    if returns.empty or len(returns.columns) < 2:
        raise HTTPException(status_code=422, detail="Insufficient return data")
    cov = returns.cov().values * 252
    rf = await data.get_risk_free_rate()
    points = optimizer.efficient_frontier(
        returns, cov, n=150, rf=rf, current_weights=weights
    )
    return EfficientFrontierResponse(points=points)


@router.post("/analyze", response_model=RiskMetrics)
@limiter.limit("10/minute")
async def analyze(
    request: Request,
    body: PortfolioInput,
    data: DataService = Depends(get_data_service),
) -> RiskMetrics:
    weights = {h.ticker: h.weight for h in body.holdings}
    returns, rf = await _fetch_returns_and_rf(data, list(weights.keys()))
    return risk_service.compute_all(weights, returns, rf=rf)
