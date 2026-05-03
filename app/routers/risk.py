from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.common import PortfolioInput
from app.schemas.risk import RiskMetrics, VaRResult
from app.services.data_service import DataService, LOOKBACK_DEFAULT
from app.services.portfolio_loader import load_holdings
from app.services.risk_service import BENCHMARK, RiskService


router = APIRouter()
risk_service = RiskService()


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


@router.post("/analyze", response_model=RiskMetrics)
async def analyze(
    body: PortfolioInput,
    data: DataService = Depends(get_data_service),
) -> RiskMetrics:
    weights = {h.ticker: h.weight for h in body.holdings}
    returns, rf = await _fetch_returns_and_rf(data, list(weights.keys()))
    return risk_service.compute_all(weights, returns, rf=rf)
