

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.limiter import limiter
from app.schemas.backtest import (
    BacktestResult,
    BacktestRunRequest,
    BacktestStatelessRequest,
    CompareRequest,
    CompareResult,
)
from app.services.backtester import VectorizedBacktester, _run_sync
from app.services.data_service import DataService
from app.services.portfolio_loader import load_holdings


router = APIRouter()
backtester = VectorizedBacktester()


def get_data_service(request: Request) -> DataService:
    return request.app.state.data_service


def _trading_days(years: int) -> int:
    return max(years * 252 + 30, 252)


async def _fetch_returns_with_benchmark(
    data: DataService, tickers: list[str], lookback_days: int
):
    universe = list({*tickers, "SPY"})
    returns = await data.get_returns(universe, lookback_days=lookback_days)
    if returns.empty:
        raise HTTPException(status_code=422, detail="No return data available")
    benchmark = returns["SPY"] if "SPY" in returns.columns else None
    user_cols = [t for t in tickers if t in returns.columns]
    if not user_cols:
        raise HTTPException(status_code=422, detail="None of the requested tickers have data")
    return returns[user_cols], benchmark


@router.post("/run/{portfolio_id}", response_model=BacktestResult)
async def backtest_run_demo(
    portfolio_id: uuid.UUID,
    body: BacktestRunRequest | None = None,
    db: Session = Depends(get_db),
    data: DataService = Depends(get_data_service),
) -> BacktestResult:
    body = body or BacktestRunRequest(portfolio_id=portfolio_id)
    weights = load_holdings(db, portfolio_id)
    returns, benchmark = await _fetch_returns_with_benchmark(
        data, list(weights.keys()), _trading_days(body.lookback_years)
    )
    return await asyncio.to_thread(
        _run_sync, weights, returns, 10_000.0,
        body.rebalance_freq, body.transaction_cost_bps, benchmark,
    )


@router.post("/run", response_model=BacktestResult)
@limiter.limit("10/minute")
async def backtest_run_stateless(
    request: Request,
    body: BacktestStatelessRequest,
    data: DataService = Depends(get_data_service),
) -> BacktestResult:
    weights = {h.ticker: h.weight for h in body.holdings}
    returns, benchmark = await _fetch_returns_with_benchmark(
        data, list(weights.keys()), _trading_days(body.lookback_years)
    )
    return await asyncio.to_thread(
        _run_sync, weights, returns, 10_000.0,
        body.rebalance_freq, body.transaction_cost_bps, benchmark,
    )


@router.post("/compare", response_model=CompareResult)
@limiter.limit("5/minute")
async def backtest_compare(
    request: Request,
    body: CompareRequest,
    data: DataService = Depends(get_data_service),
) -> CompareResult:
    all_tickers = {h.ticker for p in body.portfolios for h in p.holdings}
    returns_full, benchmark = await _fetch_returns_with_benchmark(
        data, list(all_tickers), _trading_days(body.lookback_years)
    )

    def _solve_all() -> list[BacktestResult]:
        out: list[BacktestResult] = []
        for portfolio in body.portfolios:
            weights = {h.ticker: h.weight for h in portfolio.holdings}
            cols = [t for t in weights if t in returns_full.columns]
            sub = returns_full[cols]
            out.append(
                _run_sync(
                    weights, sub, 10_000.0,
                    body.rebalance_freq, body.transaction_cost_bps, benchmark,
                )
            )
        return out

    results = await asyncio.to_thread(_solve_all)

    benchmark_only = await asyncio.to_thread(
        _run_sync,
        {"SPY": 1.0},
        benchmark.to_frame(name="SPY") if benchmark is not None else returns_full.iloc[:, :1],
        10_000.0, body.rebalance_freq, 0.0, benchmark,
    )

    return CompareResult(results=results, benchmark_metrics=benchmark_only.metrics)
