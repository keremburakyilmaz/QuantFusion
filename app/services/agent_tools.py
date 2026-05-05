"""Tool adapters that turn portfolio_id + an internal service into a JSON dict.

Used by AgentService.query — the LangGraph agent calls these directly,
not via HTTP self-loopback.
"""


import asyncio
import uuid
from typing import Any

from app.services.data_service import DataService, LOOKBACK_DEFAULT
from app.services.optimizer import PortfolioOptimizer
from app.services.portfolio_loader import load_holdings
from app.services.regime_service import RegimeService
from app.services.risk_service import RiskService


async def holdings_tool(
    portfolio_id: uuid.UUID, session_factory
) -> dict[str, Any]:
    weights = await asyncio.to_thread(_load_holdings, session_factory, portfolio_id)
    return {"weights": weights, "tickers": list(weights.keys())}


def _load_holdings(session_factory, portfolio_id):
    with session_factory() as db:
        return load_holdings(db, portfolio_id)


async def risk_tool(
    portfolio_id: uuid.UUID,
    session_factory,
    data: DataService,
    risk: RiskService,
) -> dict[str, Any]:
    weights = await asyncio.to_thread(_load_holdings, session_factory, portfolio_id)
    universe = list({*weights.keys(), "SPY"})
    returns = await data.get_returns(universe, lookback_days=LOOKBACK_DEFAULT)
    rf = await data.get_risk_free_rate()
    metrics = await asyncio.to_thread(risk.compute_all, weights, returns, rf)
    return metrics.model_dump(mode="json")


async def regime_tool(regime_service: RegimeService | None) -> dict[str, Any]:
    if regime_service is None:
        return {"error": "regime service unavailable"}
    snapshot = await regime_service.predict_current()
    if snapshot is None:
        return {"error": "regime model not yet available"}
    return snapshot.model_dump(mode="json")


async def optimize_tool(
    portfolio_id: uuid.UUID,
    session_factory,
    data: DataService,
    optimizer: PortfolioOptimizer,
    method: str = "mvo",
    target: str = "max_sharpe",
) -> dict[str, Any]:
    weights = await asyncio.to_thread(_load_holdings, session_factory, portfolio_id)
    tickers = list(weights.keys())
    returns = await data.get_returns(tickers, lookback_days=LOOKBACK_DEFAULT)
    if returns.empty or len(returns.columns) < 2:
        return {"error": "insufficient return data"}
    cov = returns.cov().values * 252
    rf = await data.get_risk_free_rate()

    def _solve():
        if method == "risk_parity":
            return optimizer.risk_parity(cov, list(returns.columns), returns=returns, rf=rf)
        return optimizer.mvo(returns, cov, target=target, rf=rf)

    result = await asyncio.to_thread(_solve)
    return result.model_dump(mode="json")


async def backtest_tool(
    portfolio_id: uuid.UUID,
    session_factory,
    data: DataService,
    lookback_years: int = 3,
    rebalance_freq: str = "monthly",
    transaction_cost_bps: float = 10.0,
) -> dict[str, Any]:
    from app.services.backtester import _run_sync

    weights = await asyncio.to_thread(_load_holdings, session_factory, portfolio_id)
    days = max(lookback_years * 252 + 30, 252)
    universe = list({*weights.keys(), "SPY"})
    returns = await data.get_returns(universe, lookback_days=days)
    if returns.empty:
        return {"error": "insufficient return data"}
    user_cols = [t for t in weights.keys() if t in returns.columns]
    sub = returns[user_cols]
    benchmark = returns["SPY"] if "SPY" in returns.columns else None

    result = await asyncio.to_thread(
        _run_sync, weights, sub, 10_000.0,
        rebalance_freq, transaction_cost_bps, benchmark,
    )
    # Trim equity curve for prompt size — only metrics + monthly returns matter
    return {
        "metrics": result.metrics.model_dump(),
        "monthly_returns": [m.model_dump() for m in result.monthly_returns],
        "rebalance_freq": result.rebalance_freq,
        "transaction_cost_bps": result.transaction_cost_bps,
        "lookback_years": lookback_years,
    }
