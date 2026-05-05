

import uuid
from datetime import date

from pydantic import BaseModel, Field

from app.schemas.common import HoldingInput, PortfolioInput
from app.schemas.document import EarningsEvent


class EquityPoint(BaseModel):
    date: date
    value: float
    benchmark_value: float | None = None


class MonthlyReturn(BaseModel):
    year: int
    month: int
    return_pct: float


class BacktestMetrics(BaseModel):
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    best_month: float
    worst_month: float


class BacktestResult(BaseModel):
    equity_curve: list[EquityPoint]
    metrics: BacktestMetrics
    monthly_returns: list[MonthlyReturn]
    rebalance_freq: str
    transaction_cost_bps: float
    events: list[EarningsEvent] | None = None


class BacktestRunRequest(BaseModel):
    portfolio_id: uuid.UUID
    rebalance_freq: str = "monthly"
    transaction_cost_bps: float = 10.0
    lookback_years: int = Field(default=3, ge=1, le=10)


class BacktestStatelessRequest(BaseModel):
    holdings: list[HoldingInput] = Field(min_length=1)
    rebalance_freq: str = "monthly"
    transaction_cost_bps: float = 10.0
    lookback_years: int = Field(default=3, ge=1, le=10)


class CompareRequest(BaseModel):
    portfolios: list[PortfolioInput] = Field(min_length=2, max_length=4)
    rebalance_freq: str = "monthly"
    transaction_cost_bps: float = 10.0
    lookback_years: int = Field(default=3, ge=1, le=10)


class CompareResult(BaseModel):
    results: list[BacktestResult]
    benchmark_metrics: BacktestMetrics
