from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from app.schemas.backtest import (
    BacktestMetrics,
    BacktestResult,
    EquityPoint,
    MonthlyReturn,
)


logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    if freq == "daily":
        return set(index)
    if freq == "weekly":
        groups = index.to_series().groupby(
            [index.isocalendar().year, index.isocalendar().week]
        )
    elif freq == "monthly":
        groups = index.to_series().groupby([index.year, index.month])
    elif freq == "quarterly":
        groups = index.to_series().groupby([index.year, index.quarter])
    else:
        raise ValueError(f"Unknown rebalance_freq: {freq}")
    return set(groups.tail(1))


class VectorizedBacktester:
    async def run(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        initial_capital: float = 10_000.0,
        rebalance_freq: str = "monthly",
        transaction_cost_bps: float = 10.0,
        benchmark: pd.Series | None = None,
    ) -> BacktestResult:
        return _run_sync(
            weights, returns, initial_capital,
            rebalance_freq, transaction_cost_bps, benchmark,
        )

    async def compare(
        self,
        portfolios: list[dict[str, float]],
        returns: pd.DataFrame,
        initial_capital: float = 10_000.0,
        rebalance_freq: str = "monthly",
        transaction_cost_bps: float = 10.0,
        benchmark: pd.Series | None = None,
    ) -> list[BacktestResult]:
        return [
            _run_sync(
                w, returns, initial_capital,
                rebalance_freq, transaction_cost_bps, benchmark,
            )
            for w in portfolios
        ]


def _run_sync(
    weights: dict[str, float],
    returns: pd.DataFrame,
    initial_capital: float,
    rebalance_freq: str,
    transaction_cost_bps: float,
    benchmark: pd.Series | None,
) -> BacktestResult:
    cols = [t for t in weights if t in returns.columns]
    if not cols:
        return _empty_result(rebalance_freq, transaction_cost_bps)
    ret = returns[cols].dropna(how="all").fillna(0.0)
    if ret.empty:
        return _empty_result(rebalance_freq, transaction_cost_bps)

    target = np.array([weights[t] for t in cols], dtype=float)
    target = target / target.sum()
    cost_rate = transaction_cost_bps / 10_000.0

    n_days, n_assets = ret.shape
    holdings = np.zeros((n_days, n_assets))
    portfolio_value = np.zeros(n_days)

    holdings[0] = target * initial_capital
    portfolio_value[0] = initial_capital
    rebal_set = _rebalance_dates(ret.index, rebalance_freq)
    daily_rebal = rebalance_freq == "daily"

    for t in range(1, n_days):
        prev = holdings[t - 1] * (1.0 + ret.iloc[t].values)
        total = float(prev.sum())
        ts = ret.index[t]

        if daily_rebal or ts in rebal_set:
            new_holdings = target * total
            turnover = float(np.abs(new_holdings - prev).sum())
            cost = turnover * cost_rate
            total -= cost
            new_holdings = target * total
        else:
            new_holdings = prev

        holdings[t] = new_holdings
        portfolio_value[t] = total

    equity = pd.Series(portfolio_value, index=ret.index, name="equity")
    daily_returns = equity.pct_change().fillna(0.0)

    bench_curve = _benchmark_curve(benchmark, ret.index, initial_capital)

    metrics = _compute_metrics(equity, daily_returns)
    monthly = _monthly_returns(equity)
    points = _equity_points(equity, bench_curve)

    return BacktestResult(
        equity_curve=points,
        metrics=metrics,
        monthly_returns=monthly,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps,
    )


def _benchmark_curve(
    benchmark: pd.Series | None,
    index: pd.DatetimeIndex,
    initial_capital: float,
) -> pd.Series | None:
    if benchmark is None:
        return None
    aligned = benchmark.reindex(index).fillna(0.0)
    return (1.0 + aligned).cumprod() * initial_capital


def _compute_metrics(equity: pd.Series, daily_returns: pd.Series) -> BacktestMetrics:
    if equity.empty or len(equity) < 2:
        return _zero_metrics()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max(len(equity) / TRADING_DAYS, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)

    vol = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mean_ann = float(daily_returns.mean() * TRADING_DAYS)
    sharpe = mean_ann / vol if vol > 0 else 0.0

    downside = daily_returns[daily_returns < 0]
    dd_vol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
    sortino = mean_ann / dd_vol if dd_vol > 0 else 0.0

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    monthly = equity.resample("ME").last().pct_change().dropna()
    win_rate = float((monthly > 0).mean()) if not monthly.empty else 0.0
    best_month = float(monthly.max()) if not monthly.empty else 0.0
    worst_month = float(monthly.min()) if not monthly.empty else 0.0

    return BacktestMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        best_month=best_month,
        worst_month=worst_month,
    )


def _zero_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.0, cagr=0.0, sharpe=0.0, sortino=0.0,
        max_drawdown=0.0, calmar=0.0, win_rate=0.0,
        best_month=0.0, worst_month=0.0,
    )


def _monthly_returns(equity: pd.Series) -> list[MonthlyReturn]:
    if equity.empty:
        return []
    monthly = equity.resample("ME").last().pct_change().dropna()
    out: list[MonthlyReturn] = []
    for ts, value in monthly.items():
        out.append(
            MonthlyReturn(
                year=int(ts.year),
                month=int(ts.month),
                return_pct=float(value),
            )
        )
    return out


def _equity_points(
    equity: pd.Series, bench_curve: pd.Series | None
) -> list[EquityPoint]:
    out: list[EquityPoint] = []
    for ts, value in equity.items():
        bench_val = (
            float(bench_curve.loc[ts]) if bench_curve is not None and ts in bench_curve.index else None
        )
        out.append(
            EquityPoint(
                date=ts.date() if hasattr(ts, "date") else ts,
                value=float(value),
                benchmark_value=bench_val,
            )
        )
    return out


def _empty_result(freq: str, cost_bps: float) -> BacktestResult:
    return BacktestResult(
        equity_curve=[],
        metrics=_zero_metrics(),
        monthly_returns=[],
        rebalance_freq=freq,
        transaction_cost_bps=cost_bps,
    )
