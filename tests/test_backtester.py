from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.services.backtester import _run_sync


def _synth_returns(seed: int = 42, days: int = 756) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0006, 0.012, days),
            "B": rng.normal(0.0004, 0.010, days),
            "C": rng.normal(0.0008, 0.015, days),
        },
        index=idx,
    )


def _benchmark(seed: int = 1, days: int = 756) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.Series(rng.normal(0.0005, 0.011, days), index=idx, name="SPY")


def test_buy_and_hold_no_costs_matches_compounded_returns():
    returns = _synth_returns(days=252)
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    result = _run_sync(
        weights, returns, 10_000.0, "quarterly", 0.0, None,
    )
    naive = float(
        ((1 + returns).cumprod().iloc[-1].values * 10_000.0 / 3).sum()
    )
    final = result.equity_curve[-1].value
    assert abs(final - naive) / naive < 0.02


def test_metrics_are_finite():
    returns = _synth_returns()
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    result = _run_sync(weights, returns, 10_000.0, "monthly", 10.0, None)
    m = result.metrics
    for field in (
        "total_return", "cagr", "sharpe", "sortino",
        "max_drawdown", "calmar", "win_rate",
        "best_month", "worst_month",
    ):
        assert math.isfinite(getattr(m, field))


def test_transaction_costs_strictly_lose_capital():
    returns = _synth_returns()
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    free = _run_sync(weights, returns, 10_000.0, "monthly", 0.0, None)
    expensive = _run_sync(weights, returns, 10_000.0, "monthly", 100.0, None)
    assert expensive.equity_curve[-1].value < free.equity_curve[-1].value


def test_benchmark_is_attached_when_provided():
    returns = _synth_returns(days=252)
    bench = _benchmark(days=252)
    weights = {"A": 1.0}
    result = _run_sync(weights, returns, 10_000.0, "monthly", 10.0, bench)
    assert all(p.benchmark_value is not None for p in result.equity_curve)


def test_monthly_returns_count_close_to_lookback_in_months():
    returns = _synth_returns(days=756)
    weights = {"A": 0.5, "B": 0.5}
    result = _run_sync(weights, returns, 10_000.0, "monthly", 0.0, None)
    assert 30 <= len(result.monthly_returns) <= 40


def test_empty_returns_returns_empty_result():
    weights = {"X": 1.0}
    returns = pd.DataFrame()
    result = _run_sync(weights, returns, 10_000.0, "monthly", 10.0, None)
    assert result.equity_curve == []
    assert result.metrics.total_return == 0.0
