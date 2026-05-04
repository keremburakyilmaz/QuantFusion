from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.schemas.common import HoldingInput
from app.schemas.regime import RegimeSnapshotResponse
from app.services.analyzer_service import AnalyzerService
from app.services.backtester import VectorizedBacktester
from app.services.optimizer import PortfolioOptimizer
from app.services.risk_service import RiskService


def _synth_returns(days: int = 756) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(
        {
            "AAPL": rng.normal(0.0008, 0.014, days),
            "MSFT": rng.normal(0.0006, 0.012, days),
            "SPY": rng.normal(0.0005, 0.010, days),
        },
        index=idx,
    )


class _StubData:
    def __init__(self, returns: pd.DataFrame):
        self._returns = returns

    async def get_returns(self, tickers, lookback_days=756):
        cols = [t for t in tickers if t in self._returns.columns]
        return self._returns[cols]

    async def get_fundamentals(self, ticker):
        return {
            "dividend_yield": 0.01,
            "trailing_pe": 25.0,
            "market_cap": 2_000_000_000_000,
            "beta": 1.1,
            "next_earnings_date": None,
            "sector": "Tech",
        }

    async def get_risk_free_rate(self):
        return 0.04

    async def validate_tickers(self, tickers):
        return {"valid": tickers, "invalid": []}


class _StubRegime:
    async def predict_current(self):
        return RegimeSnapshotResponse(
            ts=pd.Timestamp.utcnow(),
            regime="bull",
            confidence=0.85,
            features={"log_return": 0.001},
            probabilities={"bull": 0.7, "sideways": 0.2, "bear": 0.1},
        )


class _StubRegimeNoProb:
    async def predict_current(self):
        return None


@pytest.mark.asyncio
async def test_analyzer_run_full_report():
    returns = _synth_returns()
    analyzer = AnalyzerService(
        data=_StubData(returns),
        risk=RiskService(),
        optimizer=PortfolioOptimizer(),
        backtester=VectorizedBacktester(),
        regime=_StubRegime(),
    )
    holdings = [
        HoldingInput(ticker="AAPL", weight=0.5),
        HoldingInput(ticker="MSFT", weight=0.5),
    ]
    report = await analyzer.run(holdings)

    assert report.holdings == holdings
    assert report.risk.sharpe is not None
    assert len(report.frontier) >= 100
    assert report.optimized_mvo.method == "mvo"
    assert report.optimized_rp.method == "risk_parity"
    assert report.optimized_blended is not None
    assert report.optimized_blended.method == "regime_blended"
    assert len(report.backtest_1y.equity_curve) > 0
    assert len(report.backtest_3y.equity_curve) > 0
    assert report.regime is not None
    assert report.regime.regime == "bull"
    assert set(report.fundamentals.keys()) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_analyzer_run_without_regime_omits_blended():
    returns = _synth_returns()
    analyzer = AnalyzerService(
        data=_StubData(returns),
        risk=RiskService(),
        optimizer=PortfolioOptimizer(),
        backtester=VectorizedBacktester(),
        regime=_StubRegimeNoProb(),
    )
    holdings = [HoldingInput(ticker="AAPL", weight=1.0)]
    report = await analyzer.run(holdings)
    assert report.regime is None
    assert report.optimized_blended is None


@pytest.mark.asyncio
async def test_analyzer_validate_passthrough():
    returns = _synth_returns()
    analyzer = AnalyzerService(
        data=_StubData(returns),
        risk=RiskService(),
        optimizer=PortfolioOptimizer(),
        backtester=VectorizedBacktester(),
        regime=None,
    )
    result = await analyzer.validate(["AAPL", "MSFT"])
    assert result == {"valid": ["AAPL", "MSFT"], "invalid": []}
