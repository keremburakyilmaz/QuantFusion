from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from app.services.risk_service import RiskService


def _synthetic_returns(seed: int = 42, days: int = 252) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    data = {
        "A": rng.normal(loc=0.0006, scale=0.012, size=days),
        "B": rng.normal(loc=0.0004, scale=0.010, size=days),
        "C": rng.normal(loc=0.0008, scale=0.015, size=days),
        "SPY": rng.normal(loc=0.0005, scale=0.011, size=days),
    }
    return pd.DataFrame(data, index=idx)


def test_compute_all_populates_every_field():
    svc = RiskService()
    returns = _synthetic_returns()
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}

    m = svc.compute_all(weights, returns, rf=0.04)

    for field in (
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "var_historical",
        "var_parametric",
        "var_monte_carlo",
        "cvar",
        "beta",
        "tracking_error",
    ):
        value = getattr(m, field)
        assert value is not None, f"{field} is None"
        assert math.isfinite(value), f"{field} not finite: {value}"


def test_sanity_bounds():
    svc = RiskService()
    returns = _synthetic_returns()
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    m = svc.compute_all(weights, returns, rf=0.04)

    assert 0 < m.annualized_volatility < 1
    assert -1 < m.sharpe < 5
    assert -1 <= m.max_drawdown <= 0


def test_cvar_at_most_var():
    svc = RiskService()
    returns = _synthetic_returns()
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    m = svc.compute_all(weights, returns, rf=0.04)
    assert m.cvar <= m.var_historical + 1e-9


def test_beta_one_for_self():
    svc = RiskService()
    returns = _synthetic_returns()
    # Portfolio = 100% SPY → beta ≈ 1, tracking_error ≈ 0
    weights = {"SPY": 1.0}
    m = svc.compute_all(weights, returns, rf=0.04)
    assert m.beta == pytest.approx(1.0, abs=1e-6)
    assert m.tracking_error == pytest.approx(0.0, abs=1e-9)


def test_correlation_matrix_shape():
    svc = RiskService()
    returns = _synthetic_returns()
    weights = {"A": 0.5, "B": 0.5}
    m = svc.compute_all(weights, returns, rf=0.04)
    assert set(m.correlation_matrix.keys()) == {"A", "B"}
    assert m.correlation_matrix["A"]["A"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.asyncio
async def test_analyze_endpoint_returns_full_metrics(fake_redis):
    from app.main import app

    returns = _synthetic_returns()

    class StubDataService:
        async def get_returns(self, tickers, lookback_days=756):
            cols = [t for t in tickers if t in returns.columns]
            return returns[cols]

        async def get_risk_free_rate(self):
            return 0.04

    app.state.redis = fake_redis
    app.state.data_service = StubDataService()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/risk/analyze",
            json={
                "holdings": [
                    {"ticker": "A", "weight": 0.5},
                    {"ticker": "B", "weight": 0.5},
                ]
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["sharpe"] is not None
    assert body["beta"] is not None
    assert body["var_historical"] is not None
