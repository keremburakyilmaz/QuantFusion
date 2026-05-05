

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas.common import HoldingInput
from app.schemas.regime import RegimeSnapshotResponse
from app.schemas.risk import RiskMetrics
from app.services.agent_service import AgentService


def _make_llm(*responses):
    """Build a fake LLM whose ainvoke returns a SimpleNamespace(content=text)
    for each call in order, then raises if called more times."""
    iter_responses = iter(responses)

    async def ainvoke(_prompt):
        try:
            text = next(iter_responses)
        except StopIteration as exc:
            raise AssertionError("LLM called more times than expected") from exc
        return SimpleNamespace(content=text)

    return SimpleNamespace(ainvoke=ainvoke)


def _regime() -> RegimeSnapshotResponse:
    from datetime import datetime, timezone
    return RegimeSnapshotResponse(
        ts=datetime.now(tz=timezone.utc),
        regime="bull",
        confidence=0.82,
        features={"log_return": 0.001},
        probabilities={"bull": 0.7, "sideways": 0.2, "bear": 0.1},
    )


def _holdings():
    return [
        HoldingInput(ticker="AAPL", weight=0.5),
        HoldingInput(ticker="MSFT", weight=0.5),
    ]


def _risk_metrics() -> RiskMetrics:
    return RiskMetrics(
        sharpe=1.2,
        var_historical=-0.018,
        max_drawdown=-0.12,
        beta=1.05,
    )


@pytest.mark.asyncio
async def test_regime_commentary_returns_mocked_text():
    llm = _make_llm("AAPL/MSFT look fine in a bull regime; watch beta.")
    svc = AgentService(llm_factory=lambda: llm)
    text = await svc.regime_commentary(_regime(), _holdings(), _risk_metrics())
    assert "bull" in text.lower() or "AAPL" in text


@pytest.mark.asyncio
async def test_regime_commentary_returns_empty_on_failure():
    async def boom(_p):
        raise RuntimeError("network down")
    llm = SimpleNamespace(ainvoke=boom)
    svc = AgentService(llm_factory=lambda: llm)
    text = await svc.regime_commentary(_regime(), _holdings(), _risk_metrics())
    assert text == ""


@pytest.mark.asyncio
async def test_regime_commentary_empty_when_llm_disabled():
    svc = AgentService(llm_factory=lambda: None)
    text = await svc.regime_commentary(_regime(), _holdings(), _risk_metrics())
    assert text == ""


@pytest.mark.asyncio
async def test_query_routes_to_correct_tool_and_formats_response():
    pid = uuid.uuid4()
    captured = {}

    async def risk_tool(portfolio_id, **kwargs):
        captured["portfolio_id"] = portfolio_id
        return {"sharpe": 1.2, "var_historical": -0.018}

    llm = _make_llm(
        json.dumps({"intent": "risk", "args": {}}),
        "Your Sharpe ratio is 1.20 and your 95% historical VaR is -1.8%.",
    )
    svc = AgentService(
        llm_factory=lambda: llm,
        tools={"risk": risk_tool},
    )

    response = await svc.query("what's my Sharpe?", pid)

    assert response.intent == "risk"
    assert "1.2" in response.response or "Sharpe" in response.response
    assert response.data == {"sharpe": 1.2, "var_historical": -0.018}
    assert captured["portfolio_id"] == pid


@pytest.mark.asyncio
async def test_query_falls_back_to_holdings_on_malformed_router_output():
    pid = uuid.uuid4()
    holdings_called = AsyncMock(return_value={"tickers": ["AAPL"]})

    llm = _make_llm(
        "this is not json at all",
        "You hold AAPL.",
    )
    svc = AgentService(
        llm_factory=lambda: llm,
        tools={"holdings": holdings_called},
    )

    response = await svc.query("what do I have?", pid)

    assert response.intent == "holdings"
    holdings_called.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_handles_each_intent():
    pid = uuid.uuid4()
    seen: list[str] = []

    def make_tool(name):
        async def tool(portfolio_id, **kwargs):
            seen.append(name)
            return {"intent_name": name}
        return tool

    tools = {name: make_tool(name) for name in
             ("risk", "optimize", "regime", "backtest", "holdings")}

    for intent in ("risk", "optimize", "regime", "backtest", "holdings"):
        llm = _make_llm(
            json.dumps({"intent": intent, "args": {}}),
            f"You asked about {intent}.",
        )
        svc = AgentService(llm_factory=lambda llm=llm: llm, tools=tools)
        response = await svc.query(f"{intent} please", pid)
        assert response.intent == intent

    assert sorted(seen) == sorted(["risk", "optimize", "regime", "backtest", "holdings"])


@pytest.mark.asyncio
async def test_query_raises_when_llm_disabled():
    svc = AgentService(llm_factory=lambda: None)
    with pytest.raises(RuntimeError):
        await svc.query("anything", uuid.uuid4())


@pytest.mark.asyncio
async def test_commentary_uses_redis_cache(fake_redis):
    llm = _make_llm("Cached commentary first call.")
    svc = AgentService(llm_factory=lambda: llm, redis=fake_redis)
    first = await svc.regime_commentary(_regime(), _holdings(), _risk_metrics())
    # Second call should hit the cache (LLM only has one response queued)
    second = await svc.regime_commentary(_regime(), _holdings(), _risk_metrics())
    assert first == second
    assert "Cached" in first
