from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.models import PortfolioSnapshot
from app.schemas.analyzer import AnalysisReport
from app.schemas.backtest import BacktestMetrics, BacktestResult
from app.schemas.common import HoldingInput
from app.schemas.optimization import OptimizationResult
from app.schemas.risk import RiskMetrics
from app.services.snapshot_service import SnapshotService


def _zero_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.0, cagr=0.0, sharpe=0.0, sortino=0.0,
        max_drawdown=0.0, calmar=0.0, win_rate=0.0,
        best_month=0.0, worst_month=0.0,
    )


def _zero_backtest() -> BacktestResult:
    return BacktestResult(
        equity_curve=[], metrics=_zero_metrics(), monthly_returns=[],
        rebalance_freq="monthly", transaction_cost_bps=10.0,
    )


def _zero_opt(method: str) -> OptimizationResult:
    return OptimizationResult(
        method=method, target=None,
        weights={"AAPL": 1.0},
        expected_return=0.0, volatility=0.0, sharpe=0.0, solve_ms=0,
    )


def _sample_report() -> AnalysisReport:
    return AnalysisReport(
        holdings=[HoldingInput(ticker="AAPL", weight=1.0)],
        risk=RiskMetrics(),
        frontier=[],
        optimized_mvo=_zero_opt("mvo"),
        optimized_rp=_zero_opt("risk_parity"),
        optimized_blended=None,
        backtest_1y=_zero_backtest(),
        backtest_3y=_zero_backtest(),
        regime=None,
        regime_commentary="",
        fundamentals={},
        generated_at=datetime.now(tz=timezone.utc),
    )


class _FakeSession:
    """In-memory PortfolioSnapshot table that mimics SQLAlchemy session API
    enough for the SnapshotService sync paths."""

    def __init__(self, store: dict[str, PortfolioSnapshot]):
        self.store = store
        self._pending: list[PortfolioSnapshot] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def add(self, snapshot: PortfolioSnapshot) -> None:
        if snapshot.token in self.store:
            from sqlalchemy.exc import IntegrityError
            raise IntegrityError("dup", {}, Exception("dup"))
        self._pending.append(snapshot)

    def execute(self, stmt):
        # Accept either a select or a delete statement — inspect the
        # compiled SQL string to branch.
        sql = str(stmt).lower()
        if sql.startswith("select"):
            token = stmt.compile().params.get("token_1")
            row = self.store.get(token) if token else None
            return _ScalarResult(row)
        if sql.startswith("delete"):
            now = datetime.now(tz=timezone.utc)
            to_delete = [
                t for t, snap in self.store.items()
                if snap.expires_at is not None and snap.expires_at < now
            ]
            for t in to_delete:
                del self.store[t]
            result = MagicMock()
            result.rowcount = len(to_delete)
            return result
        raise NotImplementedError(sql)

    def commit(self) -> None:
        for snap in self._pending:
            if snap.created_at is None:
                snap.created_at = datetime.now(tz=timezone.utc)
            self.store[snap.token] = snap
        self._pending.clear()


class _ScalarResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


@pytest.fixture
def session_factory():
    store: dict[str, PortfolioSnapshot] = {}
    return lambda: _FakeSession(store), store


@pytest.mark.asyncio
async def test_save_and_fetch_round_trip(session_factory):
    factory, _ = session_factory
    svc = SnapshotService(factory)
    holdings = [HoldingInput(ticker="AAPL", weight=1.0)]
    report = _sample_report()

    saved = await svc.save(holdings, report, expires_in_days=30)
    assert saved.token
    assert saved.expires_at is not None
    assert saved.share_url.endswith(saved.token)

    fetched = await svc.fetch(saved.token)
    assert fetched is not None
    assert fetched.token == saved.token
    assert [h.ticker for h in fetched.holdings] == ["AAPL"]
    assert fetched.report.holdings == holdings


@pytest.mark.asyncio
async def test_save_with_zero_days_means_never_expires(session_factory):
    factory, _ = session_factory
    svc = SnapshotService(factory)
    saved = await svc.save(
        [HoldingInput(ticker="AAPL", weight=1.0)],
        _sample_report(),
        expires_in_days=0,
    )
    assert saved.expires_at is None


@pytest.mark.asyncio
async def test_expired_snapshot_returns_none(session_factory):
    factory, store = session_factory
    svc = SnapshotService(factory)
    saved = await svc.save(
        [HoldingInput(ticker="AAPL", weight=1.0)],
        _sample_report(),
        expires_in_days=30,
    )
    # Force-expire it by mutating the in-memory store
    store[saved.token].expires_at = datetime.now(tz=timezone.utc) - timedelta(days=1)
    fetched = await svc.fetch(saved.token)
    assert fetched is None


@pytest.mark.asyncio
async def test_fetch_unknown_token_returns_none(session_factory):
    factory, _ = session_factory
    svc = SnapshotService(factory)
    fetched = await svc.fetch("does-not-exist")
    assert fetched is None


@pytest.mark.asyncio
async def test_cleanup_expired_only_removes_past(session_factory):
    factory, store = session_factory
    svc = SnapshotService(factory)
    holdings = [HoldingInput(ticker="AAPL", weight=1.0)]
    report = _sample_report()

    fresh1 = await svc.save(holdings, report, expires_in_days=30)
    fresh2 = await svc.save(holdings, report, expires_in_days=30)
    expired1 = await svc.save(holdings, report, expires_in_days=30)
    expired2 = await svc.save(holdings, report, expires_in_days=30)
    expired3 = await svc.save(holdings, report, expires_in_days=30)

    past = datetime.now(tz=timezone.utc) - timedelta(days=1)
    for tok in (expired1.token, expired2.token, expired3.token):
        store[tok].expires_at = past

    deleted = await svc.cleanup_expired()
    assert deleted == 3
    assert fresh1.token in store
    assert fresh2.token in store


@pytest.mark.asyncio
async def test_tokens_are_unique_across_many_saves(session_factory):
    factory, _ = session_factory
    svc = SnapshotService(factory)
    holdings = [HoldingInput(ticker="AAPL", weight=1.0)]
    report = _sample_report()
    tokens = set()
    for _ in range(50):
        saved = await svc.save(holdings, report, expires_in_days=30)
        tokens.add(saved.token)
    assert len(tokens) == 50
