from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import OptimizationRun
from app.schemas.optimization import (
    OptimizationResult,
    OptimizeRunRequest,
    OptimizeStatelessRequest,
)
from app.services.data_service import DataService, LOOKBACK_DEFAULT
from app.services.optimizer import PortfolioOptimizer
from app.services.portfolio_loader import load_holdings
from app.services.regime_service import RegimeService


router = APIRouter()
optimizer = PortfolioOptimizer()


def get_data_service(request: Request) -> DataService:
    return request.app.state.data_service


def get_regime_service(request: Request) -> RegimeService | None:
    return getattr(request.app.state, "regime_service", None)


async def _resolve_regime_probabilities(
    explicit: dict[str, float] | None,
    regime_service: RegimeService | None,
) -> dict[str, float]:
    if explicit is not None:
        return explicit
    if regime_service is None:
        raise HTTPException(
            status_code=503,
            detail="Regime model not available; provide regime_probabilities explicitly",
        )
    snapshot = await regime_service.predict_current()
    if snapshot is None or snapshot.probabilities is None:
        raise HTTPException(
            status_code=503,
            detail="Regime model unavailable; provide regime_probabilities explicitly",
        )
    return snapshot.probabilities


async def _run_optimization(
    method: str,
    tickers: list[str],
    data: DataService,
    target: str | None = "max_sharpe",
    constraints_dict: dict | None = None,
    views: list[dict] | None = None,
    risk_aversion: float = 2.5,
    regime_probabilities: dict[str, float] | None = None,
) -> OptimizationResult:
    returns = await data.get_returns(tickers, lookback_days=LOOKBACK_DEFAULT)
    if returns.empty or len(returns.columns) < 2:
        raise HTTPException(status_code=422, detail="Insufficient return data")
    cov = returns.cov().values * 252
    rf = await data.get_risk_free_rate()

    def _solve() -> OptimizationResult:
        if method == "mvo":
            return optimizer.mvo(
                returns, cov, target=target or "max_sharpe",
                constraints=constraints_dict, rf=rf,
            )
        if method == "risk_parity":
            return optimizer.risk_parity(cov, list(returns.columns), returns=returns, rf=rf)
        if method == "black_litterman":
            caps = {}
            return optimizer.black_litterman(
                caps, cov, list(returns.columns),
                views=views, risk_aversion=risk_aversion,
                returns=returns, rf=rf,
            )
        if method == "regime_blended":
            if not regime_probabilities:
                raise HTTPException(
                    status_code=503,
                    detail="regime_blended requires regime_probabilities",
                )
            return optimizer.regime_blended(
                returns, cov,
                regime_probabilities=regime_probabilities,
                rf=rf,
                constraints=constraints_dict,
            )
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    if method == "black_litterman":
        fundamentals = await asyncio.gather(
            *[data.get_fundamentals(t) for t in returns.columns]
        )
        caps = {
            t: f.get("market_cap")
            for t, f in zip(returns.columns, fundamentals)
            if f.get("market_cap")
        }

        def _bl_solve() -> OptimizationResult:
            return optimizer.black_litterman(
                caps, cov, list(returns.columns),
                views=views, risk_aversion=risk_aversion,
                returns=returns, rf=rf,
            )

        return await asyncio.to_thread(_bl_solve)

    return await asyncio.to_thread(_solve)


@router.post("/run", response_model=OptimizationResult)
async def optimize_run(
    body: OptimizeRunRequest,
    db: Session = Depends(get_db),
    data: DataService = Depends(get_data_service),
    regime_service: RegimeService | None = Depends(get_regime_service),
) -> OptimizationResult:
    weights = load_holdings(db, body.portfolio_id)
    tickers = list(weights.keys())
    constraints_dict = body.constraints.model_dump() if body.constraints else None
    views_dict = [v.model_dump() for v in body.views] if body.views else None

    regime_probs = None
    if body.method == "regime_blended":
        regime_probs = await _resolve_regime_probabilities(
            body.regime_probabilities, regime_service
        )

    result = await _run_optimization(
        body.method,
        tickers,
        data,
        target=body.target,
        constraints_dict=constraints_dict,
        views=views_dict,
        risk_aversion=body.risk_aversion,
        regime_probabilities=regime_probs,
    )

    db.add(
        OptimizationRun(
            portfolio_id=body.portfolio_id,
            method=body.method,
            target=body.target,
            result=result.model_dump(),
        )
    )
    db.commit()
    return result


@router.post("/stateless", response_model=OptimizationResult)
async def optimize_stateless(
    body: OptimizeStatelessRequest,
    data: DataService = Depends(get_data_service),
    regime_service: RegimeService | None = Depends(get_regime_service),
) -> OptimizationResult:
    tickers = [h.ticker for h in body.holdings]
    constraints_dict = body.constraints.model_dump() if body.constraints else None
    views_dict = [v.model_dump() for v in body.views] if body.views else None

    regime_probs = None
    if body.method == "regime_blended":
        regime_probs = await _resolve_regime_probabilities(
            body.regime_probabilities, regime_service
        )

    return await _run_optimization(
        body.method,
        tickers,
        data,
        target=body.target,
        constraints_dict=constraints_dict,
        views=views_dict,
        risk_aversion=body.risk_aversion,
        regime_probabilities=regime_probs,
    )


@router.get("/history/{portfolio_id}", response_model=list[OptimizationResult])
async def optimize_history(
    portfolio_id: uuid.UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[OptimizationResult]:
    rows = db.execute(
        select(OptimizationRun)
        .where(OptimizationRun.portfolio_id == portfolio_id)
        .order_by(desc(OptimizationRun.created_at))
        .limit(limit)
        .offset(offset)
    ).scalars().all()
    return [OptimizationResult(**row.result) for row in rows]
