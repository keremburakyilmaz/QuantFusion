

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import desc, select

from app.database import SessionLocal
from app.models import RegimeSnapshot
from app.schemas.regime import RegimeHistoryResponse, RegimeSnapshotResponse
from app.services.regime_service import RegimeService


router = APIRouter()


def get_regime_service(request: Request) -> RegimeService:
    return request.app.state.regime_service


@router.get("/current", response_model=RegimeSnapshotResponse)
async def get_current(request: Request) -> RegimeSnapshotResponse:
    with SessionLocal() as db:
        latest = db.execute(
            select(RegimeSnapshot).order_by(desc(RegimeSnapshot.ts)).limit(1)
        ).scalar_one_or_none()

    if latest is not None:
        feats = dict(latest.features or {})
        proba = feats.pop("_probabilities", None)
        return RegimeSnapshotResponse(
            ts=latest.ts,
            regime=latest.regime,
            confidence=float(latest.confidence) if latest.confidence else 0.0,
            features=feats,
            probabilities=proba,
        )

    service = get_regime_service(request)
    snapshot = await service.predict_current()
    if snapshot is None:
        raise HTTPException(
            status_code=503,
            detail="Regime model not yet available — try again shortly",
        )
    return snapshot


@router.get("/history", response_model=RegimeHistoryResponse)
async def get_history(
    days: int = Query(730, ge=1, le=3650),
) -> RegimeHistoryResponse:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    with SessionLocal() as db:
        rows = db.execute(
            select(RegimeSnapshot)
            .where(RegimeSnapshot.ts >= cutoff)
            .order_by(desc(RegimeSnapshot.ts))
        ).scalars().all()
    snapshots: list[RegimeSnapshotResponse] = []
    for r in rows:
        feats = dict(r.features or {})
        proba = feats.pop("_probabilities", None)
        snapshots.append(
            RegimeSnapshotResponse(
                ts=r.ts,
                regime=r.regime,
                confidence=float(r.confidence) if r.confidence else 0.0,
                features=feats,
                probabilities=proba,
            )
        )
    return RegimeHistoryResponse(snapshots=snapshots)
