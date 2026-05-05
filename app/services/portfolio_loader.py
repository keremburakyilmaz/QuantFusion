

import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Holding


def load_holdings(db: Session, portfolio_id: uuid.UUID) -> dict[str, float]:
    rows = db.execute(
        select(Holding.ticker, Holding.target_weight).where(
            Holding.portfolio_id == portfolio_id
        )
    ).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    weights = {ticker: float(w) for ticker, w in rows if w is not None}
    if not weights:
        raise HTTPException(
            status_code=422, detail="Portfolio has no target weights set"
        )
    return weights
