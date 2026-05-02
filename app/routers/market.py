from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.schemas.market import (
    FundamentalsResponse,
    PricePoint,
    PriceSeriesResponse,
    ValidateRequest,
    ValidateResponse,
)
from app.services.data_service import DataService, LOOKBACK_DEFAULT


router = APIRouter()


def get_data_service(request: Request) -> DataService:
    return request.app.state.data_service


def _parse_tickers(raw: str) -> list[str]:
    parts = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not parts:
        raise HTTPException(status_code=400, detail="No tickers provided")
    return parts


@router.get("/prices", response_model=list[PriceSeriesResponse])
async def get_prices(
    tickers: str = Query(..., description="Comma-separated ticker list, e.g. JEPI,VOO"),
    lookback_days: int = Query(LOOKBACK_DEFAULT, ge=1, le=2520),
    data: DataService = Depends(get_data_service),
) -> list[PriceSeriesResponse]:
    ticker_list = _parse_tickers(tickers)
    detail = await data.get_prices_detail(ticker_list, lookback_days=lookback_days)
    if not detail:
        return []

    out: list[PriceSeriesResponse] = []
    for ticker in ticker_list:
        if ticker not in detail:
            continue
        ticker_df = detail[ticker]
        points = [
            PricePoint(
                ts=_to_date(row["ts"]),
                close=float(row["close"]),
                adj_close=float(row["adj_close"]) if pd.notna(row["adj_close"]) else float(row["close"]),
                volume=int(row["volume"]) if pd.notna(row["volume"]) else None,
            )
            for _, row in ticker_df.iterrows()
        ]
        out.append(PriceSeriesResponse(ticker=ticker, prices=points))
    return out


@router.post("/validate", response_model=ValidateResponse)
async def validate(
    body: ValidateRequest,
    data: DataService = Depends(get_data_service),
) -> ValidateResponse:
    tickers = [t.strip().upper() for t in body.tickers if t.strip()]
    result = await data.validate_tickers(tickers)
    return ValidateResponse(valid=result["valid"], invalid=result["invalid"])


@router.get("/fundamentals/{ticker}", response_model=FundamentalsResponse)
async def get_fundamentals(
    ticker: str,
    data: DataService = Depends(get_data_service),
) -> FundamentalsResponse:
    info = await data.get_fundamentals(ticker.upper())
    return FundamentalsResponse(ticker=ticker.upper(), **info)


def _to_date(idx) -> date:
    if isinstance(idx, pd.Timestamp):
        return idx.date()
    if isinstance(idx, date):
        return idx
    return pd.to_datetime(idx).date()
