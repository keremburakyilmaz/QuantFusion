from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class PricePoint(BaseModel):
    ts: date
    close: float
    adj_close: float | None = None
    volume: int | None = None


class PriceSeriesResponse(BaseModel):
    ticker: str
    prices: list[PricePoint]


class ValidateRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)


class ValidateResponse(BaseModel):
    valid: list[str]
    invalid: list[str]


class FundamentalsResponse(BaseModel):
    ticker: str
    dividend_yield: float | None = None
    trailing_pe: float | None = None
    market_cap: float | None = None
    beta: float | None = None
    next_earnings_date: str | None = None
    sector: str | None = None
