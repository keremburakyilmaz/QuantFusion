
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class EarningsSignal(BaseModel):
    eps_actual: float | None = None
    eps_estimate: float | None = None
    revenue_actual: float | None = None      # billions USD
    eps_beat: bool | None = None
    sentiment: str = "neutral"               # "positive" | "negative" | "neutral"


class EarningsEvent(BaseModel):
    date: str
    ticker: str
    eps_beat: bool | None = None
    sentiment: str = "neutral"


class EarningsFetchRequest(BaseModel):
    ticker: str
    form_type: Literal["8-K", "10-Q", "10-K"] = "8-K"


class EarningsUploadResponse(BaseModel):
    id: str
    ticker: str
    uploaded_at: datetime
    filing_date: str | None
    form_type: str
    pages: int
    signals: EarningsSignal


class EarningsSignalRecord(BaseModel):
    id: str
    ticker: str
    uploaded_at: datetime
    filing_date: str | None
    signals: EarningsSignal


class EarningsSignalsResponse(BaseModel):
    ticker: str
    records: list[EarningsSignalRecord]
