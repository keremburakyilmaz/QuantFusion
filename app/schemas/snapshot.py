from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.analyzer import AnalysisReport
from app.schemas.common import HoldingInput


class SaveSnapshotRequest(BaseModel):
    holdings: list[HoldingInput] = Field(min_length=1, max_length=20)
    expires_in_days: int = Field(30, ge=0, le=365)


class SaveSnapshotResponse(BaseModel):
    token: str
    expires_at: datetime | None = None
    share_url: str


class SnapshotResponse(BaseModel):
    token: str
    holdings: list[HoldingInput]
    report: AnalysisReport
    created_at: datetime
    expires_at: datetime | None = None
