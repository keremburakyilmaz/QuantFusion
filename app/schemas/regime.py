

from datetime import datetime

from pydantic import BaseModel


class RegimeSnapshotResponse(BaseModel):
    ts: datetime
    regime: str
    confidence: float
    features: dict[str, float] | None = None
    probabilities: dict[str, float] | None = None


class RegimeHistoryResponse(BaseModel):
    snapshots: list[RegimeSnapshotResponse]
