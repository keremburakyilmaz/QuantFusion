

import uuid
from typing import Any

from pydantic import BaseModel, Field


class AgentQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    portfolio_id: uuid.UUID


class AgentQueryResponse(BaseModel):
    response: str
    intent: str
    data: dict[str, Any] | None = None
