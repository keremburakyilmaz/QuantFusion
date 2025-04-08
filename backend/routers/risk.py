from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from services.risk_engine import calculate_var

router = APIRouter()

class RiskRequest(BaseModel):
    asset_prices: List[float]
    confidence_level: float

@router.post("/var")
def calculate_var_router(request: RiskRequest):
    result = calculate_var(
        request.asset_prices,
        request.confidence_level
    )

    return result
