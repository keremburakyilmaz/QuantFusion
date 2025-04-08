from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class RiskRequest(BaseModel):
    asset_prices: List[float]
    confidence_level: float

@router.post("/var")
def calculate_var(request: RiskRequest):
    return {
        "confidence_level": request.confidence_level,
        "VaR": None,
        "volatility": None
    }
