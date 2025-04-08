from fastapi import APIRouter
from pydantic import BaseModel
from services.sentiment_analysis import analyze_sentiment

router = APIRouter()

class SentimentRequest(BaseModel):
    keyword: str
    source: str  # e.g., 'twitter', 'reddit'

@router.post("/analyze")
def analyze_sentiment_router(request: SentimentRequest):
    result = analyze_sentiment(
        request.keyword, 
        request.source
    )

    return result
