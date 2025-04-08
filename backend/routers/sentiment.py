from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class SentimentRequest(BaseModel):
    keyword: str
    source: str  # e.g., 'twitter', 'reddit'

@router.post("/analyze")
def analyze_sentiment(request: SentimentRequest):
    return {
        "keyword": request.keyword,
        "source": request.source,
        "sentiment_score": None,
        "top_comments": []
    }
