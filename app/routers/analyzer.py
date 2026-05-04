from fastapi import APIRouter, Depends, HTTPException, Request

from app.limiter import limiter
from app.schemas.analyzer import (
    AnalysisReport,
    AnalyzerRunRequest,
    AnalyzerValidateRequest,
    AnalyzerValidateResponse,
)
from app.services.analyzer_service import AnalyzerService


router = APIRouter()


def get_analyzer(request: Request) -> AnalyzerService:
    svc: AnalyzerService | None = getattr(request.app.state, "analyzer_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    return svc


@router.post("/validate", response_model=AnalyzerValidateResponse)
@limiter.limit("30/minute")
async def validate(
    request: Request,
    body: AnalyzerValidateRequest,
    analyzer: AnalyzerService = Depends(get_analyzer),
) -> AnalyzerValidateResponse:
    tickers = [t.strip().upper() for t in body.tickers if t.strip()]
    result = await analyzer.validate(tickers)
    return AnalyzerValidateResponse(valid=result["valid"], invalid=result["invalid"])


@router.post("/run", response_model=AnalysisReport)
@limiter.limit("3/minute")
async def run(
    request: Request,
    body: AnalyzerRunRequest,
    analyzer: AnalyzerService = Depends(get_analyzer),
) -> AnalysisReport:
    try:
        return await analyzer.run(body.holdings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
