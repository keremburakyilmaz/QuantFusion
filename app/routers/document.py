
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.limiter import limiter
from app.schemas.document import (
    EarningsFetchRequest,
    EarningsSignalRecord,
    EarningsSignalsResponse,
    EarningsUploadResponse,
)
from app.services.ocr_service import OCRService, TickerNotFoundError


logger = logging.getLogger(__name__)

router = APIRouter()


def get_ocr_service(request: Request) -> OCRService:
    svc: OCRService | None = getattr(request.app.state, "ocr_service", None)
    if svc is None or not svc.enabled:
        raise HTTPException(
            status_code=503,
            detail="OCR service unavailable; NVIDIA_API_KEY not configured",
        )
    return svc


@router.post("/earnings", response_model=EarningsUploadResponse)
@limiter.limit("5/minute")
async def fetch_earnings(
    request: Request,
    body: EarningsFetchRequest,
    svc: OCRService = Depends(get_ocr_service),
) -> EarningsUploadResponse:
    """Fetch the latest earnings filing for a ticker from SEC EDGAR,
    run OCR via NVIDIA NIM, extract structured signals, and persist."""
    ticker = body.ticker.upper().strip()
    try:
        doc = await svc.fetch_and_process(ticker, form_type=body.form_type)
    except TickerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("earnings fetch failed for %s", ticker)
        raise HTTPException(
            status_code=502,
            detail=f"EDGAR or OCR pipeline failed: {exc}",
        )

    from app.schemas.document import EarningsSignal

    signals = EarningsSignal(**(doc.signals or {}))
    return EarningsUploadResponse(
        id=str(doc.id),
        ticker=doc.ticker,
        uploaded_at=doc.uploaded_at,
        filing_date=doc.filing_date,
        form_type=doc.form_type or "8-K",
        pages=doc.pages or 0,
        signals=signals,
    )


@router.get("/signals/{ticker}", response_model=EarningsSignalsResponse)
@limiter.limit("30/minute")
async def get_signals(
    request: Request,
    ticker: str,
    svc: OCRService = Depends(get_ocr_service),
) -> EarningsSignalsResponse:
    """Return all stored earnings signal records for a ticker, newest first."""
    ticker = ticker.upper().strip()
    docs = await svc.get_signals(ticker)

    from app.schemas.document import EarningsSignal

    records = [
        EarningsSignalRecord(
            id=str(doc.id),
            ticker=doc.ticker,
            uploaded_at=doc.uploaded_at,
            filing_date=doc.filing_date,
            signals=EarningsSignal(**(doc.signals or {})),
        )
        for doc in docs
    ]
    return EarningsSignalsResponse(ticker=ticker, records=records)
