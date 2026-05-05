"""OCR earnings pipeline backed by SEC EDGAR + NVIDIA NIM.

Responsibilities:
  1. fetch_and_process(ticker, form_type) - EDGAR lookup → PDF/HTML download →
     NIM OCR (vision) or HTML strip → LLM signal extraction → DB persist.
  2. get_latest_signals(tickers) - {ticker: EarningsSignal} for most recent
     record per ticker; absent when no record exists (never raises).
  3. get_signals(ticker) - ordered list of all records for a ticker.
  4. get_events_for_period(tickers, since_date) - EarningsEvent list used to
     annotate backtest equity curves.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
from datetime import date, datetime, timezone
from html.parser import HTMLParser

import httpx
import openai
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import EarningsDocument
from app.schemas.document import EarningsEvent, EarningsSignal


logger = logging.getLogger(__name__)

EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
)
EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
EDGAR_HEADERS = {
    "User-Agent": "QuantFusion/1.0 contact@quantfusion.app",
    "Accept-Encoding": "gzip, deflate",
}

CIK_CACHE_KEY = "edgar:tickers"
CIK_CACHE_TTL = 86_400  # 24 h
OCR_PAGE_CAP = 10

OCR_PAGE_TIMEOUT = 60.0       # per-page NIM vision call
SIGNAL_LLM_TIMEOUT = 45.0     # signal extraction LLM call
EDGAR_HTTP_TIMEOUT = 30.0
EDGAR_DOWNLOAD_TIMEOUT = 60.0


class TickerNotFoundError(Exception):
    pass


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _strip_html(raw: bytes) -> str:
    parser = _HTMLStripper()
    try:
        parser.feed(raw.decode("utf-8", errors="replace"))
    except Exception:
        pass
    return parser.get_text()


def _parse_signals_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (ValueError, json.JSONDecodeError):
            pass
    return {}


class OCRService:
    def __init__(self, llm_factory, session_factory: sessionmaker, redis=None):
        self._llm_factory = llm_factory
        self.session_factory = session_factory
        self.redis = redis

    @property
    def enabled(self) -> bool:
        return self._llm_factory() is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_and_process(
        self, ticker: str, form_type: str = "8-K"
    ) -> EarningsDocument:
        cik, filing_date, exhibit_url = await self._edgar_lookup(ticker, form_type)

        # Short-circuit: if we already processed this filing AND eps_beat is set
        # (i.e., the pipeline previously succeeded end-to-end), skip the
        # expensive OCR/LLM/yfinance work and return the cached row.
        existing = await asyncio.to_thread(
            self._lookup_complete_existing, ticker, filing_date
        )
        if existing is not None:
            return existing

        raw_bytes, is_pdf = await self._download_exhibit(exhibit_url)

        if is_pdf:
            text, page_count = await self._ocr_pdf(raw_bytes)
        else:
            text = _strip_html(raw_bytes)
            page_count = 1

        signals_dict = await self._extract_signals(ticker, text)
        signals_dict = await self._enrich_with_yfinance(ticker, filing_date, signals_dict)

        return await asyncio.to_thread(
            self._save_sync,
            ticker,
            filing_date,
            form_type,
            page_count,
            text,
            signals_dict,
        )

    def _lookup_complete_existing(
        self, ticker: str, filing_date: str | None
    ) -> EarningsDocument | None:
        """Return existing row only if signals are fully populated (eps_beat set).
        When None, caller must run the full pipeline."""
        if not filing_date:
            return None
        with self.session_factory() as db:
            row = (
                db.query(EarningsDocument)
                .filter(
                    EarningsDocument.ticker == ticker,
                    EarningsDocument.filing_date == filing_date,
                )
                .first()
            )
            if row is None:
                return None
            signals = row.signals or {}
            if signals.get("eps_beat") is None:
                return None
            return row

    async def get_latest_signals(
        self, tickers: list[str]
    ) -> dict[str, EarningsSignal]:
        if not tickers:
            return {}
        try:
            return await asyncio.to_thread(self._get_latest_signals_sync, tickers)
        except Exception:
            logger.exception("get_latest_signals failed")
            return {}

    async def get_signals(self, ticker: str) -> list[EarningsDocument]:
        return await asyncio.to_thread(self._get_signals_sync, ticker)

    async def get_events_for_period(
        self, tickers: list[str], since_date: date
    ) -> list[EarningsEvent]:
        if not tickers:
            return []
        try:
            return await asyncio.to_thread(
                self._get_events_sync, tickers, since_date
            )
        except Exception:
            logger.exception("get_events_for_period failed")
            return []

    # ------------------------------------------------------------------
    # EDGAR helpers
    # ------------------------------------------------------------------

    async def _edgar_lookup(
        self, ticker: str, form_type: str
    ) -> tuple[int, str, str]:
        """Return (cik, filing_date, exhibit_url)."""
        cik = await self._resolve_cik(ticker)
        async with httpx.AsyncClient(headers=EDGAR_HEADERS, timeout=30) as client:
            sub_url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
            resp = await client.get(sub_url)
            resp.raise_for_status()
            data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])

        # Find most recent matching form type
        for form, accession, filing_date in zip(forms, accessions, dates):
            if form == form_type:
                exhibit_url = await self._find_exhibit(cik, accession)
                return cik, filing_date, exhibit_url

        raise TickerNotFoundError(
            f"No {form_type} filings found for {ticker} (CIK {cik})"
        )

    async def _resolve_cik(self, ticker: str) -> int:
        ticker_upper = ticker.upper()

        # Try Redis cache first
        if self.redis is not None:
            try:
                cached = await self.redis.get(CIK_CACHE_KEY)
                if cached:
                    mapping: dict = json.loads(cached)
                    if ticker_upper in mapping:
                        return int(mapping[ticker_upper])
            except Exception:
                logger.warning("Redis read failed for CIK cache")

        # Fetch from EDGAR
        async with httpx.AsyncClient(headers=EDGAR_HEADERS, timeout=30) as client:
            resp = await client.get(EDGAR_TICKERS_URL)
            resp.raise_for_status()
            raw = resp.json()

        # Build {TICKER: cik} mapping
        mapping = {
            v["ticker"].upper(): v["cik_str"]
            for v in raw.values()
            if isinstance(v, dict) and "ticker" in v and "cik_str" in v
        }

        if self.redis is not None:
            try:
                await self.redis.set(
                    CIK_CACHE_KEY, json.dumps(mapping), ex=CIK_CACHE_TTL
                )
            except Exception:
                logger.warning("Redis write failed for CIK cache")

        if ticker_upper not in mapping:
            raise TickerNotFoundError(f"Ticker {ticker} not found in EDGAR")
        return int(mapping[ticker_upper])

    async def _find_exhibit(self, cik: int, accession: str) -> str:
        """Return URL for exhibit 99.1 or first HTML/PDF in the filing."""
        accession_clean = accession.replace("-", "")
        index_url = EDGAR_INDEX_URL.format(cik=cik, accession=accession_clean)
        base_url = EDGAR_ARCHIVE_BASE.format(cik=cik, accession=accession_clean)

        async with httpx.AsyncClient(headers=EDGAR_HEADERS, timeout=30) as client:
            resp = await client.get(index_url)
            resp.raise_for_status()
            index = resp.json()

        documents = index.get("directory", {}).get("item", [])

        # Prefer exhibit 99.1 (press release), then any .htm/.pdf
        preferred: str | None = None
        fallback: str | None = None
        for doc in documents:
            name = doc.get("name", "")
            desc = doc.get("description", "").lower()
            if "ex-99.1" in desc or "ex99" in name.lower():
                preferred = base_url + name
                break
            if fallback is None and (
                name.endswith(".htm") or name.endswith(".html") or name.endswith(".pdf")
            ):
                fallback = base_url + name

        url = preferred or fallback
        if url is None:
            raise TickerNotFoundError(f"No usable exhibit found in filing {accession}")
        return url

    async def _download_exhibit(self, url: str) -> tuple[bytes, bool]:
        async with httpx.AsyncClient(headers=EDGAR_HEADERS, timeout=60) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")
        return resp.content, is_pdf

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    async def _ocr_pdf(self, pdf_bytes: bytes) -> tuple[str, int]:
        """Return (extracted_text, total_page_count). Total count reflects the
        full PDF; OCR only runs on the first OCR_PAGE_CAP pages."""
        import pdf2image

        images = await asyncio.to_thread(
            pdf2image.convert_from_bytes,
            pdf_bytes,
            dpi=150,
            fmt="jpeg",
        )
        total_pages = len(images)
        pages = images[:OCR_PAGE_CAP]
        texts = await asyncio.gather(*[self._ocr_page(img) for img in pages])
        return "\n\n".join(t for t in texts if t), total_pages

    async def _ocr_page(self, img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        client = openai.AsyncOpenAI(
            base_url=settings.NIM_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            timeout=OCR_PAGE_TIMEOUT,
        )
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.NIM_OCR_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Extract all text from this document page verbatim.",
                                },
                            ],
                        }
                    ],
                    max_tokens=2048,
                ),
                timeout=OCR_PAGE_TIMEOUT,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("NIM OCR page call failed")
            return ""

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    async def _extract_signals(self, ticker: str, text: str) -> dict:
        llm = self._llm_factory()
        if llm is None:
            return {}

        snippet = text[:4000]
        prompt = (
            f"You are a financial analyst. Extract earnings signals for {ticker} "
            "from the following earnings document text.\n\n"
            f"Text:\n{snippet}\n\n"
            "Return a JSON object with exactly these keys (use null when not found):\n"
            "  eps_actual (float): reported EPS,\n"
            "  eps_estimate (float): analyst consensus EPS estimate if mentioned,\n"
            "  revenue_actual (float, in billions): reported revenue,\n"
            "  eps_beat (bool): true if eps_actual >= eps_estimate,\n"
            "  sentiment (string): 'positive', 'negative', or 'neutral' overall tone.\n\n"
            "Output ONLY a valid JSON object. No preamble, no explanation."
        )
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(prompt), timeout=SIGNAL_LLM_TIMEOUT
            )
            raw = (response.content or "").strip()
            return _parse_signals_json(raw)
        except Exception:
            logger.exception("signal extraction LLM call failed for %s", ticker)
            return {}

    # ------------------------------------------------------------------
    # DB helpers (sync, run in thread)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # yfinance enrichment
    # ------------------------------------------------------------------

    async def _enrich_with_yfinance(
        self, ticker: str, filing_date: str, signals: dict
    ) -> dict:
        """Fill in eps_estimate, eps_beat, revenue_beat using yfinance.

        EDGAR press releases contain actuals; analyst consensus estimates come
        from yfinance.earnings_dates.  Runs in a thread (yfinance is sync).
        Fails silently - original signals dict is returned unchanged on error.
        """
        try:
            return await asyncio.to_thread(
                self._yf_enrich_sync, ticker, filing_date, signals
            )
        except Exception:
            logger.warning("yfinance enrichment failed for %s", ticker, exc_info=True)
            return signals

    @staticmethod
    def _yf_enrich_sync(ticker: str, filing_date: str, signals: dict) -> dict:
        import pandas as pd
        import yfinance as yf

        enriched = dict(signals)

        try:
            fd = pd.Timestamp(filing_date)
        except Exception:
            return enriched

        t = yf.Ticker(ticker)

        # --- primary path: earnings_dates (scrapes Yahoo Finance HTML table) ---
        row = None
        try:
            df = t.earnings_dates
            if df is not None and not df.empty:
                import numpy as np
                idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
                # explicit loop avoids TimedeltaIndex.abs() compatibility issues
                abs_secs = np.array([abs((ts - fd).total_seconds()) for ts in idx])
                closest_pos = int(np.argmin(abs_secs))
                if abs_secs[closest_pos] <= 14 * 86_400:
                    row = df.iloc[closest_pos]
        except Exception:
            # Yahoo Finance page structure changes break the scraper - fall through
            pass

        if row is not None:
            # EPS estimate
            eps_est_raw = row.get("EPS Estimate")
            if enriched.get("eps_estimate") is None and eps_est_raw is not None:
                try:
                    v = float(eps_est_raw)
                    if not pd.isna(v):
                        enriched["eps_estimate"] = v
                except (TypeError, ValueError):
                    pass

            # Surprise % → override "neutral" sentiment
            surprise_raw = row.get("Surprise(%)")
            if surprise_raw is not None and enriched.get("sentiment") == "neutral":
                try:
                    surprise = float(surprise_raw)
                    if not pd.isna(surprise):
                        if surprise > 3.0:
                            enriched["sentiment"] = "positive"
                        elif surprise < -3.0:
                            enriched["sentiment"] = "negative"
                except (TypeError, ValueError):
                    pass

        # --- fallback: calendar (dict, no HTML scraping, more reliable) ---
        # Only used when primary path couldn't find an estimate.
        if enriched.get("eps_estimate") is None:
            try:
                cal = t.calendar  # returns dict or None
                if isinstance(cal, dict):
                    est = cal.get("EPS Estimate") or cal.get("Earnings EPS Estimate")
                    if est is not None:
                        try:
                            enriched["eps_estimate"] = float(est)
                        except (TypeError, ValueError):
                            pass
            except Exception:
                pass

        # --- compute eps_beat once estimate is known ---
        if enriched.get("eps_beat") is None:
            eps_act = enriched.get("eps_actual")
            eps_est = enriched.get("eps_estimate")
            if eps_act is not None and eps_est is not None:
                enriched["eps_beat"] = bool(eps_act >= eps_est)

        return enriched

    def _save_sync(
        self,
        ticker: str,
        filing_date: str,
        form_type: str,
        pages: int,
        text: str,
        signals_dict: dict,
    ) -> EarningsDocument:
        """Upsert by (ticker, filing_date). When the row already exists, update
        in place; otherwise insert. Falls back to plain INSERT when filing_date
        is unknown (no dedupe key available)."""
        now = datetime.now(tz=timezone.utc)
        truncated = (text or "")[:50_000]  # guard against huge filings

        with self.session_factory() as db:
            existing = None
            if filing_date:
                existing = (
                    db.query(EarningsDocument)
                    .filter(
                        EarningsDocument.ticker == ticker,
                        EarningsDocument.filing_date == filing_date,
                    )
                    .first()
                )

            if existing is not None:
                existing.uploaded_at = now
                existing.form_type = form_type
                existing.pages = pages
                existing.ocr_text = truncated
                existing.signals = signals_dict
                doc = existing
            else:
                doc = EarningsDocument(
                    ticker=ticker,
                    uploaded_at=now,
                    filing_date=filing_date,
                    form_type=form_type,
                    pages=pages,
                    ocr_text=truncated,
                    signals=signals_dict,
                )
                db.add(doc)

            db.commit()
            db.refresh(doc)
        return doc

    def _get_latest_signals_sync(
        self, tickers: list[str]
    ) -> dict[str, EarningsSignal]:
        """Return the most recent filing per ticker, ordered by filing_date
        (not upload time) so a freshly fetched older 10-K cannot displace a
        more recent 8-K already in the DB."""
        result: dict[str, EarningsSignal] = {}
        with self.session_factory() as db:
            for ticker in tickers:
                row = (
                    db.query(EarningsDocument)
                    .filter(EarningsDocument.ticker == ticker)
                    .order_by(
                        EarningsDocument.filing_date.desc().nullslast(),
                        EarningsDocument.uploaded_at.desc(),
                    )
                    .first()
                )
                if row is not None and row.signals:
                    try:
                        result[ticker] = EarningsSignal(**row.signals)
                    except Exception:
                        logger.warning(
                            "failed to parse signals for %s", ticker, exc_info=True
                        )
        return result

    def _get_signals_sync(self, ticker: str) -> list[EarningsDocument]:
        with self.session_factory() as db:
            return (
                db.query(EarningsDocument)
                .filter(EarningsDocument.ticker == ticker)
                .order_by(EarningsDocument.uploaded_at.desc())
                .all()
            )

    def _get_events_sync(
        self, tickers: list[str], since_date: date
    ) -> list[EarningsEvent]:
        events: list[EarningsEvent] = []
        since_str = since_date.isoformat()
        with self.session_factory() as db:
            rows = (
                db.query(EarningsDocument)
                .filter(
                    EarningsDocument.ticker.in_(tickers),
                    EarningsDocument.filing_date >= since_str,
                )
                .order_by(
                    EarningsDocument.filing_date.asc(),
                    EarningsDocument.uploaded_at.desc(),
                )
                .all()
            )
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (row.ticker, row.filing_date or "")
            if key in seen:
                continue
            seen.add(key)
            signals = row.signals or {}
            events.append(
                EarningsEvent(
                    date=row.filing_date or "",
                    ticker=row.ticker,
                    eps_beat=signals.get("eps_beat"),
                    sentiment=signals.get("sentiment", "neutral"),
                )
            )
        return events
