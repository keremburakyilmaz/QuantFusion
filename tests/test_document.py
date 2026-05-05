
import json
import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.document import EarningsSignal
from app.services.ocr_service import OCRService, TickerNotFoundError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

FAKE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q"],
            "accessionNumber": ["0000320193-24-000123", "0000320193-24-000099"],
            "filingDate": ["2024-02-01", "2024-01-15"],
        }
    }
}

FAKE_INDEX = {
    "directory": {
        "item": [
            {"name": "exhibit99-1.htm", "description": "EX-99.1"},
            {"name": "form8k.htm", "description": "8-K"},
        ]
    }
}

FAKE_HTML_CONTENT = b"""
<html><body>
<p>Q4 2024 Results</p>
<p>EPS: $2.40 actual vs $2.10 estimate</p>
<p>Revenue: $119.6B vs $117.9B estimate</p>
<p>Guidance raised for next quarter</p>
</body></html>
"""

FAKE_SIGNALS_DICT = {
    "eps_actual": 2.40,
    "eps_estimate": 2.10,
    "revenue_actual": 119.6,
    "eps_beat": True,
    "sentiment": "positive",
}


def _make_llm(response_text: str):
    async def ainvoke(_prompt):
        return SimpleNamespace(content=response_text)
    return SimpleNamespace(ainvoke=ainvoke)


def _make_session_factory(docs=None):
    """Return a mock session_factory that returns no stored docs by default."""
    docs = docs or []
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.filter.return_value.order_by.return_value = mock_query
    mock_query.filter.return_value.order_by.return_value.first.return_value = None
    mock_query.filter.return_value.order_by.return_value.all.return_value = docs
    mock_query.first.return_value = None
    mock_query.all.return_value = docs
    mock_session.query.return_value = mock_query
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    factory = MagicMock(return_value=mock_session)
    return factory, mock_session


def _make_http_responses():
    """Return side_effect list for httpx.AsyncClient.get."""

    async def _get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "company_tickers" in url:
            resp.json = MagicMock(return_value=FAKE_TICKERS_JSON)
        elif "submissions" in url:
            resp.json = MagicMock(return_value=FAKE_SUBMISSIONS)
        elif "index.json" in url:
            resp.json = MagicMock(return_value=FAKE_INDEX)
        elif "exhibit" in url or ".htm" in url:
            resp.content = FAKE_HTML_CONTENT
            resp.headers = {"content-type": "text/html"}
        else:
            resp.content = FAKE_HTML_CONTENT
            resp.headers = {"content-type": "text/html"}
        return resp

    return _get


# ---------------------------------------------------------------------------
# Tests: fetch_and_process (HTML path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_and_process_html_path():
    """Happy path: EDGAR returns HTML exhibit - no pdf2image/OCR needed."""
    llm = _make_llm(json.dumps(FAKE_SIGNALS_DICT))
    session_factory, mock_session = _make_session_factory()

    # Capture what gets saved
    saved_doc = None

    def _add(doc):
        nonlocal saved_doc
        saved_doc = doc

    mock_session.add = MagicMock(side_effect=_add)
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()

    svc = OCRService(llm_factory=lambda: llm, session_factory=session_factory)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=_make_http_responses())
        mock_client_cls.return_value = mock_client

        with patch.object(svc, "_save_sync") as mock_save:
            from app.models import EarningsDocument
            fake_doc = EarningsDocument(
                id=uuid.uuid4(),
                ticker="AAPL",
                uploaded_at=datetime.now(tz=timezone.utc),
                filing_date="2024-02-01",
                form_type="8-K",
                pages=1,
                signals=FAKE_SIGNALS_DICT,
            )
            mock_save.return_value = fake_doc

            doc = await svc.fetch_and_process("AAPL", form_type="8-K")

    assert doc.ticker == "AAPL"
    assert doc.filing_date == "2024-02-01"
    assert doc.signals == FAKE_SIGNALS_DICT


@pytest.mark.asyncio
async def test_ticker_not_found_raises():
    """Unknown ticker should raise TickerNotFoundError."""
    llm = _make_llm("{}")
    session_factory, _ = _make_session_factory()
    svc = OCRService(llm_factory=lambda: llm, session_factory=session_factory)

    tickers_without_xyz = {}  # empty, so EDGAR has no entry

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def _get(url, **kw):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=tickers_without_xyz)
            return resp

        mock_client.get = AsyncMock(side_effect=_get)
        mock_client_cls.return_value = mock_client

        with pytest.raises(TickerNotFoundError):
            await svc.fetch_and_process("XYZ_UNKNOWN")


# ---------------------------------------------------------------------------
# Tests: get_latest_signals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_latest_signals_empty_for_unknown_ticker():
    """Tickers with no stored records are absent from the result dict."""
    session_factory, _ = _make_session_factory(docs=[])
    svc = OCRService(llm_factory=lambda: None, session_factory=session_factory)
    result = await svc.get_latest_signals(["AAPL", "MSFT"])
    assert result == {}


@pytest.mark.asyncio
async def test_get_latest_signals_returns_signal():
    """Stored record is returned as EarningsSignal."""
    from app.models import EarningsDocument

    fake_doc = MagicMock(spec=EarningsDocument)
    fake_doc.ticker = "AAPL"
    fake_doc.signals = FAKE_SIGNALS_DICT

    session_factory, mock_session = _make_session_factory()
    # Make .first() return our fake doc for AAPL
    mock_session.query.return_value.filter.return_value \
        .order_by.return_value.first.return_value = fake_doc

    svc = OCRService(llm_factory=lambda: None, session_factory=session_factory)
    result = await svc.get_latest_signals(["AAPL"])

    assert "AAPL" in result
    sig = result["AAPL"]
    assert isinstance(sig, EarningsSignal)
    assert sig.eps_actual == 2.40
    assert sig.eps_beat is True
    assert sig.sentiment == "positive"


# ---------------------------------------------------------------------------
# Tests: OCR + signal extraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_signals_parses_llm_json():
    """Signal extraction correctly parses clean JSON from LLM."""
    llm = _make_llm(json.dumps(FAKE_SIGNALS_DICT))
    session_factory, _ = _make_session_factory()
    svc = OCRService(llm_factory=lambda: llm, session_factory=session_factory)

    result = await svc._extract_signals("AAPL", "some earnings text")

    assert result["eps_beat"] is True
    assert result["sentiment"] == "positive"


@pytest.mark.asyncio
async def test_extract_signals_returns_empty_on_llm_failure():
    """Returns empty dict gracefully when LLM call fails."""
    async def boom(_p):
        raise RuntimeError("network down")

    llm = SimpleNamespace(ainvoke=boom)
    session_factory, _ = _make_session_factory()
    svc = OCRService(llm_factory=lambda: llm, session_factory=session_factory)

    result = await svc._extract_signals("AAPL", "text")
    assert result == {}


@pytest.mark.asyncio
async def test_extract_signals_no_llm_returns_empty():
    """Returns empty dict when LLM is not configured."""
    session_factory, _ = _make_session_factory()
    svc = OCRService(llm_factory=lambda: None, session_factory=session_factory)
    result = await svc._extract_signals("AAPL", "text")
    assert result == {}


# ---------------------------------------------------------------------------
# Tests: earnings_tilt optimizer
# ---------------------------------------------------------------------------

def test_earnings_tilt_positive_boosts_weight():
    """Positive earnings signal increases the ticker's weight."""
    import numpy as np
    import pandas as pd
    from app.services.optimizer import PortfolioOptimizer

    rng = np.random.default_rng(42)
    n_days = 500
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, (n_days, 2)),
        columns=["AAPL", "MSFT"],
    )
    cov = returns.cov().values * 252
    regime_proba = {"bull": 0.7, "sideways": 0.2, "bear": 0.1}

    optimizer = PortfolioOptimizer()
    base = optimizer.regime_blended(returns, cov, regime_proba)

    earnings = {
        "AAPL": {"eps_beat": True, "sentiment": "positive"},
        "MSFT": {"eps_beat": False, "sentiment": "negative"},
    }
    tilted = optimizer.earnings_tilt(returns, cov, regime_proba, earnings=earnings)

    # AAPL should be boosted, MSFT reduced
    assert tilted.weights["AAPL"] > base.weights["AAPL"]
    assert tilted.weights["MSFT"] < base.weights["MSFT"]
    # Weights still sum to 1
    assert abs(sum(tilted.weights.values()) - 1.0) < 1e-6
    assert tilted.method == "earnings_tilt"


def test_earnings_tilt_no_signals_matches_regime_blended():
    """With no signals, earnings_tilt should produce the same weights as regime_blended."""
    import numpy as np
    import pandas as pd
    from app.services.optimizer import PortfolioOptimizer

    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, (400, 2)),
        columns=["AAPL", "MSFT"],
    )
    cov = returns.cov().values * 252
    regime_proba = {"bull": 0.6, "sideways": 0.3, "bear": 0.1}

    optimizer = PortfolioOptimizer()
    base = optimizer.regime_blended(returns, cov, regime_proba)
    tilted = optimizer.earnings_tilt(returns, cov, regime_proba, earnings=None)

    for t in ["AAPL", "MSFT"]:
        assert abs(tilted.weights[t] - base.weights[t]) < 1e-6


# ---------------------------------------------------------------------------
# Tests: HTML stripping
# ---------------------------------------------------------------------------

def test_strip_html_extracts_text():
    from app.services.ocr_service import _strip_html

    html = b"<html><body><p>EPS: $2.40</p><p>Revenue beat</p></body></html>"
    text = _strip_html(html)
    assert "EPS: $2.40" in text
    assert "Revenue beat" in text


def test_strip_html_handles_empty():
    from app.services.ocr_service import _strip_html

    assert _strip_html(b"") == ""
