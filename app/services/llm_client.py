"""Lazy ChatOpenAI factory pointed at the NVIDIA NIM gateway.

Returns None when NVIDIA_API_KEY is unset so callers can degrade gracefully
(analyzer skips commentary, /api/agent/query returns 503).

Model + base URL are read from settings (NIM_MODEL, NIM_BASE_URL) so you can
A/B test models via .env without code changes.
"""


import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import settings


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI | None:
    key = (settings.NVIDIA_API_KEY or "").strip()
    if not key or key.startswith("nvapi-replace"):
        logger.info("NVIDIA_API_KEY not configured; LLM features disabled")
        return None
    logger.info("LLM enabled: model=%s base=%s", settings.NIM_MODEL, settings.NIM_BASE_URL)
    return ChatOpenAI(
        base_url=settings.NIM_BASE_URL,
        api_key=key,
        model=settings.NIM_MODEL,
        temperature=0.2,
        max_tokens=512,
    )
