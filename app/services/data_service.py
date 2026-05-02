"""Single source of truth for market data.

Cache-first: Redis (intraday) -> Postgres (historical) -> yfinance (fallback).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker

from app.models import PriceHistory


logger = logging.getLogger(__name__)

CACHE_TTL_PRICES = 3600          # 1 hour
CACHE_TTL_FUNDAMENTALS = 86400   # 24 hours
LOOKBACK_DEFAULT = 756           # ~3 years of trading days


def _prices_key(ticker: str, lookback_days: int) -> str:
    return f"prices:{ticker}:{lookback_days}"


def _fundamentals_key(ticker: str) -> str:
    return f"fundamentals:{ticker}"


class DataService:
    def __init__(self, redis: Redis, session_factory: sessionmaker):
        self.redis = redis
        self.session_factory = session_factory

    # ---------- prices ----------

    async def get_prices(
        self, tickers: list[str], lookback_days: int = LOOKBACK_DEFAULT
    ) -> pd.DataFrame:
        """Return wide DataFrame: index=date, columns=tickers, values=adj_close."""
        series: dict[str, pd.Series] = {}
        for ticker in tickers:
            df = await self._get_one_ticker(ticker, lookback_days)
            if df is not None and not df.empty:
                series[ticker] = df.set_index("ts")["adj_close"].astype(float)

        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1).sort_index().ffill().dropna(how="all")

    async def _get_one_ticker(
        self, ticker: str, lookback_days: int
    ) -> pd.DataFrame | None:
        cached = await self.redis.get(_prices_key(ticker, lookback_days))
        if cached:
            return _deserialize_price_df(cached)

        db_df = await asyncio.to_thread(self._read_prices_from_db, ticker, lookback_days)
        if db_df is not None and _is_fresh(db_df):
            await self.redis.set(
                _prices_key(ticker, lookback_days),
                _serialize_price_df(db_df),
                ex=CACHE_TTL_PRICES,
            )
            return db_df

        fetched = await asyncio.to_thread(_yf_download, ticker, lookback_days)
        if fetched is None or fetched.empty:
            return db_df

        await asyncio.to_thread(self._upsert_prices, ticker, fetched)
        await self.redis.set(
            _prices_key(ticker, lookback_days),
            _serialize_price_df(fetched),
            ex=CACHE_TTL_PRICES,
        )
        return fetched

    def _read_prices_from_db(
        self, ticker: str, lookback_days: int
    ) -> pd.DataFrame | None:
        with self.session_factory() as db:
            stmt = (
                select(
                    PriceHistory.ts,
                    PriceHistory.close,
                    PriceHistory.adj_close,
                    PriceHistory.volume,
                )
                .where(PriceHistory.ticker == ticker)
                .order_by(PriceHistory.ts.desc())
                .limit(lookback_days)
            )
            rows = db.execute(stmt).all()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["ts", "close", "adj_close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"])
        df["close"] = df["close"].astype(float)
        df["adj_close"] = df["adj_close"].astype(float)
        return df.sort_values("ts").reset_index(drop=True)

    def _upsert_prices(self, ticker: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        records = [
            {
                "ticker": ticker,
                "ts": row["ts"].date() if isinstance(row["ts"], pd.Timestamp) else row["ts"],
                "close": float(row["close"]),
                "adj_close": float(row["adj_close"]) if pd.notna(row["adj_close"]) else None,
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
            }
            for _, row in df.iterrows()
        ]
        with self.session_factory() as db:
            stmt = pg_insert(PriceHistory).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "ts"],
                set_={
                    "close": stmt.excluded.close,
                    "adj_close": stmt.excluded.adj_close,
                    "volume": stmt.excluded.volume,
                },
            )
            db.execute(stmt)
            db.commit()

    async def get_prices_detail(
        self, tickers: list[str], lookback_days: int = LOOKBACK_DEFAULT
    ) -> dict[str, pd.DataFrame]:
        """Return per-ticker DataFrames with columns: ts, close, adj_close, volume."""
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = await self._get_one_ticker(ticker, lookback_days)
            if df is not None and not df.empty:
                result[ticker] = df.sort_values("ts").reset_index(drop=True)
        return result

    # ---------- returns / covariance ----------

    async def get_returns(
        self, tickers: list[str], lookback_days: int = LOOKBACK_DEFAULT
    ) -> pd.DataFrame:
        prices = await self.get_prices(tickers, lookback_days)
        if prices.empty:
            return prices
        return prices.pct_change().dropna()

    async def covariance_matrix(
        self, tickers: list[str], lookback_days: int = 252
    ) -> np.ndarray:
        returns = await self.get_returns(tickers, lookback_days)
        if returns.empty:
            return np.empty((0, 0))
        return returns.cov().values * 252

    # alias to match spec §5 wording
    get_covariance_matrix = covariance_matrix

    # ---------- fundamentals + validation ----------

    async def get_fundamentals(self, ticker: str) -> dict:
        cached = await self.redis.get(_fundamentals_key(ticker))
        if cached:
            return json.loads(cached)

        info = await asyncio.to_thread(_yf_info, ticker)
        fundamentals = _extract_fundamentals(info)
        await self.redis.set(
            _fundamentals_key(ticker),
            json.dumps(fundamentals, default=str),
            ex=CACHE_TTL_FUNDAMENTALS,
        )
        return fundamentals

    async def validate_tickers(self, tickers: list[str]) -> dict[str, list[str]]:
        results = await asyncio.gather(
            *[asyncio.to_thread(_yf_is_valid, t) for t in tickers]
        )
        valid = [t for t, ok in zip(tickers, results) if ok]
        invalid = [t for t, ok in zip(tickers, results) if not ok]
        return {"valid": valid, "invalid": invalid}


# ---------- module-level helpers (mock points for tests) ----------


def _yf_download(ticker: str, lookback_days: int) -> pd.DataFrame | None:
    period = _period_for_lookback(lookback_days)
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.warning("yfinance download failed for %s: %s", ticker, exc)
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(
        columns={"Date": "ts", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
    )
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["adj_close"] = df["adj_close"].fillna(df["close"])
    return df[["ts", "close", "adj_close", "volume"]]


def _yf_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("yfinance .info failed for %s: %s", ticker, exc)
        return {}


def _yf_is_valid(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).fast_info
        # fast_info is a dict-like; valid tickers expose last_price / market_cap
        return bool(info) and (
            getattr(info, "last_price", None) is not None
            or getattr(info, "market_cap", None) is not None
        )
    except Exception:
        return False


def _extract_fundamentals(info: dict) -> dict:
    return {
        "dividend_yield": info.get("dividendYield"),
        "trailing_pe": info.get("trailingPE"),
        "market_cap": info.get("marketCap"),
        "beta": info.get("beta"),
        "next_earnings_date": _format_earnings_date(info.get("earningsDate")),
        "sector": info.get("sector"),
    }


def _format_earnings_date(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _period_for_lookback(lookback_days: int) -> str:
    if lookback_days <= 252:
        return "1y"
    if lookback_days <= 504:
        return "2y"
    if lookback_days <= 756:
        return "3y"
    if lookback_days <= 1260:
        return "5y"
    return "max"


def _is_fresh(df: pd.DataFrame, max_age_days: int = 1) -> bool:
    last_ts = df["ts"].max()
    if isinstance(last_ts, pd.Timestamp):
        last_ts = last_ts.date()
    return (date.today() - last_ts) <= timedelta(days=max_age_days + 2)


def _serialize_price_df(df: pd.DataFrame) -> str:
    payload = []
    for _, row in df.iterrows():
        ts = row["ts"]
        if isinstance(ts, pd.Timestamp):
            ts = ts.date().isoformat()
        elif isinstance(ts, (datetime, date)):
            ts = ts.isoformat() if isinstance(ts, datetime) else ts.isoformat()
        payload.append(
            {
                "ts": ts,
                "close": float(row["close"]) if pd.notna(row["close"]) else None,
                "adj_close": float(row["adj_close"]) if pd.notna(row["adj_close"]) else None,
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
            }
        )
    return json.dumps(payload)


def _deserialize_price_df(blob: str) -> pd.DataFrame:
    records = json.loads(blob)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)
