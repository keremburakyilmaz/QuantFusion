

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from app.services import data_service as ds_mod
from app.services.data_service import DataService, _prices_key


def _fake_price_df(days: int = 10) -> pd.DataFrame:
    today = date.today()
    rows = []
    for i in range(days):
        rows.append(
            {
                "ts": pd.Timestamp(today - timedelta(days=days - i - 1)),
                "close": 100.0 + i,
                "adj_close": 100.0 + i,
                "volume": 1_000_000 + i,
            }
        )
    return pd.DataFrame(rows)


class _StubSessionFactory:
    def __call__(self):
        return _StubSession()


class _StubSession:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_args, **_kwargs):
        class _Result:
            def all(self):
                return []

        return _Result()

    def commit(self):
        pass

    def close(self):
        pass


@pytest.mark.asyncio
async def test_get_prices_cache_hit_skips_yfinance(fake_redis):
    df = _fake_price_df(5)
    service = DataService(fake_redis, _StubSessionFactory())
    fake_redis.store[_prices_key("AAPL", 756)] = ds_mod._serialize_price_df(df)

    with patch.object(ds_mod, "_yf_download") as mock_dl:
        result = await service.get_prices(["AAPL"], lookback_days=756)

    mock_dl.assert_not_called()
    assert "AAPL" in result.columns
    assert len(result) == 5


@pytest.mark.asyncio
async def test_get_prices_yfinance_fallback_writes_cache(fake_redis):
    service = DataService(fake_redis, _StubSessionFactory())
    fetched = _fake_price_df(8)

    with patch.object(ds_mod, "_yf_download", return_value=fetched) as mock_dl, \
         patch.object(DataService, "_upsert_prices", return_value=None) as mock_upsert:
        result = await service.get_prices(["AAPL"], lookback_days=756)

    mock_dl.assert_called_once_with("AAPL", 756)
    mock_upsert.assert_called_once()
    assert _prices_key("AAPL", 756) in fake_redis.store
    assert "AAPL" in result.columns
    assert len(result) == 8


@pytest.mark.asyncio
async def test_validate_tickers_splits_valid_invalid(fake_redis):
    service = DataService(fake_redis, _StubSessionFactory())

    def fake_is_valid(t: str) -> bool:
        return t in {"AAPL", "MSFT"}

    with patch.object(ds_mod, "_yf_is_valid", side_effect=fake_is_valid):
        result = await service.validate_tickers(["AAPL", "ZZZZ", "MSFT", "QQQQQQ"])

    assert result["valid"] == ["AAPL", "MSFT"]
    assert result["invalid"] == ["ZZZZ", "QQQQQQ"]


@pytest.mark.asyncio
async def test_get_fundamentals_caches(fake_redis):
    service = DataService(fake_redis, _StubSessionFactory())
    fake_info = {
        "dividendYield": 0.024,
        "trailingPE": 18.4,
        "marketCap": 1_500_000_000,
        "beta": 1.05,
        "earningsDate": None,
        "sector": "Tech",
    }

    with patch.object(ds_mod, "_yf_info", return_value=fake_info) as mock_info:
        first = await service.get_fundamentals("AAPL")
        second = await service.get_fundamentals("AAPL")

    mock_info.assert_called_once()
    assert first["sector"] == "Tech"
    assert second["beta"] == 1.05


@pytest.mark.asyncio
async def test_market_prices_endpoint_uses_data_service(fake_redis):
    from app.main import app
    from app.services.data_service import DataService as RealDS

    df = _fake_price_df(6)
    service = RealDS(fake_redis, _StubSessionFactory())
    fake_redis.store[_prices_key("AAPL", 756)] = ds_mod._serialize_price_df(df)

    app.state.redis = fake_redis
    app.state.data_service = service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/market/prices", params={"tickers": "AAPL"})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["ticker"] == "AAPL"
    assert len(data[0]["prices"]) == 6
