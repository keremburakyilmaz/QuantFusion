from __future__ import annotations

import pytest


@pytest.fixture
def fake_redis():
    """Minimal in-memory async Redis stub: get/set/delete/keys with TTL ignored."""

    class FakeRedis:
        def __init__(self):
            self.store: dict[str, str] = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, ex=None):
            self.store[key] = value

        async def delete(self, *keys):
            for k in keys:
                self.store.pop(k, None)

        async def keys(self, pattern="*"):
            return list(self.store.keys())

        async def aclose(self):
            self.store.clear()

    return FakeRedis()
