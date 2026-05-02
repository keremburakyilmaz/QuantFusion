from __future__ import annotations

from redis.asyncio import Redis
from fastapi import Request

from app.config import settings


def create_redis() -> Redis:
    return Redis.from_url(settings.REDIS_URL, decode_responses=True)


async def get_redis(request: Request) -> Redis:
    return request.app.state.redis
