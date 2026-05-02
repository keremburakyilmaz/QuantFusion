# QuantFusion Backend

FastAPI backend for portfolio analytics: risk metrics, optimization, regime
detection, backtesting, and a public stateless analyzer. Plain Postgres 15 +
Redis. Two NVIDIA NIM integrations (Nemotron LLM agent + nemotron-ocr-v1).

## Quickstart (local)

```bash
cp .env.example .env
docker compose up -d
docker compose exec api alembic upgrade head
docker compose exec api python scripts/seed_demo.py
curl localhost:8000/api/health
```

## Stack

- FastAPI + Uvicorn
- SQLAlchemy + Alembic on Postgres 15
- Redis cache
- APScheduler for background jobs
- pytest + GitHub Actions CI (Postgres & Redis service containers)

Deployed on Render via `render.yaml`.
