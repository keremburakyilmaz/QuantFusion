from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.models import Holding, Portfolio


DEMO_NAME = "Demo Portfolio"
WEIGHTS = {
    "JEPI": 0.30,
    "JEPQ": 0.30,
    "VOO": 0.20,
    "QQQ": 0.20,
}


def main() -> None:
    db = SessionLocal()
    try:
        existing = db.query(Portfolio).filter_by(is_demo=True, name=DEMO_NAME).first()
        if existing is not None:
            print(f"Demo portfolio already exists: {existing.id}")
            return

        portfolio = Portfolio(name=DEMO_NAME, is_demo=True, currency="USD")
        db.add(portfolio)
        db.flush()

        for ticker, weight in WEIGHTS.items():
            db.add(
                Holding(
                    portfolio_id=portfolio.id,
                    ticker=ticker,
                    shares=0,
                    target_weight=weight,
                )
            )
        db.commit()
        print(f"Inserted demo portfolio: {portfolio.id}")
        print("Set DEMO_PORTFOLIO_ID env var to the UUID above.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
