from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))
    currency: Mapped[str] = mapped_column(Text, server_default=text("'USD'"))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("NOW()")
    )


class Holding(Base):
    __tablename__ = "holdings"
    __table_args__ = (UniqueConstraint("portfolio_id", "ticker"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("portfolios.id", ondelete="CASCADE")
    )
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    shares: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    avg_cost: Mapped[float | None] = mapped_column(Numeric(18, 4))
    target_weight: Mapped[float | None] = mapped_column(Numeric(5, 4))


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("portfolios.id")
    )
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    txn_type: Mapped[str] = mapped_column(Text, nullable=False)
    shares: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    executed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)


class PriceHistory(Base):
    __tablename__ = "price_history"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    close: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    adj_close: Mapped[float | None] = mapped_column(Numeric(18, 4))
    volume: Mapped[int | None] = mapped_column(BigInteger)

    __table_args__ = (
        Index("idx_price_history_lookup", "ticker", text("ts DESC")),
    )


class RegimeSnapshot(Base):
    __tablename__ = "regime_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    ts: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    regime: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    features: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (Index("idx_regime_ts", text("ts DESC")),)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    holdings: Mapped[dict] = mapped_column(JSONB, nullable=False)
    report: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("NOW()")
    )
    expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    __table_args__ = (Index("idx_snapshots_expires", "expires_at"),)
