

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "portfolios",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("is_demo", sa.Boolean, server_default=sa.text("FALSE")),
        sa.Column("currency", sa.Text, server_default=sa.text("'USD'")),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )

    op.create_table(
        "holdings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "portfolio_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id", ondelete="CASCADE"),
        ),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("shares", sa.Numeric(18, 6), nullable=False),
        sa.Column("avg_cost", sa.Numeric(18, 4)),
        sa.Column("target_weight", sa.Numeric(5, 4)),
        sa.UniqueConstraint("portfolio_id", "ticker", name="uq_holdings_portfolio_ticker"),
    )

    op.create_table(
        "transactions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "portfolio_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id"),
        ),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("txn_type", sa.Text, nullable=False),
        sa.Column("shares", sa.Numeric(18, 6), nullable=False),
        sa.Column("price", sa.Numeric(18, 4), nullable=False),
        sa.Column("executed_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
    )

    op.create_table(
        "price_history",
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("ts", sa.Date, nullable=False),
        sa.Column("close", sa.Numeric(18, 4), nullable=False),
        sa.Column("adj_close", sa.Numeric(18, 4)),
        sa.Column("volume", sa.BigInteger),
        sa.PrimaryKeyConstraint("ticker", "ts"),
    )
    op.execute(
        "CREATE INDEX idx_price_history_lookup ON price_history (ticker, ts DESC)"
    )

    op.create_table(
        "regime_snapshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ts", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("regime", sa.Text, nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4)),
        sa.Column("features", postgresql.JSONB),
    )
    op.execute("CREATE INDEX idx_regime_ts ON regime_snapshots (ts DESC)")

    op.create_table(
        "portfolio_snapshots",
        sa.Column("token", sa.String, primary_key=True),
        sa.Column("holdings", postgresql.JSONB, nullable=False),
        sa.Column("report", postgresql.JSONB, nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.Column("expires_at", postgresql.TIMESTAMP(timezone=True)),
    )
    op.create_index(
        "idx_snapshots_expires", "portfolio_snapshots", ["expires_at"]
    )


def downgrade() -> None:
    op.drop_index("idx_snapshots_expires", table_name="portfolio_snapshots")
    op.drop_table("portfolio_snapshots")
    op.execute("DROP INDEX IF EXISTS idx_regime_ts")
    op.drop_table("regime_snapshots")
    op.execute("DROP INDEX IF EXISTS idx_price_history_lookup")
    op.drop_table("price_history")
    op.drop_table("transactions")
    op.drop_table("holdings")
    op.drop_table("portfolios")
