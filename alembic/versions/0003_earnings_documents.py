
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "earnings_documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column(
            "uploaded_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("filing_date", sa.Text),
        sa.Column("form_type", sa.Text, server_default=sa.text("'8-K'")),
        sa.Column("pages", sa.Integer),
        sa.Column("ocr_text", sa.Text),
        sa.Column("signals", postgresql.JSONB),
    )
    op.execute(
        "CREATE INDEX idx_earnings_docs_ticker "
        "ON earnings_documents (ticker, uploaded_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_earnings_docs_ticker")
    op.drop_table("earnings_documents")
