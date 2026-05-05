
from typing import Sequence, Union

from alembic import op


revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Dedupe existing rows: keep the most recently uploaded per (ticker, filing_date)
    op.execute(
        """
        DELETE FROM earnings_documents
        WHERE id NOT IN (
            SELECT DISTINCT ON (ticker, filing_date) id
            FROM earnings_documents
            WHERE filing_date IS NOT NULL
            ORDER BY ticker, filing_date, uploaded_at DESC
        )
        AND filing_date IS NOT NULL
        """
    )
    op.create_unique_constraint(
        "uq_earnings_ticker_filing_date",
        "earnings_documents",
        ["ticker", "filing_date"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_earnings_ticker_filing_date", "earnings_documents")
