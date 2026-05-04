import asyncio
import logging
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.models import PortfolioSnapshot
from app.schemas.analyzer import AnalysisReport
from app.schemas.common import HoldingInput
from app.schemas.snapshot import SaveSnapshotResponse, SnapshotResponse


logger = logging.getLogger(__name__)

TOKEN_BYTES = 6  # → 8 base64-url chars
MAX_RETRIES = 3


class SnapshotService:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    async def save(
        self,
        holdings: list[HoldingInput],
        report: AnalysisReport,
        expires_in_days: int = 30,
    ) -> SaveSnapshotResponse:
        return await asyncio.to_thread(
            self._save_sync, holdings, report, expires_in_days
        )

    async def fetch(self, token: str) -> SnapshotResponse | None:
        return await asyncio.to_thread(self._fetch_sync, token)

    async def cleanup_expired(self) -> int:
        return await asyncio.to_thread(self._cleanup_sync)

    # ---------- sync core ----------

    def _save_sync(
        self,
        holdings: list[HoldingInput],
        report: AnalysisReport,
        expires_in_days: int,
    ) -> SaveSnapshotResponse:
        expires_at: datetime | None = None
        if expires_in_days > 0:
            expires_at = datetime.now(tz=timezone.utc) + timedelta(days=expires_in_days)

        holdings_json = [h.model_dump() for h in holdings]
        report_json = report.model_dump(mode="json")

        for attempt in range(MAX_RETRIES):
            token = secrets.token_urlsafe(TOKEN_BYTES)
            try:
                with self.session_factory() as db:
                    db.add(
                        PortfolioSnapshot(
                            token=token,
                            holdings=holdings_json,
                            report=report_json,
                            expires_at=expires_at,
                        )
                    )
                    db.commit()
                return SaveSnapshotResponse(
                    token=token,
                    expires_at=expires_at,
                    share_url=f"/api/analyzer/snapshot/{token}",
                )
            except IntegrityError:
                logger.warning("snapshot token collision on %s, retrying", token)
                continue

        raise RuntimeError("Failed to allocate a unique snapshot token")

    def _fetch_sync(self, token: str) -> SnapshotResponse | None:
        with self.session_factory() as db:
            row = db.execute(
                select(PortfolioSnapshot).where(PortfolioSnapshot.token == token)
            ).scalar_one_or_none()

        if row is None:
            return None
        if row.expires_at is not None and row.expires_at < datetime.now(tz=timezone.utc):
            return None

        holdings = [HoldingInput.model_validate(h) for h in row.holdings]
        report = AnalysisReport.model_validate(row.report)
        return SnapshotResponse(
            token=row.token,
            holdings=holdings,
            report=report,
            created_at=row.created_at,
            expires_at=row.expires_at,
        )

    def _cleanup_sync(self) -> int:
        with self.session_factory() as db:
            result = db.execute(
                delete(PortfolioSnapshot).where(
                    PortfolioSnapshot.expires_at.isnot(None),
                    PortfolioSnapshot.expires_at < datetime.now(tz=timezone.utc),
                )
            )
            db.commit()
            return int(result.rowcount or 0)
