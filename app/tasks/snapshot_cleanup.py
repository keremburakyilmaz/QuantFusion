import logging

from app.services.snapshot_service import SnapshotService


logger = logging.getLogger(__name__)


async def cleanup_snapshots(snapshot_service: SnapshotService) -> None:
    try:
        deleted = await snapshot_service.cleanup_expired()
        logger.info("snapshot_cleanup deleted %d expired rows", deleted)
    except Exception:
        logger.exception("snapshot_cleanup failed")
