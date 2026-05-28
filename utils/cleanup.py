"""
Automatic file cleanup for Nexus CV.
Deletes old reports and uploads to prevent disk bloat.
Can be run standalone (cron) or via APScheduler background job.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# Retention periods in seconds
REPORT_MAX_AGE = 7 * 24 * 3600    # 7 days
UPLOAD_MAX_AGE = 24 * 3600        # 24 hours


def _delete_old_files(directory, max_age_seconds):
    """Delete files older than max_age_seconds from directory."""
    if not os.path.isdir(directory):
        return 0

    now = time.time()
    deleted = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            file_age = now - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                os.remove(filepath)
                deleted += 1
                logger.info("Deleted old file: %s (age: %.1f hours)",
                            filename, file_age / 3600)
        except OSError as e:
            logger.warning("Failed to delete %s: %s", filename, e)

    return deleted


def cleanup_old_files():
    """Run full cleanup cycle for reports and uploads."""
    logger.info("Starting file cleanup cycle...")

    reports_deleted = _delete_old_files(REPORTS_DIR, REPORT_MAX_AGE)
    uploads_deleted = _delete_old_files(UPLOADS_DIR, UPLOAD_MAX_AGE)

    logger.info("Cleanup complete: %d reports, %d uploads removed",
                reports_deleted, uploads_deleted)
    return reports_deleted + uploads_deleted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cleanup_old_files()
