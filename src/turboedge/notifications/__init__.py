"""Email notifications for TurboEdge-DE scan reports and health alerts."""

from turboedge.notifications.dedup import NotificationDeduplicator, notification_hash
from turboedge.notifications.gmail import (
    EmailMessageSpec,
    GmailConfig,
    GmailCredentials,
    GmailNotifier,
    NotificationError,
    SendResult,
)
from turboedge.notifications.templates import (
    ScanReportContext,
    ScanReportRow,
    render_scan_report,
    render_test_email,
)

__all__ = [
    "EmailMessageSpec",
    "GmailConfig",
    "GmailCredentials",
    "GmailNotifier",
    "NotificationDeduplicator",
    "NotificationError",
    "ScanReportContext",
    "ScanReportRow",
    "SendResult",
    "notification_hash",
    "render_scan_report",
    "render_test_email",
]
