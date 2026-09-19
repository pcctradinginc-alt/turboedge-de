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
    DailyResearchProtocolEntry,
    NoActionableDigestContext,
    ScanReportContext,
    ScanReportRow,
    render_no_actionable_digest,
    render_scan_report,
    render_test_email,
)

__all__ = [
    "DailyResearchProtocolEntry",
    "EmailMessageSpec",
    "GmailConfig",
    "GmailCredentials",
    "GmailNotifier",
    "NoActionableDigestContext",
    "NotificationDeduplicator",
    "NotificationError",
    "ScanReportContext",
    "ScanReportRow",
    "SendResult",
    "notification_hash",
    "render_no_actionable_digest",
    "render_scan_report",
    "render_test_email",
]
