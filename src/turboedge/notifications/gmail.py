"""SMTP-SSL-based Gmail notifier with dry-run support.

Credentials come exclusively from environment variables (GMAIL_USER,
GMAIL_APP_PASSWORD, TURBOEDGE_EMAIL_TO). When any are missing or the
config disables sending, notifications run in dry-run mode (no network I/O).
"""

from __future__ import annotations

import os
import smtplib
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import EmailMessage

import structlog

from turboedge.config import GmailConfig

logger = structlog.get_logger(__name__)


class NotificationError(Exception):
    """Raised when sending fails with network/SMTP error (no password in message)."""


@dataclass(frozen=True)
class GmailCredentials:
    """SMTP credentials loaded from environment variables."""

    user: str
    app_password: str = field(repr=False)
    recipients: list[str]

    @classmethod
    def from_env(cls) -> GmailCredentials | None:
        """Load credentials from GMAIL_USER, GMAIL_APP_PASSWORD, TURBOEDGE_EMAIL_TO.

        Returns None if any required var is missing or empty.
        """
        user = os.environ.get("GMAIL_USER", "").strip()
        password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
        to_raw = os.environ.get("TURBOEDGE_EMAIL_TO", "").strip()

        if not user or not password or not to_raw:
            return None

        recipients = [addr.strip() for addr in to_raw.split(",") if addr.strip()]
        if not recipients:
            return None

        return cls(user=user, app_password=password, recipients=recipients)


@dataclass(frozen=True)
class EmailMessageSpec:
    """Specification for an outgoing email."""

    subject: str
    body_text: str
    to: list[str]


@dataclass(frozen=True)
class SendResult:
    """Outcome of an attempted send."""

    sent: bool
    dry_run: bool
    recipients: list[str]
    message: str


class GmailNotifier:
    """Sends emails via SMTP SSL (smtp.gmail.com:465).

    If credentials are None or config.enabled is False, runs in dry-run mode
    (logs the subject, does not send).
    """

    def __init__(
        self,
        cfg: GmailConfig,
        credentials: GmailCredentials | None,
        smtp_factory: Callable[..., smtplib.SMTP_SSL] = smtplib.SMTP_SSL,
    ) -> None:
        """Initialize the notifier.

        Args:
            cfg: Configuration from gmail.yaml (includes subject_prefix).
            credentials: Credentials from env or None for dry-run.
            smtp_factory: Injectable SMTP connection factory (for testing).
        """
        self.cfg = cfg
        self.credentials = credentials
        self.smtp_factory = smtp_factory

    def send(self, spec: EmailMessageSpec) -> SendResult:
        """Send an email message, or simulate in dry-run mode.

        Prepends cfg.subject_prefix to subject if not already present.
        Uses email.message.EmailMessage with UTF-8 encoding.

        Args:
            spec: The email to send.

        Returns:
            SendResult indicating success/dry-run and outcome message.

        Raises:
            NotificationError: If SMTP/network fails (after retries).
                Password is never included in the error message.
        """
        # Prepare subject with prefix
        subject = spec.subject
        if not subject.startswith(self.cfg.subject_prefix):
            subject = f"{self.cfg.subject_prefix} {subject}"

        # Dry-run: no credentials or sending disabled
        if self.credentials is None or not self.cfg.enabled:
            dry_run_msg = f"dry-run: {subject}"
            logger.info("notification_dry_run", subject=subject, recipients=spec.to)
            return SendResult(
                sent=False,
                dry_run=True,
                recipients=spec.to,
                message=dry_run_msg,
            )

        # Attempt to send
        try:
            email_msg = EmailMessage()
            email_msg["Subject"] = subject
            email_msg["From"] = self.credentials.user
            email_msg["To"] = ", ".join(spec.to)
            email_msg.set_content(spec.body_text, charset="utf-8")

            context = ssl.create_default_context()
            with self.smtp_factory(
                self.cfg.smtp_host, self.cfg.smtp_port, timeout=30, context=context
            ) as smtp:
                smtp.login(self.credentials.user, self.credentials.app_password)
                smtp.send_message(email_msg)

            logger.info(
                "notification_sent",
                subject=subject,
                recipients=spec.to,
                user=self.credentials.user,
            )
            return SendResult(
                sent=True,
                dry_run=False,
                recipients=spec.to,
                message="sent",
            )

        except (smtplib.SMTPException, OSError) as exc:
            # Never include password in error message
            error_msg = f"SMTP/network error: {exc!s}"
            logger.error(
                "notification_failed",
                subject=subject,
                recipients=spec.to,
                error=error_msg,
            )
            raise NotificationError(error_msg) from exc


__all__ = [
    "EmailMessageSpec",
    "GmailCredentials",
    "GmailNotifier",
    "NotificationError",
    "SendResult",
]
