"""Tests for notifications/gmail.py."""

from __future__ import annotations

import smtplib
from unittest.mock import Mock

import pytest

from turboedge.config import GmailConfig
from turboedge.notifications.gmail import (
    EmailMessageSpec,
    GmailCredentials,
    GmailNotifier,
    NotificationError,
    SendResult,
)


class TestGmailCredentials:
    """Test GmailCredentials.from_env()."""

    def test_from_env_all_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Load credentials when all env vars are present."""
        monkeypatch.setenv("GMAIL_USER", "user@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "app_pass_123")
        monkeypatch.setenv("TURBOEDGE_EMAIL_TO", "recipient1@example.com, recipient2@example.com")

        creds = GmailCredentials.from_env()

        assert creds is not None
        assert creds.user == "user@gmail.com"
        assert creds.app_password == "app_pass_123"
        assert creds.recipients == ["recipient1@example.com", "recipient2@example.com"]

    def test_from_env_missing_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return None if GMAIL_USER is missing."""
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "pass")
        monkeypatch.setenv("TURBOEDGE_EMAIL_TO", "recipient@example.com")

        assert GmailCredentials.from_env() is None

    def test_from_env_missing_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return None if GMAIL_APP_PASSWORD is missing."""
        monkeypatch.setenv("GMAIL_USER", "user@gmail.com")
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        monkeypatch.setenv("TURBOEDGE_EMAIL_TO", "recipient@example.com")

        assert GmailCredentials.from_env() is None

    def test_from_env_missing_recipients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return None if TURBOEDGE_EMAIL_TO is missing."""
        monkeypatch.setenv("GMAIL_USER", "user@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "pass")
        monkeypatch.delenv("TURBOEDGE_EMAIL_TO", raising=False)

        assert GmailCredentials.from_env() is None

    def test_from_env_empty_strings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return None if env vars are empty strings."""
        monkeypatch.setenv("GMAIL_USER", "")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "pass")
        monkeypatch.setenv("TURBOEDGE_EMAIL_TO", "recipient@example.com")

        assert GmailCredentials.from_env() is None

    def test_from_env_whitespace_trimming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Trim whitespace from recipients."""
        monkeypatch.setenv("GMAIL_USER", "user@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "pass")
        monkeypatch.setenv("TURBOEDGE_EMAIL_TO", "  r1@example.com  ,  r2@example.com  ")

        creds = GmailCredentials.from_env()

        assert creds is not None
        assert creds.recipients == ["r1@example.com", "r2@example.com"]

    def test_credentials_repr_hides_password(self) -> None:
        """Verify repr= False on app_password hides the value."""
        creds = GmailCredentials(
            user="user@gmail.com", app_password="secret", recipients=["r@e.com"]
        )

        repr_str = repr(creds)

        # repr should not contain the password value
        assert "secret" not in repr_str
        # The field itself should not appear in repr because repr=False
        assert "app_password" not in repr_str


class TestGmailNotifier:
    """Test GmailNotifier.send()."""

    def test_send_dry_run_no_credentials(self) -> None:
        """Send returns dry-run when credentials are None."""
        cfg = GmailConfig(
            enabled=True,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            send_on=[],
            subject_prefix="[TurboEdge]",
            daily_digest=False,
        )
        notifier = GmailNotifier(cfg, credentials=None)

        spec = EmailMessageSpec(subject="Test", body_text="Body", to=["r@e.com"])
        result = notifier.send(spec)

        assert not result.sent
        assert result.dry_run
        assert result.recipients == ["r@e.com"]
        assert "Test" in result.message

    def test_send_dry_run_disabled(self) -> None:
        """Send returns dry-run when cfg.enabled is False."""
        cfg = GmailConfig(
            enabled=False,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            send_on=[],
            subject_prefix="[TurboEdge]",
            daily_digest=False,
        )
        creds = GmailCredentials(user="u@g.com", app_password="pass", recipients=["r@e.com"])
        notifier = GmailNotifier(cfg, credentials=creds)

        spec = EmailMessageSpec(subject="Test", body_text="Body", to=["r@e.com"])
        result = notifier.send(spec)

        assert not result.sent
        assert result.dry_run

    def test_send_prepends_subject_prefix(self) -> None:
        """Subject is prefixed with cfg.subject_prefix if not already there."""
        cfg = GmailConfig(
            enabled=False,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            send_on=[],
            subject_prefix="[TurboEdge]",
            daily_digest=False,
        )
        creds = GmailCredentials(user="u@g.com", app_password="pass", recipients=["r@e.com"])
        notifier = GmailNotifier(cfg, credentials=creds)

        spec = EmailMessageSpec(subject="Test Report", body_text="Body", to=["r@e.com"])
        result = notifier.send(spec)

        # In dry-run, the message includes the prefixed subject
        assert "[TurboEdge] Test Report" in result.message

    def test_send_does_not_double_prefix(self) -> None:
        """Subject is not prefixed twice if it already starts with the prefix."""
        cfg = GmailConfig(
            enabled=False,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            send_on=[],
            subject_prefix="[TurboEdge]",
            daily_digest=False,
        )
        creds = GmailCredentials(user="u@g.com", app_password="pass", recipients=["r@e.com"])
        notifier = GmailNotifier(cfg, credentials=creds)

        spec = EmailMessageSpec(subject="[TurboEdge] Test", body_text="Body", to=["r@e.com"])
        result = notifier.send(spec)

        # Should not double-prefix
        assert result.message.count("[TurboEdge]") == 1

    def test_send_success(self) -> None:
        """Successful send returns sent=True."""
        cfg = GmailConfig(
            enabled=True,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            send_on=[],
            subject_prefix="[TurboEdge]",
            daily_digest=False,
        )
        creds = GmailCredentials(user="u@gmail.com", app_password="pass", recipients=["r@e.com"])

        # Mock SMTP
        mock_smtp = Mock()
        factory = Mock(return_value=mock_smtp)
        mock_smtp.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp.__exit__ = Mock(return_value=None)

        notifier = GmailNotifier(cfg, credentials=creds, smtp_factory=factory)

        spec = EmailMessageSpec(subject="Test", body_text="Body text", to=["r@e.com"])
        result = notifier.send(spec)

        assert result.sent
        assert not result.dry_run
        assert result.message == "sent"
        assert result.recipients == ["r@e.com"]

        # Verify SMTP calls
        factory.assert_called_once()
        mock_smtp.login.assert_called_once_with("u@gmail.com", "pass")
        mock_smtp.send_message.assert_called_once()

    def test_send_smtp_failure(self) -> None:
        """SMTP error raises NotificationError without password."""
        cfg = GmailConfig(
            enabled=True,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            send_on=[],
            subject_prefix="[TurboEdge]",
            daily_digest=False,
        )
        creds = GmailCredentials(
            user="u@gmail.com", app_password="super_secret", recipients=["r@e.com"]
        )

        # Mock SMTP to raise error
        mock_smtp = Mock()
        mock_smtp.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp.__exit__ = Mock(return_value=None)
        mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, "Bad credentials")

        factory = Mock(return_value=mock_smtp)
        notifier = GmailNotifier(cfg, credentials=creds, smtp_factory=factory)

        spec = EmailMessageSpec(subject="Test", body_text="Body", to=["r@e.com"])

        with pytest.raises(NotificationError) as exc_info:
            notifier.send(spec)

        # Ensure password is not in error message
        assert "super_secret" not in str(exc_info.value)
        assert "SMTP/network error" in str(exc_info.value)


class TestEmailMessageSpec:
    """Test EmailMessageSpec dataclass."""

    def test_create_and_access(self) -> None:
        """Create and access fields."""
        spec = EmailMessageSpec(
            subject="Test Subject",
            body_text="Body text here",
            to=["a@example.com", "b@example.com"],
        )

        assert spec.subject == "Test Subject"
        assert spec.body_text == "Body text here"
        assert spec.to == ["a@example.com", "b@example.com"]


class TestSendResult:
    """Test SendResult dataclass."""

    def test_create_success(self) -> None:
        """Create a successful send result."""
        result = SendResult(
            sent=True,
            dry_run=False,
            recipients=["r@e.com"],
            message="sent",
        )

        assert result.sent
        assert not result.dry_run
        assert result.message == "sent"

    def test_create_dry_run(self) -> None:
        """Create a dry-run result."""
        result = SendResult(
            sent=False,
            dry_run=True,
            recipients=["r@e.com"],
            message="dry-run: test subject",
        )

        assert not result.sent
        assert result.dry_run
