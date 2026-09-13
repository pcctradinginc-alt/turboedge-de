"""Authenticated envelope encryption for TurboEdge-DE state archives.

Scheme: a passphrase (``TURBOEDGE_STATE_KEY``) is stretched into a 256-bit
key via scrypt (``n=2**15, r=8, p=1``, 16-byte random salt), which then keys
AES-256-GCM (12-byte random nonce) over the plaintext. The archive header
(magic + version + salt + nonce) is passed as GCM associated data (AAD), so
tampering with any header byte -- not just the ciphertext -- also fails
authentication.

Wire format (all fields fixed-width, no length prefixes needed)::

    MAGIC (6 bytes, b"TEDGE1") | VERSION (1 byte) | salt (16 bytes)
    | nonce (12 bytes) | ciphertext‖tag (AES-GCM output, tag is the trailing
    16 bytes)

A wrong passphrase or any bit of corruption/tampering anywhere in the blob
(header or ciphertext) raises :class:`StateCryptoError` -- there is no
partial/best-effort decryption.
"""

from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC: bytes = b"TEDGE1"
VERSION: int = 1

SALT_LEN = 16
NONCE_LEN = 12
KEY_LEN = 32

SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1

MIN_PASSPHRASE_LEN = 24
_KEYGEN_HINT = 'python -c "import secrets;print(secrets.token_urlsafe(32))"'

_HEADER_LEN = len(MAGIC) + 1 + SALT_LEN + NONCE_LEN


class StateCryptoError(Exception):
    """Encryption/decryption failure: too-short passphrase, wrong key, a
    malformed header, or corrupted/tampered ciphertext. Deliberately does not
    distinguish "wrong key" from "corrupted data" in the exception type (GCM
    authentication cannot tell them apart) -- only in the message."""


def _check_passphrase(passphrase: str) -> None:
    if len(passphrase) < MIN_PASSPHRASE_LEN:
        raise StateCryptoError(
            f"TURBOEDGE_STATE_KEY must be at least {MIN_PASSPHRASE_LEN} characters "
            f"(got {len(passphrase)}). Generate a strong one with:\n  {_KEYGEN_HINT}"
        )


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_bytes(data: bytes, passphrase: str) -> bytes:
    """Encrypt ``data`` with ``passphrase``. See module docstring for format.

    Raises:
        StateCryptoError: if ``passphrase`` is shorter than
            :data:`MIN_PASSPHRASE_LEN`.
    """
    _check_passphrase(passphrase)
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = _derive_key(passphrase, salt)
    header = MAGIC + bytes([VERSION]) + salt + nonce
    ciphertext = AESGCM(key).encrypt(nonce, data, header)
    return header + ciphertext


def decrypt_bytes(blob: bytes, passphrase: str) -> bytes:
    """Decrypt a blob produced by :func:`encrypt_bytes`.

    Raises:
        StateCryptoError: if ``passphrase`` is too short, ``blob`` is too
            short/malformed to contain a valid header, the magic/version
            does not match, or GCM authentication fails (wrong passphrase or
            the blob was corrupted/tampered with).
    """
    _check_passphrase(passphrase)
    if len(blob) < _HEADER_LEN:
        raise StateCryptoError(
            f"ciphertext too short ({len(blob)} bytes, need at least {_HEADER_LEN}); "
            "not a valid TurboEdge-DE state archive"
        )

    magic = blob[: len(MAGIC)]
    if magic != MAGIC:
        raise StateCryptoError(
            f"bad magic header {magic!r} (expected {MAGIC!r}); not a TurboEdge-DE state archive"
        )

    offset = len(MAGIC)
    version = blob[offset]
    offset += 1
    if version != VERSION:
        raise StateCryptoError(f"unsupported state archive version {version} (expected {VERSION})")

    salt = blob[offset : offset + SALT_LEN]
    offset += SALT_LEN
    nonce = blob[offset : offset + NONCE_LEN]
    offset += NONCE_LEN

    header = blob[:offset]
    ciphertext = blob[offset:]

    key = _derive_key(passphrase, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, header)
    except InvalidTag as exc:
        raise StateCryptoError(
            "decryption failed: wrong TURBOEDGE_STATE_KEY or the archive is corrupted/tampered with"
        ) from exc


__all__ = [
    "MAGIC",
    "MIN_PASSPHRASE_LEN",
    "VERSION",
    "StateCryptoError",
    "decrypt_bytes",
    "encrypt_bytes",
]
