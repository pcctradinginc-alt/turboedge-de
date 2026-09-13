from __future__ import annotations

import pytest

from turboedge.state.crypto import (
    MAGIC,
    MIN_PASSPHRASE_LEN,
    StateCryptoError,
    decrypt_bytes,
    encrypt_bytes,
)

_GOOD_PASSPHRASE = "a" * MIN_PASSPHRASE_LEN
_OTHER_PASSPHRASE = "b" * MIN_PASSPHRASE_LEN


def test_roundtrip() -> None:
    data = b"turboedge state archive payload \x00\x01\x02"
    blob = encrypt_bytes(data, _GOOD_PASSPHRASE)
    assert decrypt_bytes(blob, _GOOD_PASSPHRASE) == data


def test_roundtrip_empty_payload() -> None:
    blob = encrypt_bytes(b"", _GOOD_PASSPHRASE)
    assert decrypt_bytes(blob, _GOOD_PASSPHRASE) == b""


def test_blob_starts_with_magic_and_version() -> None:
    blob = encrypt_bytes(b"hello", _GOOD_PASSPHRASE)
    assert blob[: len(MAGIC)] == MAGIC
    assert blob[len(MAGIC)] == 1


def test_two_encryptions_of_same_data_differ() -> None:
    """Random salt+nonce per call -> ciphertext is never repeated."""
    data = b"same plaintext"
    blob1 = encrypt_bytes(data, _GOOD_PASSPHRASE)
    blob2 = encrypt_bytes(data, _GOOD_PASSPHRASE)
    assert blob1 != blob2
    assert decrypt_bytes(blob1, _GOOD_PASSPHRASE) == data
    assert decrypt_bytes(blob2, _GOOD_PASSPHRASE) == data


def test_wrong_key_raises_state_crypto_error() -> None:
    blob = encrypt_bytes(b"secret trade suggestion", _GOOD_PASSPHRASE)
    with pytest.raises(StateCryptoError):
        decrypt_bytes(blob, _OTHER_PASSPHRASE)


def test_tampered_ciphertext_raises() -> None:
    blob = bytearray(encrypt_bytes(b"secret trade suggestion", _GOOD_PASSPHRASE))
    blob[-1] ^= 0xFF  # flip a bit in the GCM tag/ciphertext tail
    with pytest.raises(StateCryptoError):
        decrypt_bytes(bytes(blob), _GOOD_PASSPHRASE)


def test_tampered_header_raises() -> None:
    """AAD covers the header -- flipping a header byte (here: inside the
    salt) must also fail authentication, not just ciphertext tampering."""
    blob = bytearray(encrypt_bytes(b"secret trade suggestion", _GOOD_PASSPHRASE))
    blob[len(MAGIC) + 1] ^= 0xFF  # first salt byte
    with pytest.raises(StateCryptoError):
        decrypt_bytes(bytes(blob), _GOOD_PASSPHRASE)


def test_bad_magic_raises() -> None:
    blob = bytearray(encrypt_bytes(b"data", _GOOD_PASSPHRASE))
    blob[0:1] = b"X"
    with pytest.raises(StateCryptoError, match="magic"):
        decrypt_bytes(bytes(blob), _GOOD_PASSPHRASE)


def test_truncated_blob_raises() -> None:
    blob = encrypt_bytes(b"data", _GOOD_PASSPHRASE)[:10]
    with pytest.raises(StateCryptoError, match="too short"):
        decrypt_bytes(blob, _GOOD_PASSPHRASE)


def test_short_passphrase_rejected_on_encrypt() -> None:
    with pytest.raises(StateCryptoError, match="24"):
        encrypt_bytes(b"data", "too-short")


def test_short_passphrase_rejected_on_decrypt() -> None:
    blob = encrypt_bytes(b"data", _GOOD_PASSPHRASE)
    with pytest.raises(StateCryptoError, match="24"):
        decrypt_bytes(blob, "too-short")


def test_short_passphrase_error_mentions_keygen_hint() -> None:
    with pytest.raises(StateCryptoError, match=r"secrets\.token_urlsafe"):
        encrypt_bytes(b"data", "short")
