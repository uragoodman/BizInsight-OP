"""Authenticated encryption for short-lived preference Outbox payloads."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


PAYLOAD_KEY_VERSION = "v1"


class PreferencePayloadError(RuntimeError):
    """Payload key missing, malformed, or unable to authenticate ciphertext."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def payload_key_material(settings: object, *, override: str | bytes | None = None):
    """Return the configured retained keyring and active version.

    ``MEMORY_PREFERENCE_PAYLOAD_KEY`` remains the v1 compatibility fallback.
    During rotation, configure a JSON keyring containing both old and new keys,
    then switch the current version. Old versions stay available for pending
    Outbox rows until those rows are drained or expire.
    """
    if override is not None:
        return override, PAYLOAD_KEY_VERSION
    configured_value = getattr(settings, "memory_preference_payload_keyring", None) or ""
    if isinstance(configured_value, str):
        try:
            configured = json.loads(configured_value) if configured_value.strip() else {}
        except json.JSONDecodeError as exc:
            raise PreferencePayloadError("payload_keyring_invalid") from exc
    else:
        configured = configured_value
    if not isinstance(configured, dict) or any(
        not isinstance(version, str) or not isinstance(value, str)
        for version, value in configured.items()
    ):
        raise PreferencePayloadError("payload_keyring_invalid")
    if configured:
        version = str(getattr(settings, "memory_preference_payload_current_key_version", "v1"))
        if version not in configured:
            raise PreferencePayloadError("payload_current_key_version_missing")
        return configured, version
    fallback = str(getattr(settings, "memory_preference_payload_key", "") or "")
    if not fallback:
        raise PreferencePayloadError("payload_key_missing")
    return {PAYLOAD_KEY_VERSION: fallback}, PAYLOAD_KEY_VERSION


@dataclass(frozen=True)
class EncryptedPreferencePayload:
    nonce: bytes
    ciphertext: bytes
    key_version: str = PAYLOAD_KEY_VERSION


def _key_bytes(key: str | bytes) -> bytes:
    try:
        raw = bytes.fromhex(key) if isinstance(key, str) else bytes(key)
    except (ValueError, TypeError) as exc:
        raise PreferencePayloadError("payload_key_invalid") from exc
    if len(raw) != 32:
        raise PreferencePayloadError("payload_key_must_be_32_bytes_hex")
    return raw


def _aad(*, tenant_id: str, user_id: str, operation_id: str, key_version: str) -> bytes:
    return "|".join(("bizinsight-memory", key_version, tenant_id, user_id, operation_id)).encode("utf-8")


def encrypt_preference_payload(
    value: dict,
    *,
    key: str | bytes | Mapping[str, str | bytes],
    tenant_id: str,
    user_id: str,
    operation_id: str,
    key_version: str = PAYLOAD_KEY_VERSION,
) -> EncryptedPreferencePayload:
    if not isinstance(value, dict):
        raise PreferencePayloadError("typed_preference_value_required")
    if isinstance(key, Mapping):
        selected_key = key.get(key_version)
        if selected_key is None:
            raise PreferencePayloadError("payload_key_version_unsupported")
    else:
        if key_version != PAYLOAD_KEY_VERSION:
            raise PreferencePayloadError("payload_key_version_unsupported")
        selected_key = key
    nonce = os.urandom(12)
    aad = _aad(
        tenant_id=tenant_id, user_id=user_id, operation_id=operation_id,
        key_version=key_version,
    )
    plaintext = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ciphertext = AESGCM(_key_bytes(selected_key)).encrypt(nonce, plaintext, aad)
    return EncryptedPreferencePayload(nonce=nonce, ciphertext=ciphertext, key_version=key_version)


def decrypt_preference_payload(
    ciphertext: bytes,
    nonce: bytes,
    *,
    key: str | bytes | Mapping[str, str | bytes],
    key_version: str,
    tenant_id: str,
    user_id: str,
    operation_id: str,
) -> dict:
    if isinstance(key, Mapping):
        selected_key = key.get(key_version)
        if selected_key is None:
            raise PreferencePayloadError("payload_key_version_unsupported")
    else:
        if key_version != PAYLOAD_KEY_VERSION:
            raise PreferencePayloadError("payload_key_version_unsupported")
        selected_key = key
    aad = _aad(
        tenant_id=tenant_id, user_id=user_id, operation_id=operation_id,
        key_version=key_version,
    )
    try:
        plaintext = AESGCM(_key_bytes(selected_key)).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise PreferencePayloadError("payload_authentication_failed") from exc
    try:
        value = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreferencePayloadError("payload_json_invalid") from exc
    if not isinstance(value, dict):
        raise PreferencePayloadError("typed_preference_value_required")
    return value
