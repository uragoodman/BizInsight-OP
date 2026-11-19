"""Authorization foundation: authoritative policy facts and runtime snapshots."""

from app.auth.context import (
    AuthorizationContext,
    AuthorizationDenied,
    build_authorization_context,
)

__all__ = [
    "AuthorizationContext",
    "AuthorizationDenied",
    "build_authorization_context",
]
