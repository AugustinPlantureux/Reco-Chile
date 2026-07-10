"""Typed exceptions raised by the calculation engine.

The engine exposes translation keys rather than importing the Streamlit/i18n
layer. User interfaces localize these errors at the presentation boundary.
"""

from __future__ import annotations


class MtbEngineError(ValueError):
    """Base class for expected, user-facing MTB calculation errors."""

    def __init__(self, message_key: str, **message_kwargs) -> None:
        self.message_key = message_key
        self.message_kwargs = message_kwargs
        super().__init__(message_key)


class InvalidStudentIdentifier(MtbEngineError):
    """Raised when the supplied RUN/IPE cannot be used for hashing."""


class UnknownProgram(MtbEngineError):
    """Raised when no program data matches a wish or precomputed value."""


class EmptyWishList(MtbEngineError):
    """Raised when a calculation receives no valid wishes."""
