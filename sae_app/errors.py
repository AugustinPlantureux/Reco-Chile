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


class CandidateEvaluationError(ValueError):
    """Raised when one recommendation candidate has malformed row data.

    Unexpected programming errors must not be wrapped in this exception.
    """


class DataSchemaError(ValueError):
    """Raised when an input CSV is readable but structurally unsafe to use.

    Shared by both sae_app.data_loading_pandas and sae_app.data_loading_duckdb
    so callers can catch one exception type regardless of which loader
    implementation is active.
    """
