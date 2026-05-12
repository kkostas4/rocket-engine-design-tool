"""
exceptions.py
=============

Project-wide typed exceptions.

All sizing failures inherit from SizingInfeasibleError so callers can
catch by type without importing subsystem-specific exception classes.
"""

from __future__ import annotations


class SizingInfeasibleError(RuntimeError):
    """
    Raised when a sizing step cannot find a geometry that satisfies all
    engineering constraints.

    Inherits from RuntimeError for backward compatibility with code that
    catches RuntimeError.  Callers that want to handle sizing failures
    specifically should catch SizingInfeasibleError.
    """
