"""
cache.py
========

Per-Engine-instance memoization of design() intermediate results.

Key design principle — key propagation
---------------------------------------
Intermediate objects (CombustionGas, RaoNozzle, ...) are not frozen
dataclasses and therefore not directly hashable.  Instead of hashing
the output objects, each step's cache key incorporates the *key* of
every upstream step it depends on:

    gas_key  = ("gas",    propellant, MR, Pc)
    nozzle_key = ("nozzle", gas_key,  thrust, eps)
    ...

If thermochem inputs change, gas_key changes, and therefore every
downstream key that contains gas_key also changes — correctness is
enforced by key composition, not by side-channel invalidation.

This requires all key components to be hashable.  EngineRequirements
and PropellantPair are frozen dataclasses (hashable by value).  Scalar
floats and strings are always hashable.  No numpy arrays, callables, or
dicts appear in keys.

Out of scope
------------
- On-disk persistence (joblib.Memory).  Pure in-memory; not persisted.
- Cross-Engine instance sharing.
- Cache size limits / LRU eviction.
"""

from __future__ import annotations

from typing import Any, Callable


class StepCache:
    """
    Per-Engine-instance memoization of design() intermediates.

    Keyed by hash of input tuples.  Pure in-memory; not persisted across runs.

    Parameters
    ----------
    None.

    Examples
    --------
    >>> cache = StepCache()
    >>> key = ("gas", propellant, 3.6, 6.895e6)
    >>> gas = cache.get_or_compute(key, lambda: _thermochem(...))
    >>> # Second call with same key returns cached value, fn is not called.
    >>> gas2 = cache.get_or_compute(key, lambda: _thermochem(...))
    >>> gas is gas2
    True
    """

    def __init__(self) -> None:
        self._store: dict[tuple, Any] = {}

    # ------------------------------------------------------------------

    def get_or_compute(self, key: tuple, fn: Callable[[], Any]) -> Any:
        """
        Return the cached result for *key*, computing it via *fn* if absent.

        Parameters
        ----------
        key : tuple
            Hashable tuple that uniquely identifies the step's output.
            All elements must be hashable.
        fn : Callable[[], Any]
            Zero-argument factory that computes the result when the cache
            misses.  Only called on a miss; the result is stored and returned.

        Returns
        -------
        Any
            Cached or freshly computed result.

        Raises
        ------
        TypeError
            If *key* contains non-hashable elements (programmer error).
        """
        if key not in self._store:
            self._store[key] = fn()
        return self._store[key]

    # ------------------------------------------------------------------

    def invalidate(self, prefix: tuple | None = None) -> None:
        """
        Remove cached entries.

        Parameters
        ----------
        prefix : tuple or None
            If None, clear the entire cache.
            If a tuple, remove all entries whose key starts with *prefix*
            (i.e., key[:len(prefix)] == prefix).
        """
        if prefix is None:
            self._store.clear()
        else:
            to_remove = [
                k for k in self._store
                if k[: len(prefix)] == prefix
            ]
            for k in to_remove:
                del self._store[k]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: tuple) -> bool:
        return key in self._store
