"""Random-number helpers for compatibility with BIMETS R."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.special import ndtri

_UINT32_MASK = 0xFFFFFFFF
_MT_SIZE = 624
_MT_OFFSET = 397
_MT_MATRIX_A = 0x9908B0DF
_MT_UPPER_MASK = 0x80000000
_MT_LOWER_MASK = 0x7FFFFFFF
_UINT32_SCALE = 1.0 / 2**32


class RMersenneTwister:
    """Generate values with R's default ``set.seed`` and normal-kind semantics."""

    __slots__ = ("_index", "_state")

    def __init__(self, seed: int) -> None:
        """Initialize the state using R's 69069 linear-congruential seeding."""
        value = seed & _UINT32_MASK
        for _ in range(50):
            value = (69069 * value + 1) & _UINT32_MASK
        seeded: list[int] = []
        for _ in range(_MT_SIZE + 1):
            value = (69069 * value + 1) & _UINT32_MASK
            seeded.append(value)
        # R reserves the first generated word for the state position.
        self._state = seeded[1:]
        self._index = _MT_SIZE

    def uniform(
        self,
        lower: float,
        upper: float,
        shape: int | Sequence[int],
    ) -> np.ndarray:
        """Return R-compatible ``runif`` values in an R-shaped array."""
        dimensions = (shape,) if isinstance(shape, int) else tuple(shape)
        count = int(np.prod(dimensions, dtype=np.intp))
        values = np.fromiter(
            (self._next_uniform() for _ in range(count)),
            dtype=float,
            count=count,
        )
        values = lower + (upper - lower) * values
        # R matrices consume vectors column by column.
        return values.reshape(dimensions, order="F")

    def normal(
        self,
        mean: float,
        standard_deviation: float,
        shape: int | Sequence[int],
    ) -> np.ndarray:
        """Return R-compatible default ``rnorm`` values in an R-shaped array.

        R's default ``normal.kind="Inversion"`` constructs each probability
        from two successive MT19937 uniforms, retaining 27 high-order bits
        from the first and completing the fraction with the second. Applying
        the inverse standard-normal CDF then reproduces ``stats::rnorm`` and
        its column-major matrix filling order.
        """
        dimensions = (shape,) if isinstance(shape, int) else tuple(shape)
        count = int(np.prod(dimensions, dtype=np.intp))
        probabilities = np.fromiter(
            (
                (
                    math.floor(134_217_728.0 * self._next_uniform())
                    + self._next_uniform()
                )
                / 134_217_728.0
                for _ in range(count)
            ),
            dtype=float,
            count=count,
        )
        values = np.asarray(
            mean + standard_deviation * ndtri(probabilities), dtype=float
        )
        return values.reshape(dimensions, order="F")

    def sample_with_replacement(
        self,
        population_size: int,
        shape: int | Sequence[int],
    ) -> np.ndarray:
        """Return zero-based R-compatible replacement-sampling indexes.

        R versions from 3.6 onward use rejection sampling for ``sample()``.
        Each candidate is assembled from 16-bit chunks of the same uniform
        stream and rejected outside the population, avoiding modulo bias.
        Results retain R's column-major matrix filling order.
        """
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        dimensions = (shape,) if isinstance(shape, int) else tuple(shape)
        count = int(np.prod(dimensions, dtype=np.intp))
        bits = math.ceil(math.log2(population_size))
        indexes = np.fromiter(
            (self._sample_index(population_size, bits) for _ in range(count)),
            dtype=np.intp,
            count=count,
        )
        return indexes.reshape(dimensions, order="F")

    def _sample_index(self, population_size: int, bits: int) -> int:
        """Draw one unbiased zero-based index using R's rejection method."""
        mask = (1 << bits) - 1
        while True:
            value = 0
            for _ in range(0, bits + 1, 16):
                chunk = math.floor(self._next_uniform() * 65_536.0)
                value = 65_536 * value + chunk
            value &= mask
            if value < population_size:
                return value

    def _next_uniform(self) -> float:
        """Return one open-interval uniform value from the MT state."""
        value = self._next_uint32() * _UINT32_SCALE
        if value <= 0.0:
            return 0.5 * _UINT32_SCALE
        if value >= 1.0:
            return 1.0 - 0.5 * _UINT32_SCALE
        return value

    def _next_uint32(self) -> int:
        """Advance and temper the MT19937 state used by R."""
        if self._index >= _MT_SIZE:
            self._twist()
        value = self._state[self._index]
        self._index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & _UINT32_MASK

    def _twist(self) -> None:
        """Regenerate all 624 words of the MT19937 state."""
        state = self._state
        for index in range(_MT_SIZE - _MT_OFFSET):
            value = (state[index] & _MT_UPPER_MASK) | (
                state[index + 1] & _MT_LOWER_MASK
            )
            state[index] = (
                state[index + _MT_OFFSET]
                ^ (value >> 1)
                ^ (_MT_MATRIX_A if value & 1 else 0)
            )
        for index in range(_MT_SIZE - _MT_OFFSET, _MT_SIZE - 1):
            value = (state[index] & _MT_UPPER_MASK) | (
                state[index + 1] & _MT_LOWER_MASK
            )
            state[index] = (
                state[index + _MT_OFFSET - _MT_SIZE]
                ^ (value >> 1)
                ^ (_MT_MATRIX_A if value & 1 else 0)
            )
        value = (state[-1] & _MT_UPPER_MASK) | (state[0] & _MT_LOWER_MASK)
        state[-1] = (
            state[_MT_OFFSET - 1] ^ (value >> 1) ^ (_MT_MATRIX_A if value & 1 else 0)
        )
        self._index = 0
