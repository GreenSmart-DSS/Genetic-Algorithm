"""Individual chromosome representation.

A chromosome is a list of *blocks*. Each YAML ``variables`` row produces one
block:

* ``float`` / ``int`` / ``binary``  -> a single scalar value (within bounds)
* ``matrix``                        -> a sequence of category indices (repeats allowed)
* ``permutation``                   -> a permutation of indices ``0..size-1``

The block layout (bounds, dtypes, names, sequence lengths) travels with the
values so objective functions and operators can decode a chromosome without
re-reading the YAML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

SEQUENCE_DTYPES = ("matrix", "permutation")


class Chromosome:
    """A single candidate solution.

    ``values`` has one entry per YAML ``variables`` row: a scalar for
    ``float``/``int``/``binary``, or a list of ints for ``matrix``/``permutation``.
    """

    __slots__ = ("values", "bounds", "dtypes", "names", "seq_lengths")

    def __init__(
        self,
        values: Sequence[Any],
        bounds: Sequence[Tuple[Any, Any]],
        dtypes: Sequence[str],
        names: Sequence[str],
        seq_lengths: Optional[Sequence[Optional[int]]] = None,
    ):
        self.values: List[Any] = list(values)
        self.bounds: List[Tuple[Any, Any]] = list(bounds)
        self.dtypes: List[str] = list(dtypes)
        self.names: List[str] = list(names)
        if seq_lengths is None:
            seq_lengths = [
                len(v) if d in SEQUENCE_DTYPES else None for d, v in zip(self.dtypes, self.values)
            ]
        self.seq_lengths: List[Optional[int]] = list(seq_lengths)

    def is_sequence(self, i: int) -> bool:
        """True if block ``i`` is a matrix/permutation sequence."""
        return self.dtypes[i] in SEQUENCE_DTYPES

    # -- decoding -------------------------------------------------------------
    def decode(self) -> Dict[str, Any]:
        """Return block values keyed by variable name, e.g. ``obj(**genome.decode())``."""
        return dict(zip(self.names, self.values))

    def to_dict(self) -> Dict[str, Any]:
        """Full representational dict (values + layout)."""
        return {
            "values": self.values,
            "bounds": self.bounds,
            "dtypes": self.dtypes,
            "names": self.names,
            "seq_lengths": self.seq_lengths,
        }

    def copy(self) -> "Chromosome":
        """Independent deep copy of this chromosome."""
        return Chromosome(
            values=[v.copy() if isinstance(v, (list, tuple)) else v for v in self.values],
            bounds=self.bounds,
            dtypes=self.dtypes,
            names=self.names,
            seq_lengths=self.seq_lengths,
        )

    def to_indices(self, i: int) -> List[int]:
        """Return the sequence block ``i`` as a flat list of int indices."""
        return list(self.values[i])

    def __repr__(self) -> str:
        return f"Chromosome({self.values!r})"