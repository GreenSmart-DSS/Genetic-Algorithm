"""Population factory -- builds a random initial population from YAML variables."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .genome import Chromosome

SUPPORTED_DTYPES = ("float", "int", "binary", "matrix", "permutation")


def _validate_row(row: Dict[str, Any], index: int) -> None:
    if "name" not in row or not row["name"]:
        raise ValueError(f"Variable #{index} is missing a 'name'.")
    if "dtype" not in row or row["dtype"] not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Variable '{row.get('name', index)}': 'dtype' must be one of "
            f"{', '.join(SUPPORTED_DTYPES)}."
        )


def _parse_bounds(row: Dict[str, Any]) -> tuple[Any, Any]:
    """Return a (lower, upper) pair for a scalar variable.

    * both bounds given -> (lb, ub)
    * only lower given  -> (lb, lb + 1)  -- gives the variable room to vary
    * otherwise         -> (0, 1)
    """
    lb = row.get("lb", 0.0)
    ub = row.get("ub")
    ub = lb + 1.0 if ub is None else ub
    if lb > ub:
        raise ValueError(f"Variable '{row['name']}': lb ({lb}) must be <= ub ({ub}).")
    return lb, ub


def _make_population(
    pop_size: int, variables: List[Dict[str, Any]], rng: np.random.RandomState
) -> List[Chromosome]:
    population: List[Chromosome] = []

    for _ in range(pop_size):
        values: List[Any] = []
        bounds: List[tuple[Any, Any]] = []
        dtypes: List[str] = []
        names: List[str] = []
        seq_lengths: List[Optional[int]] = []

        for i, row in enumerate(variables):
            _validate_row(row, i)
            name = row["name"]
            dtype = row["dtype"]

            if dtype == "float":
                lb, ub = _parse_bounds(row)
                values.append(float(rng.uniform(lb, ub)))
                bounds.append((lb, ub))
                dtypes.append(dtype)
                names.append(name)
                seq_lengths.append(None)

            elif dtype == "int":
                lb, ub = _parse_bounds(row)
                values.append(int(rng.randint(int(lb), int(ub) + 1)))
                bounds.append((int(lb), int(ub)))
                dtypes.append(dtype)
                names.append(name)
                seq_lengths.append(None)

            elif dtype == "binary":
                lb, ub = _parse_bounds(row)
                values.append(int(rng.randint(0, 2)))
                bounds.append((int(lb), int(ub)))
                dtypes.append(dtype)
                names.append(name)
                seq_lengths.append(None)

            elif dtype == "matrix":
                vocab: List[Any]
                if row.get("values"):
                    vocab = list(row["values"])
                    high = len(vocab)
                else:
                    high = int(row.get("high", 2))
                    if high < 2:
                        raise ValueError(f"Variable '{name}': matrix 'high' must be >= 2.")
                    vocab = list(range(high))
                length = int(row.get("size", 1))
                if length < 1:
                    raise ValueError(f"Variable '{name}': matrix 'size' must be >= 1.")
                seq = [int(int(rng.randint(0, high))) for _ in range(length)]
                values.append(seq)
                bounds.append((0, high - 1))
                dtypes.append(dtype)
                names.append(name)
                seq_lengths.append(length)

            elif dtype == "permutation":
                size = int(row.get("size", len(row.get("values", [1]))))
                if size < 2:
                    raise ValueError(f"Variable '{name}': permutation 'size' must be >= 2.")
                # Use explicit int conversion to strip numpy types from the permutation.
                values.append([int(i) for i in rng.permutation(size)])
                bounds.append((0, size - 1))
                dtypes.append(dtype)
                names.append(name)
                seq_lengths.append(size)

        population.append(
            Chromosome(values, bounds, dtypes, names, seq_lengths)
        )
    return population


def random_population(
    pop_size: int, variables: List[Dict[str, Any]], rng: np.random.RandomState
) -> List[Chromosome]:
    return _make_population(pop_size, variables, rng)