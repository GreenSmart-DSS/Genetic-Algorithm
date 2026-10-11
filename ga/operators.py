"""Selection, crossover and mutation operators for the GA.

Every operator works on :class:`Chromosome` objects and takes a numpy
``RandomState`` (or ``Generator``) so runs are reproducible. Lower fitness is
always better everywhere in this module.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .genome import Chromosome

# Type alias: numpy PRNG accepted everywhere (RandomState or Generator).
rng_fn = np.random.RandomState


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _clip(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))


def _rand(rng) -> float:
    return float(rng.uniform(0.0, 1.0))


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def selection_tournament(
    population: List[Chromosome],
    fitness: List[float],
    tournament_size: int,
    rng: rng_fn,
) -> List[Chromosome]:
    """Tournament selection; returns ``len(population)`` parents (with replacement)."""
    n = len(population)
    selected: List[Chromosome] = []
    for _ in range(n):
        contenders = rng.randint(0, n, size=tournament_size)
        best = int(contenders[0])
        for c in contenders[1:]:
            if fitness[int(c)] < fitness[best]:
                best = int(c)
        selected.append(population[best])
    return selected


def selection_roulette(
    population: List[Chromosome], fitness: List[float], rng: rng_fn
) -> List[Chromosome]:
    """Roulette-wheel selection using inverse fitness (lower fitness = bigger slice)."""
    inv = [1.0 / f if f > 1e-12 else 1e12 for f in fitness]
    total = sum(inv)
    selected: List[Chromosome] = []
    n = len(population)
    for _ in range(n):
        r = _rand(rng) * total
        acc = 0.0
        chosen = population[-1]
        for i, w in enumerate(inv):
            acc += w
            if acc >= r:
                chosen = population[i]
                break
        selected.append(chosen)
    return selected


def selection_rank(
    population: List[Chromosome], fitness: List[float], rng: rng_fn
) -> List[Chromosome]:
    """Rank-based selection; linear weight from best (n) to worst (1)."""
    order = sorted(range(len(population)), key=lambda i: fitness[i])
    n = len(population)
    weights = [n - i for i in range(n)]
    total = sum(weights)
    selected: List[Chromosome] = []
    for _ in range(n):
        r = _rand(rng) * total
        acc = 0.0
        chosen = population[order[-1]]
        for i, idx in enumerate(order):
            acc += weights[i]
            if acc >= r:
                chosen = population[idx]
                break
        selected.append(chosen)
    return selected


SELECTION_METHODS: Dict[str, Callable] = {
    "tournament": selection_tournament,
    "roulette": selection_roulette,
    "rank": selection_rank,
}


def select_parents(
    population: List[Chromosome],
    fitness: List[float],
    method: str,
    tournament_size: int,
    rng: rng_fn,
) -> List[Chromosome]:
    """General selection entrypoint (lower fitness is always better)."""
    fn = SELECTION_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unsupported selection method: {method}")
    if method == "tournament":
        return fn(population, fitness, tournament_size, rng)
    return fn(population, fitness, rng)


# --------------------------------------------------------------------------
# crossover
# --------------------------------------------------------------------------

def _is_sequence(ch: Chromosome, i: int) -> bool:
    return ch.is_sequence(i)


def crossover_uniform(parent_a: Chromosome, parent_b: Chromosome, rng: rng_fn) -> None:
    """Uniform crossover (per-block swap with p=0.5); sequence blocks shared as-is."""
    for i in range(len(parent_a.values)):
        if _is_sequence(parent_a, i):
            continue  # untouched for matrix/permutation
        if _rand(rng) < 0.5:
            parent_a.values[i], parent_b.values[i] = parent_b.values[i], parent_a.values[i]


def crossover_single_point(
    parent_a: Chromosome, parent_b: Chromosome, rng: rng_fn
) -> None:
    """Single-point crossover at a random cut; sequence blocks handled separately."""
    n = len(parent_a.values)
    if n <= 1:
        return
    cut = int(rng.randint(1, n))
    for i in range(cut, n):
        if _is_sequence(parent_a, i):
            continue
        parent_a.values[i], parent_b.values[i] = parent_b.values[i], parent_a.values[i]


def crossover_two_point(
    parent_a: Chromosome, parent_b: Chromosome, rng: rng_fn
) -> None:
    """Two-point crossover between two random cuts (sequence blocks untouched)."""
    n = len(parent_a.values)
    if n <= 2:
        return
    cuts = sorted(rng.choice(n, 2, replace=False).tolist())
    for i in range(cuts[0], cuts[1]):
        if _is_sequence(parent_a, i):
            continue
        parent_a.values[i], parent_b.values[i] = parent_b.values[i], parent_a.values[i]


def crossover_sbx(
    parent_a: Chromosome, parent_b: Chromosome, rng: rng_fn, eta: float = 1.0
) -> None:
    """Simulated Binary Crossover (SBX) for float/int blocks."""
    for i in range(len(parent_a.values)):
        if _is_sequence(parent_a, i):
            continue
        if parent_a.dtypes[i] not in ("float", "int"):
            continue
        x1, x2 = float(parent_a.values[i]), float(parent_b.values[i])
        lb, ub = parent_a.bounds[i]
        if abs(x1 - x2) < 1e-12:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        d = x2 - x1
        u = _rand(rng)
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1 = _clip(0.5 * ((x1 + x2) - beta * d), lb, ub)
        c2 = _clip(0.5 * ((x1 + x2) + beta * d), lb, ub)
        if parent_a.dtypes[i] == "int":
            parent_a.values[i] = int(round(c1))
            parent_b.values[i] = int(round(c2))
        else:
            parent_a.values[i] = c1
            parent_b.values[i] = c2


def crossover_blend(
    parent_a: Chromosome, parent_b: Chromosome, rng: rng_fn, alpha: float = 0.5
) -> None:
    """BLX-alpha blend crossover for float/int blocks (continuous-friendly)."""
    for i in range(len(parent_a.values)):
        if _is_sequence(parent_a, i):
            continue
        if parent_a.dtypes[i] not in ("float", "int"):
            continue
        lb, ub = parent_a.bounds[i]
        lo = min(parent_a.values[i], parent_b.values[i])
        hi = max(parent_a.values[i], parent_b.values[i])
        d = hi - lo
        c1 = _clip(rng.uniform(lo - alpha * d, hi + alpha * d), lb, ub)
        c2 = _clip(rng.uniform(lo - alpha * d, hi + alpha * d), lb, ub)
        if parent_a.dtypes[i] == "int":
            parent_a.values[i] = int(round(c1))
            parent_b.values[i] = int(round(c2))
        else:
            parent_a.values[i] = c1
            parent_b.values[i] = c2


def crossover_pmx(parent_a: Chromosome, parent_b: Chromosome, rng: rng_fn) -> None:
    """Partially-matched crossover (PMX) for permutation blocks."""
    for i in range(len(parent_a.values)):
        if parent_a.dtypes[i] != "permutation":
            continue
        a = list(parent_a.values[i])
        b = list(parent_b.values[i])
        n = len(a)
        if n < 3:
            continue
        start = int(rng.randint(0, n - 1))
        end = int(rng.randint(start + 1, n))
        child = [None] * n
        child[start:end] = a[start:end]
        mapping = {}
        for j in range(start, end):
            mapping[b[j]] = a[j]
        for j in range(n):
            if child[j] is None:
                val = b[j]
                while val in mapping:
                    val = mapping[val]
                child[j] = val
        parent_a.values[i] = child
        parent_b.values[i] = child


def crossover_order(parent_a: Chromosome, parent_b: Chromosome, rng: rng_fn) -> None:
    """Order crossover (OX) for permutation blocks."""
    for i in range(len(parent_a.values)):
        if parent_a.dtypes[i] != "permutation":
            continue
        a = list(parent_a.values[i])
        b = list(parent_b.values[i])
        n = len(a)
        if n < 3:
            continue
        start = int(rng.randint(0, n - 1))
        end = int(rng.randint(start + 1, n + 1))
        seg = a[start:end]
        child = [None] * n
        child[start:end] = seg
        fill = [v for v in b if v not in seg]
        pos = end % n
        for v in fill:
            while child[pos] is not None:
                pos = (pos + 1) % n
            child[pos] = v
            pos = (pos + 1) % n
        parent_a.values[i] = child
        parent_b.values[i] = child


CROSSOVER_METHODS: Dict[str, Callable] = {
    "uniform": crossover_uniform,
    "single_point": crossover_single_point,
    "two_point": crossover_two_point,
    "sbx": crossover_sbx,
    "blend": crossover_blend,
    "pmx": crossover_pmx,
    "order": crossover_order,
}


def perform_crossover(
    parent_a: Chromosome, parent_b: Chromosome, method: str, rng: rng_fn
) -> None:
    fn = CROSSOVER_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unsupported crossover method: {method}")
    fn(parent_a, parent_b, rng)


# --------------------------------------------------------------------------
# mutation
# --------------------------------------------------------------------------

def _mutate_scalar(ch: Chromosome, i: int, rng: rng_fn, prob: float, scale: float) -> None:
    dtype = ch.dtypes[i]
    lb, ub = ch.bounds[i]
    r = _rand(rng)
    if dtype == "float":
        if r < prob:
            ch.values[i] = _clip(ch.values[i] + rng.normal(0.0, scale * (ub - lb)), lb, ub)
    elif dtype == "int":
        if r < prob and (ub - lb) >= 1:
            ch.values[i] = int(rng.randint(int(lb), int(ub) + 1))
    elif dtype == "binary":
        if r < prob:
            ch.values[i] = 1 - ch.values[i]


def _mutate_sequence(ch: Chromosome, i: int, rng: rng_fn, prob: float) -> None:
    """Per-position categorical re-sample for matrix; swap mutation for permutation."""
    if ch.dtypes[i] == "permutation":
        seq = list(ch.values[i])
        n = len(seq)
        for _ in range(max(1, int(round(prob * n)))):
            a, b = rng.choice(n, 2, replace=False)
            seq[a], seq[b] = seq[b], seq[a]
        ch.values[i] = seq
    else:  # matrix -- categorical re-sample
        seq = list(ch.values[i])
        lb, ub = ch.bounds[i]
        n = len(seq)
        for j in range(n):
            if _rand(rng) < prob:
                seq[j] = int(rng.randint(int(lb), int(ub) + 1))
        ch.values[i] = seq


def mutate(ch: Chromosome, rng: rng_fn, prob: float, scale: float = 0.1) -> None:
    """Apply per-block mutation to ``ch`` in place.

    ``prob`` is the per-gene mutation probability (scalars) or per-element
    probability (sequences); ``scale`` controls the Gaussian step for floats.
    """
    for i in range(len(ch.values)):
        if ch.is_sequence(i):
            _mutate_sequence(ch, i, rng, prob)
        else:
            _mutate_scalar(ch, i, rng, prob, scale)