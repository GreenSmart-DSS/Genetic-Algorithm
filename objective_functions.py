"""Built-in example objective functions.

Each function receives the *decoded* chromosome: a dict mapping every variable
name (from the YAML ``variables`` section) to its current value. Return a plain
float. "Better" (lower for ``sense: minimize``) is decided by the config, not
here.

To use one, point ``objective_file`` at this file (or copy it into your project)
and set ``objective_function`` to the function name, e.g.::

    objective_file: objective_functions.py
    objective_function: rastrigin
"""

from __future__ import annotations

import math


def sphere(x):
    """Classic bowl test: sum of squares. Minimum 0 at the origin."""
    return sum(v * v for v in x.values())


def rastrigin(x):
    """Many-modal test function. Minimum 0 at the origin."""
    n = len(x)
    return 10.0 * n + sum(
        v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in x.values()
    )


def rosenbrock(x):
    """Banana-shaped valley. Minimum 0 at (1, 1, ..., 1)."""
    vals = list(x.values())
    total = 0.0
    for i in range(len(vals) - 1):
        total += 100.0 * (vals[i + 1] - vals[i] ** 2) ** 2 + (1.0 - vals[i]) ** 2
    return total


def schwefel(x):
    """Deceptive test function; optimum near f(420.9687*ones) ≈ 0."""
    return 418.9829 * len(x) - sum(v * math.sin(math.sqrt(abs(v))) for v in x.values())


# Fixed benchmark instance so the TSP example is deterministic.
TSP_15_CITIES = [
    (0, 0),
    (34, 4),
    (25, 12),
    (3, 9),
    (10, 24),
    (47, 29),
    (23, 39),
    (40, 47),
    (15, 41),
    (36, 57),
    (9, 55),
    (28, 66),
    (33, 72),
    (45, 79),
    (20, 87),
]


def tsp_15_cities(x):
    """Total round-trip distance for the fixed 15-city instance.

    Expects a single permutation block named ``route``: the decoded ``route``
    is a permutation of city indices 0..14 in visit order.
    """
    import math

    route = x["route"]
    total = 0.0
    for i in range(len(route)):
        a, b = route[i], route[(i + 1) % len(route)]
        xa, ya = TSP_15_CITIES[a]
        xb, yb = TSP_15_CITIES[b]
        total += math.hypot(xa - xb, ya - yb)
    return total


def make_tsp_objective(coords):
    """Return a TSP objective bound to arbitrary 2D coordinates.

    Usage::

        tsp_obj = make_tsp_objective([(0, 0), (1, 0), ...])
        # then pass tsp_obj as objective_function (programmatic use)

    ``coords`` is a list of (x, y) positions; city indices are list positions.
    """

    def _objective(x):
        import math

        route = x["route"]
        total = 0.0
        for i in range(len(route)):
            a, b = route[i], route[(i + 1) % len(route)]
            xa, ya = coords[a]
            xb, yb = coords[b]
            total += math.hypot(xa - xb, ya - yb)
        return total

    return _objective


def constrained_linear(x):
    """Minimize a + b subject to a + b >= 8 (dimension-bearing example).

    Used with a constraint function returning the numeric violation so the GA
    pushes feasible solutions toward the boundary. See
    config_constrained_example.yaml for wiring.
    """
    return x["a"] + x["b"] + 0.5 * x["c"]


def constrained_linear_constraints(x):
    """Constraint function: returns {name: (current_value, target, tolerance)}."""
    return {
        "sum_ge_8": (x["a"] + x["b"], 8.0, 0.5),  # a+b must reach 8 (±0.5)
        "c_leq_2": (x["c"], 2.0, 0.0),  # c must be <= 2 (target 2)
    }


def ackley(x):
    """Multimodal test with a nearly flat outer region and a deep hole.
    Global minimum 0 at the origin. Typical domain: [-32, 32].
    """
    vals = list(x.values())
    n = len(vals)
    sum_sq = sum(v * v for v in vals)
    sum_cos = sum(math.cos(2.0 * math.pi * v) for v in vals)
    return (
        -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / n))
        - math.exp(sum_cos / n)
        + 20.0
        + math.e
    )


def griewank(x):
    """Multimodal with many regularly spaced local minima.
    Global minimum 0 at the origin. Typical domain: [-600, 600].
    """
    vals = list(x.values())
    sum_sq = sum(v * v for v in vals) / 4000.0
    prod_cos = 1.0
    for i, v in enumerate(vals, start=1):
        prod_cos *= math.cos(v / math.sqrt(i))
    return 1.0 + sum_sq - prod_cos


def levy(x):
    """Multimodal with a narrow global basin.
    Global minimum 0 at (1, 1, ..., 1). Typical domain: [-10, 10].
    """
    vals = list(x.values())
    n = len(vals)
    w = [1.0 + (v - 1.0) / 4.0 for v in vals]

    term1 = math.sin(math.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    term2 = 0.0
    for i in range(n - 1):
        term2 += (w[i] - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * w[i] + 1.0) ** 2)
    return term1 + term2 + term3


# ------------------------------------------------------------
# Agricultural TSP: irrigation monitoring route
# Coordinates are in meters.
# Node 0 = farm depot / monitoring station.
# Nodes 1..14 = irrigation monitoring points.
# ------------------------------------------------------------

AGRI_TSP_15_POINTS = [
    (50, 50),  # 0  Depot
    (180, 120),  # 1  Block A1
    (320, 70),  # 2  Block A2
    (470, 150),  # 3  Block A3
    (620, 90),  # 4  Block A4
    (760, 190),  # 5  Block A5
    (980, 120),  # 6  Block B1
    (1100, 300),  # 7  Block B2
    (900, 390),  # 8  Block B3
    (700, 330),  # 9  Block B4
    (610, 560),  # 10 Block C1
    (430, 460),  # 11 Block C2
    (260, 610),  # 12 Block C3
    (120, 480),  # 13 Block D1
    (70, 280),  # 14 Block D2
]


def agricultural_tsp_15(x):
    """Minimize the round-trip travel distance of an agricultural
    monitoring vehicle visiting 14 irrigation monitoring points.

    Node 0 is the farm depot.
    Nodes 1..14 are monitoring/irrigation points.

    The chromosome must contain a permutation block named ``route``.
    Coordinates are in meters, so the objective value is in meters.
    """
    route = x["route"]

    total = 0.0

    for i in range(len(route)):
        a = route[i]
        b = route[(i + 1) % len(route)]

        xa, ya = AGRI_TSP_15_POINTS[a]
        xb, yb = AGRI_TSP_15_POINTS[b]

        total += math.hypot(xa - xb, ya - yb)

    return total


class Objective:
    """Lightweight adapter for stateful objectives (e.g. from data).

    Wrap a closure that captures precomputed data::

        obj = Objective(lambda x: tsp_with_coords(coords)(x))

    ``Objective`` instances are callable and can be referenced from the YAML by
    name when loaded programmatically.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)


__all__ = [
    "Objective",
    "ackley",
    "agricultural_tsp_15",
    "constrained_linear",
    "constrained_linear_constraints",
    "griewank",
    "levy",
    "make_tsp_objective",
    "rastrigin",
    "rosenbrock",
    "schwefel",
    "sphere",
    "tsp_15_cities",
]
