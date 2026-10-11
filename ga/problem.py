"""Problem wrapper: ties the objective function to penalty handling and constraints.

The objective function is loaded from a user file as a plain Python callable.
It receives a *decoded* chromosome (a dict of ``name -> value``) and must return
a single numeric fitness *before* penalties. ``objective_function`` may also be
set to a fully-qualified dotted name (``package.module.function``) instead of a
file+name pair for the rare case the user already has the function importable.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Callable, Dict, Optional


class ObjectiveNotFound(ValueError):
    """Raised when the objective function cannot be located/imported."""


def _locate_module_file(path: str) -> ModuleType:
    """Import a Python file as a module by its filesystem path."""
    import importlib.util
    import os
    import sys

    abs_path = os.path.abspath(path)
    module_name = f"_user_objective_{os.path.splitext(os.path.basename(path))[0]}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ObjectiveNotFound(f"Could not create import spec for '{path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - propagation detail
        raise ObjectiveNotFound(f"Failed importing objective file '{path}': {exc}") from exc
    return module


def _load_module(obj_file: Optional[str], dotted_name: Optional[str]) -> ModuleType:
    """Import a user module, either by file path or by dotted package name."""
    if dotted_name and "." in dotted_name:
        module_name = dotted_name.rsplit(".", 1)[0]
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise ObjectiveNotFound(
                f"Could not import module '{module_name}' (from '{dotted_name}'): {exc}"
            ) from exc
    if obj_file:
        return _locate_module_file(obj_file)
    raise ObjectiveNotFound("No objective_file and no dotted objective_function given.")


def _get_callable(module: ModuleType, name: str, what: str) -> Callable:
    fn = getattr(module, name, None)
    if not callable(fn):
        raise ObjectiveNotFound(f"'{name}' ({what}) is not a callable in {getattr(module, '__name__', str(module))}.")
    return fn


def load_objective(obj_file: Optional[str], obj_name: Optional[str]) -> Callable:
    """Resolve the objective callable.

    Priority (highest first):
      1. a dotted ``objective_function`` value that imports cleanly
      2. ``objective_file`` (a filesystem path) + ``objective_function`` name
      3. a single un-underscore-prefixed callable inside ``objective_file``
    """
    module = _load_module(obj_file, obj_name)
    if obj_name and "." in obj_name:
        func_name = obj_name.rsplit(".", 1)[1]
        return _get_callable(module, func_name, "objective")
    if obj_name:
        return _get_callable(module, obj_name, "objective")
    if obj_file:
        candidates = [
            v for k, v in vars(module).items()
            if callable(v) and not k.startswith("_")
        ]
        if len(candidates) == 1:
            return candidates[0]
        raise ObjectiveNotFound(
            f"objective_file '{obj_file}' has {len(candidates)} callables; "
            "set 'objective_function' to the one you want (e.g. rosenbrock)."
        )
    raise ObjectiveNotFound("No objective callable resolved.")


def load_constraint_fn(
    obj_file: Optional[str], con_name: Optional[str]
) -> Optional[Callable]:
    """Resolve an optional constraint function from the objective file (or by dotted name)."""
    if not con_name:
        return None
    parts = con_name.rsplit(".", 1)
    func_name = parts[1] if len(parts) > 1 else parts[0]
    return _get_callable(_load_module(obj_file, con_name), func_name, "constraint")


def _ensure_numeric(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"Objective returned a non-numeric value {value!r} ({context}). "
            "Objective functions must return a real number."
        )
    if value != value:  # NaN
        raise ValueError(f"Objective returned NaN ({context}).")
    return float(value)


class Problem:
    """Evaluates chromosomes for fitness; applies constraints via penalties.

    ``objective_fn`` receives a dict ``{name: value, ...}`` of decoded variables
    and returns the raw objective (higher or lower depending on ``sense``).
    """

    def __init__(
        self,
        objective_fn: Callable,
        sense: str = "minimize",
        constraint_fn: Optional[Callable] = None,
        penalty_weight: str = "auto",
    ) -> None:
        self.objective_fn = objective_fn
        self.sense = sense
        self.constraint_fn = constraint_fn
        self.penalty_weight = penalty_weight

    # -- raw objective -------------------------------------------------------
    def objective(self, decoded: Dict[str, Any]) -> float:
        value = self.objective_fn(decoded)
        return _ensure_numeric(value, "objective")

    # -- penalties -----------------------------------------------------------
    def constraint_penalty(self, decoded: Dict[str, Any], raw: float) -> tuple[float, list[str]]:
        """Return (penalty, diagnostics) from the user constraint function.

        The constraint function returns a dict of ``name -> (value, threshold,
        tolerance)`` (tolerance optional, default 0). The total violation is the
        sum of ``max(0, abs(value - target) - tol)`` over all entries.
        """
        if self.constraint_fn is None:
            return 0.0, []
        violations = self.constraint_fn(decoded)
        return self._compute_penalty(violations, raw)

    def _compute_penalty(self, violations_dict: Dict[str, Any], raw: float) -> tuple[float, list[str]]:
        total = 0.0
        issues: list[str] = []
        for name, spec in violations_dict.items():
            try:
                if isinstance(spec, (list, tuple)) and len(spec) >= 2:
                    value, target = spec[0], spec[1]
                    tol = spec[2] if len(spec) > 2 else 0.0
                else:
                    value = spec
                    target, tol = 0.0, 0.0
                if value is None:
                    continue
                violation = max(0.0, float(abs(value - target)) - float(tol))
                if violation > 1e-12:
                    issues.append(
                        f"{name}: {float(value):.4g} (target {float(target):.4g}, "
                        f"tol {float(tol):.4g}, viol {violation:.4g})"
                    )
                    total += violation
            except Exception as exc:
                raise ValueError(f"Bad constraint spec for '{name}': {exc}") from exc
        return self._scale_penalty(total, raw), issues

    def _scale_penalty(self, total: float, raw: float) -> float:
        """Deterministic penalty.

        * ``penalty_weight: auto`` -> ``total * (1 + abs(raw))``. A unit of
          violation always costs *more* than the raw objective, so any
          infeasible individual is worse than the best feasible one; while still
          scaling with the objective's magnitude. Re-evaluating any chromosome
          always yields the same fitness (no hidden running max).
        * ``penalty_weight: <float>`` -> ``w * total``.
        """
        if total <= 0:
            return 0.0
        if self.penalty_weight == "auto":
            return total * (1.0 + abs(raw))
        return float(self.penalty_weight) * total

    # -- fitness -------------------------------------------------------------
    def evaluate(self, ch: Any) -> float:
        """Evaluate one chromosome to a fitness value (lower is better)."""
        decoded = ch.decode()
        raw = self.objective(decoded)
        penalty, _ = self.constraint_penalty(decoded, raw)
        if self.sense == "maximize":
            return -raw + penalty  # negate so lower fitness = better
        return raw + penalty