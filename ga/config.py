"""YAML-driven configuration for a genetic algorithm run.

Every setting lives in config.yaml and is loaded via :func:`GASettings.from_yaml`.
Anything not present falls back to the documented defaults, so a minimal run
needs only the objective function file plus the ``variables`` section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

DEFAULTS: Dict[str, Any] = {
    # --- population / generations -------------------------------------------
    "pop_size": 100,
    "max_generations": 500,
    "no_improve_limit": 60,  # early stop; set to 0 to disable
    # --- selection -----------------------------------------------------------
    "selection": "tournament",  # "tournament" | "roulette" | "rank"
    "tournament_size": 3,
    "elitism": 2,  # N best individuals copied into next generation unchanged
    # --- genetic operators ---------------------------------------------------
    "crossover_rate": 0.9,
    "crossover": "blend",  # depends on variable types, see README
    "mutation_rate": 0.1,
    # --- objective -----------------------------------------------------------
    "sense": "minimize",  # "minimize" | "maximize"
    "objective_file": "objective_functions.py",
    "objective_function": "sphere",
    # optional constraint function (name in the same objective file)
    "constraint_function": None,
    "constraint_penalty_weight": "auto",  # "auto" or a float scale
    # --- random / output -----------------------------------------------------
    "seed": None,  # int for a reproducible run
    "save_history": True,
    "verbose": True,
    "plot": False,
    "outdir": "results",
}


class ConfigError(ValueError):
    """Raised when the YAML config is missing or contains invalid settings."""


def resolve_objective_file(obj_path: Optional[str]) -> Optional[str]:
    """Expand a shell-like path so objective files can be anywhere on disk."""
    if not obj_path:
        return None
    return str(Path(obj_path).expanduser())


def _validate_selection(value: str) -> str:
    if value not in ("tournament", "roulette", "rank"):
        raise ConfigError(
            f"Unsupported selection method '{value}'. "
            "Use 'tournament', 'roulette' or 'rank'."
        )
    return value


def _validate_crossover(value: str) -> str:
    known_crossovers = {"uniform", "single_point", "two_point", "sbx", "blend", "order", "pmx"}
    if value not in known_crossovers:
        raise ConfigError(
            f"Unsupported crossover '{value}'. Known: {', '.join(sorted(known_crossovers))}."
        )
    return value


def _validate_sense(value: str) -> str:
    if value not in ("minimize", "maximize"):
        raise ConfigError(f"Unsupported sense '{value}'. Use 'minimize' or 'maximize'.")
    return value


class GASettings:
    """Immutable GA configuration.

    Attributes mirror the YAML keys (see ``DEFAULTS``); refer to config.yaml
    directly for the human-readable description of every field.
    """

    __slots__ = (
        "pop_size", "max_generations", "no_improve_limit",
        "selection", "tournament_size", "elitism",
        "crossover_rate", "crossover", "mutation_rate",
        "sense", "objective_file", "objective_function",
        "constraint_function", "constraint_penalty_weight",
        "seed", "save_history", "verbose", "plot", "outdir",
        "variables",
    )

    def __init__(self, config: Dict[str, Any]) -> None:
        merged: Dict[str, Any] = {**DEFAULTS, **config}
        self.pop_size = int(merged["pop_size"])
        self.max_generations = int(merged["max_generations"])
        self.no_improve_limit = int(merged["no_improve_limit"])
        self.selection = _validate_selection(str(merged["selection"]))
        self.tournament_size = int(merged.get("tournament_size", DEFAULTS["tournament_size"]))
        self.elitism = int(merged["elitism"])
        self.crossover_rate = float(merged["crossover_rate"])
        self.crossover = _validate_crossover(str(merged["crossover"]))
        self.mutation_rate = float(merged["mutation_rate"])
        self.sense = _validate_sense(str(merged["sense"]))
        self.objective_file = resolve_objective_file(merged.get("objective_file"))
        self.objective_function = str(merged["objective_function"])
        self.constraint_function = merged.get("constraint_function")
        self.constraint_penalty_weight = merged.get("constraint_penalty_weight", "auto")
        self.seed = merged.get("seed")
        self.save_history = bool(merged["save_history"])
        self.verbose = bool(merged["verbose"])
        self.plot = bool(merged["plot"])
        self.outdir = str(merged["outdir"])
        self.variables = merged.get("variables", [])
        self._sanity_check()

    def _sanity_check(self) -> None:
        checks: Sequence[tuple[bool, str]] = (
            (self.pop_size >= 2, "pop_size must be >= 2"),
            (self.max_generations >= 1, "max_generations must be >= 1"),
            (0.0 <= self.crossover_rate <= 1.0, "crossover_rate must be within [0, 1]"),
            (0.0 <= self.mutation_rate <= 1.0, "mutation_rate must be within [0, 1]"),
            (self.elitism >= 0, "elitism must be >= 0"),
            (self.elitism <= self.pop_size, "elitism must be <= pop_size"),
            (self.seed is None or isinstance(self.seed, int), "seed must be an int or null"),
            (self.tournament_size >= 1, "tournament_size must be >= 1"),
            (self.no_improve_limit >= 0, "no_improve_limit must be >= 0"),
        )
        for ok, msg in checks:
            if not ok:
                raise ConfigError(msg)

    @classmethod
    def from_yaml(cls, path: str) -> "GASettings":
        """Load settings from a YAML file (relative to the current directory)."""
        raw = _read_yaml(path)
        return cls(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GASettings":
        return cls(raw)

    def __repr__(self) -> str:
        active: Dict[str, Any] = {}
        for name in self.__slots__:
            if getattr(self, name, None) is not None:
                active[name] = getattr(self, name)
        return f"GASettings({active})"


def _read_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")

    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {p}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"Config file is empty: {p}")
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must contain a mapping, got {type(raw).__name__}")

    # Normalize top-level keys so the settings are forgiving of casing/spacing.
    normalized: Dict[str, Any] = {}
    for key, value in raw.items():
        normalized[key.strip().lower()] = value
    return normalized