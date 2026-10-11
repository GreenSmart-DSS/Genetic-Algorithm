"""High-level convenience: one-call run from a YAML config file."""

from __future__ import annotations

from typing import Optional

from .algorithm import EvolutionResult, GeneticAlgorithm
from .config import GASettings
from .problem import load_objective


def run_from_config(
    config_path: str,
    seed: Optional[int] = None,
    plot: Optional[bool] = None,
    verbose: Optional[bool] = None,
    outdir: Optional[str] = None,
    save_history: Optional[bool] = None,
) -> EvolutionResult:
    """Load ``config_path`` and run a full GA, returning the result.

    All optional parameters override the corresponding YAML setting.
    """
    settings = GASettings.from_yaml(config_path)
    if seed is not None:
        settings.seed = seed

    objective_fn = load_objective(settings.objective_file, settings.objective_function)

    ga = GeneticAlgorithm(settings, objective_fn)
    return ga.run(
        plot=plot,
        verbose=verbose,
        outdir=outdir,
        save_history=save_history,
    )


__all__ = ["run_from_config"]