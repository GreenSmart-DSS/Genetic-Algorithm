"""Small utilities: result persistence and convergence plotting."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

import numpy as np


def _to_serializable(value: Any) -> Any:
    """Convert numpy scalars/arrays to plain JSON-friendly types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _to_plain_list(values: list) -> list:
    """Strip numpy wrappers from chromosome values for display."""
    return [_to_serializable(v) for v in values]


def save_best(result: Any, outdir: str) -> Dict[str, Any]:
    """Persist the best chromosome + fitness to ``best_solution.json``."""
    os.makedirs(outdir, exist_ok=True)
    ch = result.best_chromosome
    payload = {
        "fitness": result.best_fitness,
        "generation": result.generation,
        "stopped_early": result.stopped_early,
        "converged": result.converged,
        "solution": {
            name: _to_serializable(val)
            for name, val in zip(ch.names, ch.values)
        },
    }
    path = os.path.join(outdir, "best_solution.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_to_serializable)
    return payload


def save_history_csv(history: List[float], outdir: str) -> str:
    """Write best-fitness-per-generation history to ``history.csv``."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "history.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generation", "best_fitness"])
        for gen, fit in enumerate(history):
            writer.writerow([gen, fit])
    return path


def plot_history(
    history: List[float],
    outdir: str,
    title: str = "GA convergence",
    filename: str = "convergence.png",
) -> str:
    """Plot best-so-far fitness vs generation; returns the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(history)), history, lw=2.0, color="#1565c0")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = ["save_best", "save_history_csv", "plot_history"]