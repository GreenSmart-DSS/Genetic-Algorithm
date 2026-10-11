"""Command-line entry point for the genetic algorithm.

Run::

    python main.py --config config.yaml [--seed 42] [--outdir results]

Requires PyYAML (and matplotlib if ``--plot``). See README.md for the config
guide and how to plug in your own objective function.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genetic-algorithm",
        description="Config-driven genetic algorithm for arbitrary optimization problems.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed for a reproducible run.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory for history.csv / best_solution.json / PNG plots.",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        action="store_true",
        dest="plot",
        default=None,
        help="Force saving a convergence plot.",
    )
    plot_group.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable the convergence plot.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,
        help="Print per-generation progress.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        dest="verbose",
        help="Suppress per-generation progress output.",
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        default=None,
        help="Write history to history.csv.",
    )
    parser.add_argument(
        "--no-save-history",
        action="store_false",
        dest="save_history",
        help="Do not write history to history.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from ga.algorithm import GeneticAlgorithm
        from ga.config import GASettings
        from ga.problem import load_objective
        from ga.utils import save_best
    except ImportError as exc:
        print(f"Fatal: could not import ga package: {exc}", file=sys.stderr)
        return 2

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        settings = GASettings.from_yaml(str(config_path))
        if args.seed is not None:
            settings.seed = args.seed
        objective_fn = load_objective(
            settings.objective_file, settings.objective_function
        )
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    ga = GeneticAlgorithm(settings, objective_fn)
    try:
        result = ga.run(
            plot=args.plot,
            verbose=args.verbose,
            outdir=args.outdir,
            save_history=args.save_history,
        )
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    # Always persist the best solution; most useful artifact.
    try:
        outdir = args.outdir or settings.outdir
        summary = save_best(result, outdir)
    except Exception as exc:
        print(f"Warning: could not save best solution: {exc}", file=sys.stderr)

    print("\n=== Result ===")
    print(f"Best fitness : {result.best_fitness:.6g}")
    print(f"Found at gen : {result.generation}")
    print(f"Early stop   : {result.stopped_early}")
    print("Solution:")
    payload = summary if "summary" in locals() else {}
    solution = payload.get("solution", {})
    for name, value in solution.items():
        print(f"    {name} = {value}")
    print(f"\nArtifacts written under: {args.outdir or settings.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
