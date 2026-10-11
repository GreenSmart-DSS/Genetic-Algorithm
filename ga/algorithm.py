"""The main genetic algorithm loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from . import operators as ops
from .genome import Chromosome
from .population import random_population


@dataclass
class EvolutionResult:
    """Final outcome of a GA run."""

    best_chromosome: Chromosome
    best_fitness: float
    history: List[float]          # best-so-far fitness per generation
    generation: int               # generation at which best was found
    stopped_early: bool           # True when no_improve_limit fired
    converged: bool               # True when population collapsed


def _fitness_sort(items: List[tuple[Chromosome, float]]) -> List[tuple[Chromosome, float]]:
    """Sort (chromosome, fitness) ascending by fitness (lower is better)."""
    return sorted(items, key=lambda pair: pair[1])


def _strict_ga(
    pop_size: int,
    max_generations: int,
    variables: List[Dict[str, Any]],
    evaluate: Callable,
    selection_fn: Callable,
    crossover_fn: Callable,
    mutation_fn: Callable,
    crossover_rate: float,
    elitism: int,
    no_improve_limit: int,
    rng: np.random.RandomState,
    verbose: bool = False,
    return_history: bool = False,
) -> EvolutionResult:
    """Lower-level GA loop; see :func:`GeneticAlgorithm.run` for the public API."""
    pop = random_population(pop_size, variables, rng)
    fitness = [evaluate(ch) for ch in pop]
    items = list(zip(pop, fitness))

    best_ever: Optional[Chromosome] = None
    best_ever_fitness: float = float("inf")
    history: List[float] = []
    stall = 0
    early_stop = False
    converged = False
    converged_at = 0
    final_gen = 0

    for gen in range(max_generations):
        final_gen = gen
        best_now = min(items, key=lambda it: it[1])
        if best_now[1] < best_ever_fitness:
            best_ever = best_now[0].copy()
            best_ever_fitness = best_now[1]
            stall = 0
        else:
            stall += 1

        history.append(best_ever_fitness)

        if verbose:
            print(f"[gen {gen:>4}] best={best_ever_fitness:.6g} "
                  f"mean={np.mean([it[1] for it in items]):.6g} "
                  f"(stall {stall})")

        # Early-stop on no improvement.
        if no_improve_limit > 0 and stall >= no_improve_limit:
            early_stop = True
            best_ever = best_now[0].copy()  # keep the reproducing best now
            break

        # Convergence test: all individuals have the same fitness value.
        if len({it[1] for it in items}) == 1:
            converged = True
            converged_at = gen + 1
            best_ever = best_now[0].copy()
            break

        # Preserve the N best chromosomes (elitism).
        items_sorted = _fitness_sort(items)
        elites = [(ch.copy(), fit) for ch, fit in items_sorted[:elitism]]

        # Select parents.
        parents = selection_fn([it[0] for it in items], [it[1] for it in items])

        offspring: List[Chromosome] = []
        n = pop_size
        idx = 0
        while len(offspring) < n:
            p1 = parents[idx % len(parents)]
            p2 = parents[(idx + 1) % len(parents)]
            idx += 2
            c1, c2 = p1.copy(), p2.copy()
            if rng.uniform() < crossover_rate:
                crossover_fn(c1, c2)
            mutation_fn(c1)
            mutation_fn(c2)
            offspring.append(c1)
            if len(offspring) < n:
                offspring.append(c2)

        # Compose next generation: elites + offspring.
        next_pop = [ch for ch, _ in elites] + offspring[: n - len(elites)]
        items = list(zip(next_pop, [evaluate(ch) for ch in next_pop]))

    # Ensure a best-ever was recorded even if nothing ever improved.
    if best_ever is None:
        best_item = min(items, key=lambda it: it[1])
        best_ever = best_item[0].copy()
        best_ever_fitness = best_item[1]

    return EvolutionResult(
        best_chromosome=best_ever,
        best_fitness=best_ever_fitness,
        history=history,
        generation=final_gen,
        stopped_early=early_stop or converged,
        converged=converged,
    )


class GeneticAlgorithm:
    """Configurably-driven GA (defaults from YAML, overridable programmatically).

    Typical usage::

        ga = GeneticAlgorithm(settings)          # from GASettings
        result = ga.run(plot=False, verbose=True)
    """

    def __init__(self, settings: Any, objective_fn: Callable) -> None:
        from .config import GASettings
        if not isinstance(settings, GASettings):
            raise TypeError("settings must be a GASettings instance.")
        self.settings = settings
        self.objective_fn = objective_fn
        self.problem = None  # built in run()

    def _make_rng(self, seed: Optional[int]) -> np.random.RandomState:
        if seed is not None:
            return np.random.RandomState(seed)
        return np.random.RandomState()

    def run(
        self,
        plot: Optional[bool] = None,
        verbose: Optional[bool] = None,
        outdir: Optional[str] = None,
        save_history: Optional[bool] = None,
    ) -> EvolutionResult:
        """Execute the GA and return the :class:`EvolutionResult`."""
        from .problem import Problem, load_constraint_fn

        plot = plot if plot is not None else self.settings.plot
        verbose = verbose if verbose is not None else self.settings.verbose
        outdir = outdir if outdir is not None else self.settings.outdir
        save_history = save_history if save_history is not None else self.settings.save_history

        rng = self._make_rng(self.settings.seed)

        constraint_fn = load_constraint_fn(
            self.settings.objective_file, self.settings.constraint_function
        )
        self.problem = Problem(
            objective_fn=self.objective_fn,
            sense=self.settings.sense,
            constraint_fn=constraint_fn,
            penalty_weight=self.settings.constraint_penalty_weight,
        )

        def _selection(chromosomes, fitnesses):
            return ops.select_parents(
                chromosomes, fitnesses,
                method=self.settings.selection,
                tournament_size=self.settings.tournament_size,
                rng=rng,
            )

        crossover_fn = lambda a, b: ops.perform_crossover(a, b, self.settings.crossover, rng)
        mutation_fn = lambda ch: ops.mutate(ch, rng, self.settings.mutation_rate)

        problem = self.problem

        result = _strict_ga(
            pop_size=self.settings.pop_size,
            max_generations=self.settings.max_generations,
            variables=self.settings.variables,
            evaluate=lambda ch: problem.evaluate(ch),
            selection_fn=_selection,
            crossover_fn=crossover_fn,
            mutation_fn=mutation_fn,
            crossover_rate=self.settings.crossover_rate,
            elitism=self.settings.elitism,
            no_improve_limit=self.settings.no_improve_limit,
            rng=rng,
            verbose=verbose,
        )
        if save_history:
            self._save_outputs(result, outdir, plot)
        return result

    def _save_outputs(self, result: EvolutionResult, outdir: str, plot: bool) -> None:
        import csv
        import os

        os.makedirs(outdir, exist_ok=True)
        csv_path = os.path.join(outdir, "history.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["generation", "best_fitness"])
            for gen, fit in enumerate(result.history):
                writer.writerow([gen, fit])

        if plot:
            try:
                from .utils import plot_history
                plot_history(result.history, outdir)
            except Exception as exc:
                if verbose := self.settings.verbose:
                    print(f"Could not create plot: {exc}")