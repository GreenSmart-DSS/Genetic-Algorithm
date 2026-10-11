# Genetic Algorithm (config-driven)

A general-purpose genetic algorithm for Python — solve any optimization problem
by (1) writing one objective function and (2) editing a YAML file. No algorithm
code required.

## Features

- **Config-driven** — every GA parameter lives in YAML; nothing to code.
- **Mixed variable types** — `float`, `int`, `binary`, `matrix` (categorical
  sequence), and `permutation` (TSP-style ordering) in a single problem.
- **Multiple senses** — minimize or maximize.
- **Constraint handling** — optional constraint function, auto-scaled penalties.
- **Choice of operators** — 7 crossover methods, 3 selection methods,
  type-aware mutation.
- **Reproducible** — seed the RNG; re-running with the same seed yields the
  same result.
- **Artifacts** — best solution JSON, per-generation history CSV, convergence
  plot PNG.

## Project layout

```text
Genetic Algorithm/
├── ga/                       # the framework (do not edit)
│   ├── config.py             # YAML settings + validation
│   ├── genome.py             # chromosome representation
│   ├── population.py         # random initial population
│   ├── operators.py          # selection / crossover / mutation
│   ├── problem.py            # objective + constraint penalties
│   ├── algorithm.py          # main loop + elitism + early stop
│   ├── utils.py              # CSV / JSON / PNG output
│   └── common.py             # run_from_config() one-liner
├── main.py                   # CLI entry point
├── objective_functions.py    # objective + constraint examples live here
├── config_rosenbrock_example.yaml     # ROSENBROCK: 4 continuous, smooth valley
├── config_sphere_example.yaml         # SPHERE: 5 continuous, convex bowl
├── config_rastrigin_example.yaml      # RASTRIGIN: 5 continuous, many-modal
├── config_schwefel_example.yaml       # SCHWEFEL: 5 continuous, deceptive
├── config_ackley_example.yaml         # ACKLEY: 5 continuous, narrow basin
├── config_griewank_example.yaml       # GRIEWANK: 5 continuous, fractal basin
├── config_levy_example.yaml           # LEVY: 5 continuous, off-center optimum
├── config_tsp_example.yaml            # TSP: permutation, 15 cities
├── config_constrained_example.yaml    # constrained optimization
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# Continuous test functions
python main.py --config config_rosenbrock_example.yaml   # Rosenbrock
python main.py --config config_sphere_example.yaml       # Sphere
python main.py --config config_rastrigin_example.yaml    # Rastrigin
python main.py --config config_schwefel_example.yaml     # Schwefel

# Permutation
python main.py --config config_tsp_example.yaml          # 15-city TSP

# Constrained
python main.py --config config_constrained_example.yaml  # linear constraints

# Reproducible run + force a convergence plot
python main.py --config config_rosenbrock_example.yaml --seed 7 --plot
```

The CLI prints the final best solution; artifacts are written to the configured
`outdir`:

- `best_solution.json` — the best chromosome + fitness
- `history.csv` — best-so-far fitness per generation
- `convergence.png` — plotted convergence (only with `plot`)

## Bundled examples

| Config | Function | Variables | Highlights |
|--------|----------|-----------|------------|
| `config_rosenbrock_example.yaml` | Rosenbrock | 4 × float `[-10, 10]` | banana valley, narrow ridge |
| `config_sphere_example.yaml` | Sphere | 5 × float `[-10, 10]` | convex, easy baseline |
| `config_rastrigin_example.yaml` | Rastrigin | 5 × float `[-5.12, 5.12]` | dense local minima |
| `config_schwefel_example.yaml` | Schwefel | 5 × float `[-500, 500]` | deceptive, far optimum |
| `config_ackley_example.yaml` | Ackley | 5 × float `[-32, 32]` | flat with narrow hole |
| `config_griewank_example.yaml` | Griewank | 5 × float `[-600, 600]` | fractal multimodal bowl |
| `config_levy_example.yaml` | Levy | 5 × float `[-10, 10]` | off-center narrow optimum |
| `config_tsp_example.yaml` | TSP | 1 × permutation (15 cities) | order/PMX crossover |
| `config_constrained_example.yaml` | linear + constraints | 3 × float | penalty-based feasibility |

Each config is fully commented, explaining every parameter. Run one, then edit
`config_rosenbrock_example.yaml` and watch the convergence plot change.

## Solving your own problem

### 1. Write the objective function

Create a file (or reuse `objective_functions.py`):

```python
# my_objective.py
def my_function(x):
    """x is a dict: {variable_name: current_value, ...}."""
    a = x["width"]
    b = x["height"]
    # ... compute some number ...
    return a * b + 10
```

The function receives **only** the variables you declare in the YAML, by name,
and must return a single `float`/`int`. Whether small is better depends on the
`sense` setting — your function never deals with penalties or fitness.

For stateful objectives (e.g. a cost computed against preloaded data), build a
closure and return it. See `objective_functions.py` → `make_tsp_objective`.

### 2. Declare variables in the YAML

```yaml
variables:
  - { name: width,  dtype: float,       lb: 0.0, ub: 100.0 }
  - { name: height, dtype: int,         lb: 1,   ub: 50 }
  - { name: enabled, dtype: binary }
  - { name: material, dtype: matrix, size: 5, values: ["steel", "alu", "wood"] }
  - { name: route,   dtype: permutation, size: 15 }
```

| `dtype`       | Value passed to objective                       | Options                          |
|---------------|-------------------------------------------------|----------------------------------|
| `float`       | `float`                                         | `lb`, `ub` (default `0..1`)      |
| `int`         | `int`                                           | `lb`, `ub`                       |
| `binary`      | `int` (0 or 1)                                  | —                                |
| `matrix`      | `list[int]` of category indices, length `size`  | `size`, `values` OR `high`       |
| `permutation` | `list[int]` = ordering of `0..size-1`           | `size`                           |

Bounds are optional:

- `lb` and `ub` both given → used as-is (must be `lb <= ub`)
- only `lb` given → upper becomes `lb + 1` (room to vary)
- neither given → `0..1`

### 3. Configure and run

Point `objective_file` + `objective_function` at your function, then:

```bash
python main.py --config my_config.yaml
```

## Configuration reference

All keys are optional; missing keys fall back to the defaults. `objective_file`,
`objective_function` and `variables` are the only ones you really need.

| Key                 | Default       | Meaning                                                   |
|---------------------|---------------|-----------------------------------------------------------|
| `pop_size`          | `100`         | Individuals per generation                                |
| `max_generations`   | `500`         | Hard cap on generations                                   |
| `no_improve_limit`  | `60`          | Early stop after N gens without best improvement (`0` off)|
| `selection`         | `tournament`  | `tournament` \| `roulette` \| `rank`                      |
| `tournament_size`   | `3`           | Tournament contenders                                     |
| `elitism`           | `2`           | N best copied to next gen unchanged                       |
| `crossover_rate`    | `0.9`         | Probability a pair undergoes crossover                    |
| `crossover`         | `blend`       | `uniform` \| `single_point` \| `two_point` \| `sbx` \| `blend` \| `pmx` \| `order` |
| `mutation_rate`     | `0.1`         | Per-gene mutation probability                             |
| `sense`             | `minimize`    | `minimize` \| `maximize`                                  |
| `objective_file`    | —             | Path to `.py` with the objective                          |
| `objective_function`| —             | Callable name inside `objective_file`                     |
| `constraint_function` | —           | Optional constraint fn name in the same file (see below)  |
| `constraint_penalty_weight` | `auto` | `auto` or a float multiplier for violation total       |
| `seed`              | `null`        | Int → reproducible run                                    |
| `save_history`      | `true`        | Write `history.csv`                                       |
| `plot`              | `false`       | Render `convergence.png` (needs matplotlib)               |
| `verbose`           | `true`        | Print per-generation progress                             |
| `outdir`            | `results`     | Where artifacts are written                               |

### Crossover by variable type

Choose `crossover` once for the whole run; it is applied block-by-block and only
to matching block types:

- **float / int / binary** (scalar blocks): `uniform`, `single_point`,
  `two_point`, `sbx` (Simulated Binary), `blend` (BLX-α, continuous-friendly
  default). pmx/order do no work on scalars.
- **permutation** blocks: `order` (OX, good for adjacency — TSP), `pmx`
  (partially-mapped, position-preserving — assignment/Hamiltonian problems).
  Scalar crossovers do no work on permutations (a per-gene exchange would
  corrupt the permutation's validity).
- **matrix** blocks: treated as categorical; scalar crossovers pass the whole
  sequence through unrecombined, so use mutation or the permutation crossovers
  if you need sequence recombination.

### Mutation

`mutation_rate` is the per-gene probability. Applied per block:

- float → Gaussian step scaled by the variable's range
- int → random re-sample within bounds
- binary → flip
- permutation → swap (number scaled by `mutation_rate × size`)
- matrix → per-position categorical re-sample

## Constraints

Optional. Define a constraint function in the objective file that returns a dict
of `name -> (current_value, target, tolerance)` or `name -> current_value`
(tolerance 0):

```python
# my_objective.py
def constraints(x):
    return {
        "max_a_plus_b": (x["width"] + x["height"], 120.0, 0.0),
        "min_width": (x["width"], 5.0, 0.0),
    }
```

Then reference it in the YAML:

```yaml
constraint_function: constraints
constraint_penalty_weight: auto   # or a float to scale explicitly
```

The penalty is added to the raw objective (after negation for `maximize`) so
infeasible individuals are pushed down.

With `constraint_penalty_weight: auto` the violation total is scaled by
`(1 + |raw objective|)`, so each unit of violation always costs more than the
objective itself; a feasible individual is therefore never beaten by an
infeasible one. With an explicit float weight `w`, the penalty is
`w * total_violation`. The penalty is deterministic — re-evaluating any
chromosome always yields the same fitness.

## Command-line flags

All optional. When a flag is omitted, the corresponding YAML value is used.

```
--config <path>      Path to the YAML configuration file (required).
--seed <int>         Override the seed in the YAML.
--outdir <path>      Override the output directory.
--plot / --no-plot   Force or disable the convergence plot.
-v / -q              Force or suppress per-generation progress output.
--save-history / --no-save-history
                     Force or disable writing history.csv.
```

## Programmatic use

`run_from_config` loads everything from YAML and runs:

```python
from ga import run_from_config

result = run_from_config("config_rosenbrock_example.yaml")
print(result.best_chromosome.values, result.best_fitness)
```

Or drive the pieces directly:

```python
from ga import GASettings, GeneticAlgorithm
from ga.problem import load_objective
from objective_functions import rosenbrock

settings = GASettings.from_yaml("config_rosenbrock_example.yaml")
objective = load_objective(settings.objective_file, settings.objective_function)
result = GeneticAlgorithm(settings, objective).run(plot=False)
```

## Under the hood

The GA flow per generation:

1. evaluate every individual (objective + penalties)
2. `elitism` best individuals are carried over unchanged
3. `selection` picks parents (tournament by default)
4. `crossover_rate` fraction of parent pairs cross over
5. mutation applied to every offspring
6. next population = elites + offspring; repeat until `max_generations`,
   `no_improve_limit` stalling, or total convergence (all fitnesses equal)

Fitness is always handled as **lower-is-better** internally; `sense: maximize`
negates the objective before the loop, so maximizing problems work unchanged.