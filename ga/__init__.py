"""ga - a config-driven genetic algorithm framework.

Configure the search via YAML (see config_rosenbrock_example.yaml); plug in
your own objective function as a plain Python function. See README.md for the
full guide, or run examples/search_sphere.py for a quick demo.

Public API:
    from ga import GASettings, GeneticAlgorithm
    from ga.common import run_from_config
"""

from .config import GASettings
from .algorithm import GeneticAlgorithm
from .common import run_from_config

__version__ = "1.0.0"
__all__ = ["GASettings", "GeneticAlgorithm", "run_from_config", "__version__"]