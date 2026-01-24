"""
VisuaLex API
Italian legal text scrapers and utilities

Usage:
    from visualex.scrapers import NormattivaScraper
    from visualex.models import Norma, NormaVisitata
    from visualex.utils import generate_urn
"""

__version__ = "0.1.0"

from . import scrapers
from . import models
from . import utils

__all__ = [
    "scrapers",
    "models",
    "utils",
    "__version__",
]
