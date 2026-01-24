"""
VisuaLex Scrapers
Scrapers for Italian and European legal databases
"""

from .normattiva import NormattivaScraper
from .brocardi import BrocardiScraper
from .eurlex import EurlexScraper

__all__ = [
    "NormattivaScraper",
    "BrocardiScraper",
    "EurlexScraper",
]
