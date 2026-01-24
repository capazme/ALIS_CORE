"""
MERL-T Sources
==============

Scrapers per fonti giuridiche italiane.

Scrapers disponibili:
- NormattivaScraper: Testi ufficiali da Normattiva.it
- BrocardiScraper: Enrichment (massime, spiegazioni, ratio)

Eccezioni:
- ScraperError: Errore base
- NetworkError: Errori di rete
- DocumentNotFoundError: Documento non trovato
- ParsingError: Errore parsing HTML

Esempio:
    from merlt.sources import NormattivaScraper, BrocardiScraper

    scraper = NormattivaScraper()
    text, url = await scraper.fetch_document(norma_visitata)
"""

from visualex.scrapers.normattiva import NormattivaScraper
from visualex.scrapers.brocardi import BrocardiScraper
from visualex.scrapers.base import (
    BaseScraper,
    ScraperConfig,
    ScraperError,
    NetworkError,
    DocumentNotFoundError,
    ParsingError,
)

__all__ = [
    # Scrapers
    "NormattivaScraper",
    "BrocardiScraper",
    "BaseScraper",
    # Config
    "ScraperConfig",
    # Eccezioni
    "ScraperError",
    "NetworkError",
    "DocumentNotFoundError",
    "ParsingError",
]
