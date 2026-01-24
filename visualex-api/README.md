# VisuaLex API

> Italian legal text scrapers and utilities

[![PyPI version](https://badge.fury.io/py/visualex.svg)](https://pypi.org/project/visualex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install visualex
```

For browser-based scraping (Normattiva):
```bash
pip install visualex[browser]
playwright install chromium
```

## Features

- **Normattiva Scraper**: Official Italian legislation database (normattiva.it)
- **Brocardi Scraper**: Italian legal encyclopedia and annotations (brocardi.it)
- **EurLex Scraper**: European Union law database (eur-lex.europa.eu)
- **URN Generator**: Generate Italian law URN identifiers (NIR standard)
- **Text Utilities**: Text processing for legal documents

## Quick Start

```python
from visualex.scrapers import NormattivaScraper, BrocardiScraper
from visualex.models import Norma

# Fetch a law from Normattiva
async def main():
    scraper = NormattivaScraper()

    # Fetch the Italian Privacy Code
    norma = await scraper.fetch_by_urn(
        "urn:nir:stato:decreto.legislativo:2003-06-30;196"
    )

    print(f"Title: {norma.titolo}")
    print(f"Articles: {len(norma.articoli)}")

# Or use Brocardi for annotations
async def get_annotations():
    scraper = BrocardiScraper()
    annotations = await scraper.fetch_article_annotations(
        codice="codice-civile",
        article="1"
    )
    print(annotations)
```

## Supported Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| Normattiva | Official Italian legislation | All Italian laws since 1861 |
| Brocardi | Legal encyclopedia | Civil, Penal, Procedural codes |
| EUR-Lex | EU legislation | All EU regulations and directives |

## API Reference

### Scrapers

```python
from visualex.scrapers import NormattivaScraper

scraper = NormattivaScraper()

# By URN
norma = await scraper.fetch_by_urn("urn:nir:stato:legge:2020-12-30;178")

# By parameters
norma = await scraper.fetch(
    act_type="legge",
    date="2020-12-30",
    number="178"
)
```

### Models

```python
from visualex.models import Norma, NormaVisitata

# Norma contains the document structure
norma = Norma(
    tipo_atto="legge",
    data="2020-12-30",
    numero="178",
    titolo="Bilancio di previsione...",
    articoli=[...]
)

# NormaVisitata tracks visit history
visitata = NormaVisitata(norma=norma, visited_at=datetime.now())
```

### Utilities

```python
from visualex.utils import generate_urn, normalize_text

# Generate URN
urn = generate_urn(
    authority="stato",
    act_type="decreto.legislativo",
    date="2003-06-30",
    number="196"
)
# -> "urn:nir:stato:decreto.legislativo:2003-06-30;196"

# Normalize legal text
clean = normalize_text("Art.   1  comma  2")
# -> "Art. 1 comma 2"
```

## Development

```bash
git clone https://github.com/visualex/visualex-api
cd visualex-api
pip install -e ".[dev]"
pytest
```

### Local start script

```bash
./start_dev.sh
```

This script runs the Quart API server on `http://localhost:5000`.

## Contributing

Contributions are welcome! Please read our contributing guidelines.

## License

MIT License - see [LICENSE](LICENSE)
