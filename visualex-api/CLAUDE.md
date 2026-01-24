# CLAUDE.md - VisuaLex API

> **Istruzioni per agenti AI che lavorano su questo repository**

---

## Contesto Progetto

Questa è la **libreria Python open source** per l'accesso alle fonti del diritto italiano. Fornisce scraper per Normattiva, Brocardi e EUR-Lex, oltre a utilities per la gestione di testi normativi.

**Parte di**: Monorepo ALIS_CORE
**Tipo**: Python library (pubblicata su PyPI)
**Licenza**: MIT (Open Source)
**PyPI**: `pip install visualex`

---

## Stack Tecnologico

- **Python 3.10+**
- **asyncio** per operazioni async
- **Playwright** per browser automation (Normattiva)
- **BeautifulSoup4** per parsing HTML
- **Pydantic** per data models
- **Quart** per API server (opzionale)

---

## Comandi Utili

```bash
# Installazione development
pip install -e ".[dev]"

# Test
pytest                         # Tutti i test
pytest tests/unit/             # Solo unit test
pytest -k "normattiva"         # Test specifici

# Linting
black visualex/                # Formattazione
ruff check visualex/           # Linting
mypy visualex/                 # Type checking

# API Server (dev)
./start_dev.sh                 # Avvia Quart server (port 5000)

# Build
python -m build                # Build wheel
twine upload dist/*            # Pubblica su PyPI
```

---

## Struttura Cartelle

```
visualex-api/
├── visualex/
│   ├── __init__.py            # Export pubblici
│   ├── scrapers/              # Scraper per fonte
│   │   ├── normattiva.py      # Normattiva.it (ufficiale)
│   │   ├── brocardi.py        # Brocardi.it (commenti)
│   │   └── eurlex.py          # EUR-Lex (UE)
│   │
│   ├── models/                # Data models (Pydantic)
│   │   ├── norma.py           # Norma, Articolo, Comma
│   │   ├── reference.py       # Riferimenti normativi
│   │   └── tree.py            # Struttura gerarchica
│   │
│   ├── utils/                 # Utilities
│   │   ├── text_op.py         # Operazioni testo
│   │   ├── treextractor.py    # Estrazione struttura
│   │   ├── urngenerator.py    # Generazione URN NIR
│   │   ├── cache_manager.py   # Cache
│   │   ├── http_client.py     # HTTP robusto
│   │   └── circuit_breaker.py # Resilienza
│   │
│   └── api/                   # API Server
│       └── app.py             # Quart endpoints
│
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
│
├── pyproject.toml             # Configurazione progetto
├── setup.py                   # Legacy setup
└── start_dev.sh               # Script avvio
```

---

## File Critici - Leggere Prima di Modificare

| File | Importanza | Note |
|------|------------|------|
| `scrapers/normattiva.py` | **CRITICO** | Scraper ufficiale, usa Playwright - fragile |
| `scrapers/brocardi.py` | **ALTA** | Parsing complesso, struttura HTML variabile |
| `models/norma.py` | **ALTA** | Modello dati centrale, usato ovunque |
| `utils/urngenerator.py` | **ALTA** | Generazione URN standard NIR |
| `utils/treextractor.py` | **ALTA** | Parsing struttura normativa |

---

## Pattern da Seguire

### Scraper Asincrono
```python
# scrapers/esempio.py
from visualex.utils.http_client import HttpClient
from visualex.models import Norma

class EsempioScraper:
    def __init__(self, client: HttpClient | None = None):
        self.client = client or HttpClient()

    async def fetch_by_urn(self, urn: str) -> Norma:
        """Fetch norma by URN."""
        url = self._build_url(urn)
        html = await self.client.get(url)
        return self._parse(html)

    def _build_url(self, urn: str) -> str:
        # URL construction logic
        pass

    def _parse(self, html: str) -> Norma:
        # HTML parsing logic
        pass
```

### Data Model
```python
# models/esempio.py
from pydantic import BaseModel
from datetime import date
from typing import List, Optional

class Articolo(BaseModel):
    numero: str
    rubrica: Optional[str] = None
    testo: str
    commi: List["Comma"] = []

    class Config:
        # Pydantic v2
        frozen = True  # Immutable
```

### Utility Function
```python
# utils/esempio.py
import re
from typing import Optional

def normalize_article_number(raw: str) -> Optional[str]:
    """
    Normalizza numero articolo.

    Args:
        raw: Stringa grezza (es. "Art. 1-bis")

    Returns:
        Numero normalizzato (es. "1-bis") o None
    """
    match = re.search(r'(\d+(?:-\w+)?)', raw)
    return match.group(1) if match else None
```

---

## Convenzioni Codice

### Python Style
- **Black** per formattazione (line length 88)
- **Ruff** per linting
- **Type hints** obbligatori
- **Docstrings** Google style

### Naming
- Classi: PascalCase (`NormattivaScraper`)
- Funzioni/variabili: snake_case (`fetch_by_urn`)
- Costanti: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`)
- File: snake_case (`normattiva.py`)

### Async
- Tutti i metodi I/O devono essere `async`
- Usa `asyncio.gather()` per parallelismo
- Non bloccare event loop con operazioni sync

### Error Handling
```python
from visualex.exceptions import ScraperError, ParsingError

try:
    norma = await scraper.fetch_by_urn(urn)
except ScraperError as e:
    logger.error(f"Fetch failed: {e}")
    raise
except ParsingError as e:
    logger.warning(f"Parse warning: {e}")
    # Attempt recovery
```

---

## Anti-Pattern - Cosa NON Fare

❌ **Non** usare `requests` - usa `HttpClient` (async)
   - Il progetto è async-first

❌ **Non** modificare URN format senza verificare standard NIR
   - URN devono essere conformi a NormeInRete

❌ **Non** hardcodare selettori CSS senza commento
   - I siti cambiano - documenta perché quel selettore

❌ **Non** ignorare eccezioni durante parsing
   - Logga sempre, permetti recovery parziale

❌ **Non** aggiungere dipendenze senza approvazione
   - La libreria deve rimanere leggera

---

## Gestione Siti Web Esterni

### Normattiva.it
- **Problema**: Richiede JavaScript, usa Playwright
- **Rate limiting**: Max 10 req/min
- **Caching**: Obbligatorio per articoli

### Brocardi.it
- **Problema**: HTML non strutturato, cambia spesso
- **Selettori**: Documentati in `SELECTORS.md`
- **Fallback**: Se parse fallisce, ritorna raw HTML

### EUR-Lex
- **API ufficiale** disponibile (CELLAR)
- **Preferire** API a scraping dove possibile

---

## Testing

### Unit Tests
```python
# tests/unit/test_urngenerator.py
import pytest
from visualex.utils.urngenerator import generate_urn

def test_generate_urn_legge():
    urn = generate_urn(
        authority="stato",
        act_type="legge",
        date="2020-12-30",
        number="178"
    )
    assert urn == "urn:nir:stato:legge:2020-12-30;178"
```

### Integration Tests
```python
# tests/integration/test_normattiva.py
import pytest
from visualex.scrapers import NormattivaScraper

@pytest.mark.integration
async def test_fetch_codice_civile():
    scraper = NormattivaScraper()
    norma = await scraper.fetch_by_urn(
        "urn:nir:stato:regio.decreto:1942-03-16;262"
    )
    assert norma.tipo_atto == "regio.decreto"
    assert len(norma.articoli) > 2000  # Codice civile ha ~2900 articoli
```

### Mock per test
- Usa `pytest-asyncio` per test async
- Mock `HttpClient` per unit test
- Fixtures in `tests/fixtures/` per HTML campione

---

## API Server

### Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/fetch` | Fetch norma by params |
| GET | `/api/v1/urn/{urn}` | Fetch by URN |
| GET | `/api/v1/search` | Search (se implementato) |
| GET | `/health` | Health check |

### Avvio
```bash
./start_dev.sh           # Dev mode (port 5000)
hypercorn api.app:app    # Production
```

---

## Dipendenze del Progetto

Questa libreria è usata da:
- **visualex-platform**: Backend Python API
- **merlt**: Per ingestion dati nel Knowledge Graph
- **visualex-merlt**: Indirettamente via merlt

---

## Workflow di Sviluppo

1. **Branch** da main: `feature/nome-feature`
2. **Sviluppa** con test
3. **Test** con `pytest`
4. **Lint** con `black` e `ruff`
5. **Type check** con `mypy`
6. **PR** con descrizione dettagliata
7. Se release: bump version in `pyproject.toml`

---

## Agenti Consigliati per Task

| Task | Agente |
|------|--------|
| Nuovo scraper | `scraper-builder` |
| Fix parsing | `builder` + test |
| Nuovo modello dati | `architect` poi `builder` |
| Ottimizzazione | `builder` |
| Bug investigation | `debugger` |
| Documentazione | `scribe` |

---

## Riferimenti

- [README Principale](../README.md)
- [Architettura](../ARCHITETTURA.md)
- [Glossario](../GLOSSARIO.md)
- [Guida Navigazione](../GUIDA_NAVIGAZIONE.md)
- [Standard URN NIR](https://www.normattiva.it/static/urn.html)

---

*Ultimo aggiornamento: Gennaio 2026*
