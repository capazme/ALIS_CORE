"""
NER Module for Legal Entity Extraction.

Provides Named Entity Recognition for Italian legal text:
- Article references (art. 1453 c.c.)
- Legal concepts (inadempimento, risoluzione)
- Temporal context (contratto del 2019)
- Party references (compratore, venditore)
- Norm/law references (legge 241/1990)
- Code references (codice civile)

Example:
    >>> from visualex.ner import NERService, NERConfig
    >>> service = NERService()
    >>> result = await service.extract("L'art. 1453 c.c. regola la risoluzione")
    >>> for entity in result.entities:
    ...     print(f"{entity.text}: {entity.entity_type.value}")
    art. 1453: ARTICLE_REF
    c.c.: CODE_REF
    risoluzione: LEGAL_CONCEPT
"""

from .entities import (
    EntityType,
    ExtractedEntity,
    ExtractionResult,
    LABEL_TO_ENTITY_TYPE,
)
from .service import (
    NERService,
    NERConfig,
    ARTICLE_PATTERNS,
    CODE_PATTERNS,
    NORM_PATTERNS,
    TEMPORAL_PATTERNS,
    PARTY_PATTERNS,
    LEGAL_CONCEPT_KEYWORDS,
    LEGAL_CONCEPT_PATTERN,
)

__all__ = [
    # Entity types
    "EntityType",
    "ExtractedEntity",
    "ExtractionResult",
    "LABEL_TO_ENTITY_TYPE",
    # Service
    "NERService",
    "NERConfig",
    # Patterns (for extension/customization)
    "ARTICLE_PATTERNS",
    "CODE_PATTERNS",
    "NORM_PATTERNS",
    "TEMPORAL_PATTERNS",
    "PARTY_PATTERNS",
    "LEGAL_CONCEPT_KEYWORDS",
    "LEGAL_CONCEPT_PATTERN",
]
