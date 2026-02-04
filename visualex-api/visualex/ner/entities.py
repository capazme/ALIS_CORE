"""
NER Entity Dataclasses for Legal Text Processing.

Entity types for Italian legal domain:
- ARTICLE_REF: Article references (art. 1453 c.c.)
- LEGAL_CONCEPT: Legal concepts (inadempimento, risoluzione)
- TEMPORAL: Temporal context (contratto del 2019)
- PARTY: Party references (compratore, venditore)
- NORM_REF: Full norm references (legge 241/1990)
- CODE_REF: Code references (codice civile)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class EntityType(str, Enum):
    """Legal entity types for NER extraction."""

    ARTICLE_REF = "ARTICLE_REF"  # Article references
    LEGAL_CONCEPT = "LEGAL_CONCEPT"  # Legal concepts/terms
    TEMPORAL = "TEMPORAL"  # Temporal context
    PARTY = "PARTY"  # Party references
    NORM_REF = "NORM_REF"  # Full norm/law references
    CODE_REF = "CODE_REF"  # Code references
    UNKNOWN = "UNKNOWN"  # Unclassified entity


# Mapping from spaCy/custom NER labels to our EntityType
LABEL_TO_ENTITY_TYPE = {
    "ARTICOLO": EntityType.ARTICLE_REF,
    "LEGGE": EntityType.NORM_REF,
    "CODICE": EntityType.CODE_REF,
    "COMMA": EntityType.ARTICLE_REF,  # Comma is part of article ref
    "LETTERA": EntityType.ARTICLE_REF,  # Lettera is part of article ref
    "RIFERIMENTO": EntityType.NORM_REF,
    "DATE": EntityType.TEMPORAL,
    "ORG": EntityType.PARTY,  # Organizations as parties
    "PER": EntityType.PARTY,  # Persons as parties
    "LOC": EntityType.TEMPORAL,  # Locations often imply jurisdiction
}


@dataclass
class ExtractedEntity:
    """
    An entity extracted from legal text.

    Attributes:
        text: Original text span
        entity_type: Type of entity (from EntityType enum)
        start: Character offset start position
        end: Character offset end position
        confidence: Confidence score (0.0-1.0)
        resolved_urn: Resolved URN if applicable
        is_ambiguous: Flag for potential F1 feedback
        metadata: Additional entity-specific data
    """

    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    resolved_urn: Optional[str] = None
    is_ambiguous: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
            "resolved_urn": self.resolved_urn,
            "is_ambiguous": self.is_ambiguous,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        urn_str = f", urn={self.resolved_urn}" if self.resolved_urn else ""
        amb_str = " [AMBIGUOUS]" if self.is_ambiguous else ""
        return (
            f"ExtractedEntity('{self.text}', {self.entity_type.value}, "
            f"[{self.start}:{self.end}], conf={self.confidence:.2f}{urn_str}{amb_str})"
        )


@dataclass
class ExtractionResult:
    """
    Result of NER extraction on a text.

    Attributes:
        text: Original input text
        entities: List of extracted entities
        processing_time_ms: Time taken for extraction
        has_errors: Whether errors occurred during extraction
        error_message: Error message if any
        warnings: List of warnings (e.g., partial results)
    """

    text: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    processing_time_ms: float = 0.0
    has_errors: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "entity_count": len(self.entities),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "has_errors": self.has_errors,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "ambiguous_count": sum(1 for e in self.entities if e.is_ambiguous),
        }

    @property
    def article_refs(self) -> List[ExtractedEntity]:
        """Get article reference entities."""
        return [e for e in self.entities if e.entity_type == EntityType.ARTICLE_REF]

    @property
    def legal_concepts(self) -> List[ExtractedEntity]:
        """Get legal concept entities."""
        return [e for e in self.entities if e.entity_type == EntityType.LEGAL_CONCEPT]

    @property
    def temporal_refs(self) -> List[ExtractedEntity]:
        """Get temporal reference entities."""
        return [e for e in self.entities if e.entity_type == EntityType.TEMPORAL]

    @property
    def party_refs(self) -> List[ExtractedEntity]:
        """Get party reference entities."""
        return [e for e in self.entities if e.entity_type == EntityType.PARTY]

    @property
    def norm_refs(self) -> List[ExtractedEntity]:
        """Get full norm reference entities."""
        return [e for e in self.entities if e.entity_type == EntityType.NORM_REF]

    @property
    def ambiguous_entities(self) -> List[ExtractedEntity]:
        """Get entities flagged as ambiguous (for F1 feedback)."""
        return [e for e in self.entities if e.is_ambiguous]
