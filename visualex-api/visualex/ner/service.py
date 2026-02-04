"""
NER Service for Legal Text Processing.

Extracts legal entities from Italian legal text with URN resolution.
Supports both spaCy-based extraction (when available) and rule-based fallback.

Performance target: <500ms for typical query.
"""

import re
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import structlog

from .entities import (
    EntityType,
    ExtractedEntity,
    ExtractionResult,
    LABEL_TO_ENTITY_TYPE,
)

log = structlog.get_logger()

# ============================================================================
# Pattern definitions for rule-based extraction
# ============================================================================

# Article patterns: art. 1453, articolo 52-bis, artt. 1 e 2
ARTICLE_PATTERNS = [
    # Single article: art. 1453, Art. 52-bis
    re.compile(
        r'\b(?:art(?:icolo)?\.?)\s*(\d+(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)',
        re.IGNORECASE
    ),
    # Multiple articles: artt. 1 e 2, articoli 1, 2 e 3
    re.compile(
        r'\b(?:artt\.?|articoli)\s*([\d,\s]+(?:e\s*\d+)?)',
        re.IGNORECASE
    ),
    # Comma reference: comma 1, co. 3
    re.compile(
        r'\b(?:comma|co\.?)\s*(\d+(?:-(?:bis|ter|quater))?)',
        re.IGNORECASE
    ),
    # Lettera reference: lettera a), lett. b
    re.compile(
        r'\b(?:lettera|lett\.?)\s*([a-z])\)?',
        re.IGNORECASE
    ),
]

# Code patterns: codice civile, c.c., c.p., codice penale
CODE_PATTERNS = [
    # Full code names
    re.compile(
        r'\b(?:codice\s+(?:civile|penale|procedura\s+civile|procedura\s+penale|navigazione|strada|ambiente|consumo|crisi|privacy|appalti))',
        re.IGNORECASE
    ),
    # Abbreviated codes: c.c., c.p., c.p.c., c.p.p.
    re.compile(
        r'\b(c\.c\.|c\.p\.|c\.p\.c\.|c\.p\.p\.|c\.d\.s\.)',
        re.IGNORECASE
    ),
]

# Norm/Law patterns: legge 241/1990, D.Lgs. 50/2016, DPR 380/2001
# NOTE: Order matters - D.Lgs must come before D.L. to avoid partial matches
NORM_PATTERNS = [
    # Decreto legislativo: D.Lgs. 50/2016, decreto legislativo n. 50/2016
    # Must be before D.L. to avoid matching "D.L" part of "D.Lgs"
    re.compile(
        r'\b(?:d\.?\s*lgs\.?|decreto\s+legislativo)\s*(?:n\.?\s*)?(\d+)(?:/|del\s*)(\d{2,4})',
        re.IGNORECASE
    ),
    # Law with number/year: legge 241/1990, l. 104/92
    re.compile(
        r'\b(?:legge|l\.)\s*(?:n\.?\s*)?(\d+)(?:/|del\s*)(\d{2,4})',
        re.IGNORECASE
    ),
    # DPR: DPR 380/2001
    re.compile(
        r'\b(?:d\.?p\.?r\.?|decreto\s+(?:del\s+)?presidente\s+(?:della\s+)?repubblica)\s*(?:n\.?\s*)?(\d+)(?:/|del\s*)(\d{2,4})',
        re.IGNORECASE
    ),
    # Decreto legge: D.L. 18/2020 (after D.Lgs to avoid partial match)
    re.compile(
        r'\b(?:d\.?\s*l\.?|decreto\s+legge)\s*(?:n\.?\s*)?(\d+)(?:/|del\s*)(\d{2,4})',
        re.IGNORECASE
    ),
    # Costituzione
    re.compile(r'\b(costituzione)\b', re.IGNORECASE),
]

# Temporal patterns: del 2019, anno 2020, dal 1990 al 2000
TEMPORAL_PATTERNS = [
    # Year reference: del 2019, anno 2020
    re.compile(r'\b(?:del|anno|nel)\s*((?:19|20)\d{2})\b', re.IGNORECASE),
    # Date range: dal 1990 al 2000
    re.compile(
        r'\b(?:dal|a\s+partire\s+dal)\s*((?:19|20)\d{2})\s*(?:al|fino\s+al)\s*((?:19|20)\d{2})\b',
        re.IGNORECASE
    ),
    # Full date: 16 marzo 1942, 30/12/2020
    re.compile(
        r'\b(\d{1,2})\s*(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s*((?:19|20)\d{2})\b',
        re.IGNORECASE
    ),
    re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2})\b'),
]

# Party patterns: compratore, venditore, locatore, conduttore
PARTY_PATTERNS = [
    # Common contractual parties
    re.compile(
        r'\b(compratore|venditore|acquirente|alienante|locatore|conduttore|comodante|comodatario|mutuante|mutuatario|mandante|mandatario|appaltatore|committente|prestatore|utilizzatore|cedente|cessionario|debitore|creditore|garante|fideiussore|assicurato|assicuratore)\b',
        re.IGNORECASE
    ),
    # Parties in litigation
    re.compile(
        r'\b(attore|convenuto|ricorrente|resistente|appellante|appellato|imputato|querelante|parte\s+civile)\b',
        re.IGNORECASE
    ),
]

# Legal concept patterns (common Italian legal terms)
LEGAL_CONCEPT_KEYWORDS = {
    # Contract law
    "inadempimento", "risoluzione", "rescissione", "recesso", "nullità",
    "annullabilità", "simulazione", "rappresentanza", "procura", "mandato",
    "caparra", "penale", "clausola", "condizione", "termine", "mora",
    "impossibilità", "sopravvenuta", "prestazione", "obbligazione",
    # Property law
    "proprietà", "possesso", "detenzione", "usucapione", "servitù",
    "usufrutto", "uso", "abitazione", "superficie", "enfiteusi",
    # Tort law
    "danno", "risarcimento", "responsabilità", "colpa", "dolo",
    "negligenza", "imprudenza", "imperizia", "nesso causale",
    # Procedure
    "decadenza", "prescrizione", "interruzione", "sospensione",
    "giurisdizione", "competenza", "legittimazione",
}

# Pre-compiled pattern for legal concepts (O(1) matching vs O(n) keyword iteration)
# Sorted by length descending to match longer terms first (e.g., "nesso causale" before "nesso")
_SORTED_CONCEPTS = sorted(LEGAL_CONCEPT_KEYWORDS, key=len, reverse=True)
LEGAL_CONCEPT_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in _SORTED_CONCEPTS) + r')\b',
    re.IGNORECASE
)


@dataclass
class NERConfig:
    """Configuration for NER service."""

    use_spacy: bool = True  # Try to use spaCy if available
    spacy_model: str = "it_core_news_lg"  # Fallback spaCy model
    custom_model_path: Optional[str] = None  # Path to custom NER model
    confidence_threshold: float = 0.5  # Min confidence for inclusion
    ambiguity_threshold: float = 0.7  # Below this, mark as ambiguous
    timeout_ms: float = 500.0  # Timeout for extraction
    resolve_urns: bool = True  # Whether to resolve article refs to URNs


class NERService:
    """
    NER service for extracting legal entities from Italian text.

    Supports:
    - Article references with URN resolution
    - Legal concepts
    - Temporal context
    - Party references
    - Norm/law references
    - Code references

    Uses spaCy when available, falls back to rule-based extraction.
    """

    def __init__(self, config: Optional[NERConfig] = None):
        """
        Initialize NER service.

        Args:
            config: NER configuration options
        """
        self.config = config or NERConfig()
        self._nlp = None
        self._spacy_available = False
        self._init_spacy()

    def _init_spacy(self) -> None:
        """Initialize spaCy model if available."""
        if not self.config.use_spacy:
            log.info("ner_spacy_disabled")
            return

        try:
            import spacy

            # Try custom model first
            if self.config.custom_model_path:
                try:
                    self._nlp = spacy.load(self.config.custom_model_path)
                    self._spacy_available = True
                    log.info("ner_custom_model_loaded", path=self.config.custom_model_path)
                    return
                except Exception as e:
                    log.warning("ner_custom_model_failed", error=str(e))

            # Fall back to base model
            try:
                self._nlp = spacy.load(self.config.spacy_model)
                self._spacy_available = True
                log.info("ner_spacy_loaded", model=self.config.spacy_model)
            except OSError:
                log.warning("ner_spacy_model_not_found", model=self.config.spacy_model)
                self._spacy_available = False

        except ImportError:
            log.info("ner_spacy_not_installed")
            self._spacy_available = False

    async def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """
        Extract legal entities from text.

        Args:
            text: Input text to process
            context: Optional context (e.g., source norm for resolving refs)

        Returns:
            ExtractionResult with extracted entities
        """
        start_time = time.perf_counter()
        result = ExtractionResult(text=text)

        if not text or not text.strip():
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        try:
            # Apply timeout
            timeout_sec = self.config.timeout_ms / 1000.0

            try:
                entities = await asyncio.wait_for(
                    self._extract_entities(text, context),
                    timeout=timeout_sec
                )
                result.entities = entities
            except asyncio.TimeoutError:
                # Continue with partial results from rule-based extraction
                result.warnings.append(
                    f"Extraction timed out after {self.config.timeout_ms}ms, using partial results"
                )
                result.entities = self._extract_rule_based(text, context)
                log.warning("ner_extraction_timeout", text_length=len(text))

        except Exception as e:
            result.has_errors = True
            result.error_message = str(e)
            result.warnings.append("Extraction failed, attempting rule-based fallback")
            log.error("ner_extraction_error", error=str(e))

            # Try rule-based fallback
            try:
                result.entities = self._extract_rule_based(text, context)
                result.has_errors = False  # Recovery successful
            except Exception as fallback_error:
                log.error("ner_fallback_error", error=str(fallback_error))

        # Resolve URNs if enabled
        if self.config.resolve_urns:
            result.entities = self._resolve_urns(result.entities, context)

        # Mark ambiguous entities
        result.entities = self._mark_ambiguous(result.entities)

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000

        log.debug(
            "ner_extraction_complete",
            entity_count=len(result.entities),
            time_ms=result.processing_time_ms,
        )

        return result

    async def _extract_entities(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
    ) -> List[ExtractedEntity]:
        """
        Extract entities using spaCy if available, otherwise rule-based.
        """
        entities: List[ExtractedEntity] = []

        # Rule-based extraction (always run for legal-specific patterns)
        rule_entities = self._extract_rule_based(text, context)
        entities.extend(rule_entities)

        # spaCy extraction for additional entities
        if self._spacy_available and self._nlp:
            spacy_entities = self._extract_spacy(text)
            # Merge without duplicates
            entities = self._merge_entities(entities, spacy_entities)

        return entities

    def _extract_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        if not self._nlp:
            return []

        entities: List[ExtractedEntity] = []
        doc = self._nlp(text)

        for ent in doc.ents:
            # Map spaCy label to our EntityType
            entity_type = LABEL_TO_ENTITY_TYPE.get(ent.label_, EntityType.UNKNOWN)

            # Skip unknown entities for cleaner output
            if entity_type == EntityType.UNKNOWN:
                continue

            entities.append(
                ExtractedEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,  # Default spaCy confidence
                    metadata={"source": "spacy", "label": ent.label_},
                )
            )

        return entities

    def _extract_rule_based(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
    ) -> List[ExtractedEntity]:
        """Extract entities using rule-based patterns."""
        entities: List[ExtractedEntity] = []

        # Article references
        for pattern in ARTICLE_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.ARTICLE_REF,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        metadata={
                            "source": "rule",
                            "pattern": "article",
                            "extracted_value": match.group(1) if match.lastindex and match.lastindex >= 1 else None,
                        },
                    )
                )

        # Code references
        for pattern in CODE_PATTERNS:
            for match in pattern.finditer(text):
                code_name = self._normalize_code_name(match.group(0))
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.CODE_REF,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                        metadata={
                            "source": "rule",
                            "pattern": "code",
                            "normalized_name": code_name,
                        },
                    )
                )

        # Norm/Law references
        for pattern in NORM_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.NORM_REF,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        metadata={
                            "source": "rule",
                            "pattern": "norm",
                            "groups": list(match.groups()),  # Convert tuple to list for JSON
                        },
                    )
                )

        # Temporal references
        for pattern in TEMPORAL_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.TEMPORAL,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85,
                        metadata={
                            "source": "rule",
                            "pattern": "temporal",
                            "groups": list(match.groups()),  # Convert tuple to list for JSON
                        },
                    )
                )

        # Party references
        for pattern in PARTY_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.PARTY,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        metadata={"source": "rule", "pattern": "party"},
                    )
                )

        # Legal concepts (optimized regex matching)
        for match in LEGAL_CONCEPT_PATTERN.finditer(text):
            entities.append(
                ExtractedEntity(
                    text=match.group(0),
                    entity_type=EntityType.LEGAL_CONCEPT,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8,
                    metadata={"source": "rule", "pattern": "concept"},
                )
            )

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return entities

    def _normalize_code_name(self, text: str) -> str:
        """Normalize code abbreviation to full name."""
        text_lower = text.lower()
        mappings = {
            "c.c.": "codice civile",
            "c.p.": "codice penale",
            "c.p.c.": "codice procedura civile",
            "c.p.p.": "codice procedura penale",
            "c.d.s.": "codice della strada",
        }
        for abbrev, full in mappings.items():
            if abbrev in text_lower:
                return full
        return text_lower

    def _merge_entities(
        self,
        primary: List[ExtractedEntity],
        secondary: List[ExtractedEntity],
    ) -> List[ExtractedEntity]:
        """Merge entity lists, avoiding duplicates based on span overlap."""
        result = list(primary)

        for sec_entity in secondary:
            # Check for overlap with existing entities
            has_overlap = False
            for pri_entity in primary:
                if self._spans_overlap(
                    (pri_entity.start, pri_entity.end),
                    (sec_entity.start, sec_entity.end),
                ):
                    has_overlap = True
                    break

            if not has_overlap:
                result.append(sec_entity)

        result.sort(key=lambda e: e.start)
        return result

    def _spans_overlap(
        self, span1: Tuple[int, int], span2: Tuple[int, int]
    ) -> bool:
        """Check if two spans overlap."""
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def _resolve_urns(
        self,
        entities: List[ExtractedEntity],
        context: Optional[Dict[str, Any]],
    ) -> List[ExtractedEntity]:
        """Resolve article references to URNs."""
        from ..utils.urn_pipeline import dict_to_urn

        for entity in entities:
            if entity.entity_type == EntityType.ARTICLE_REF:
                # Try to resolve using context
                urn = self._resolve_article_urn(entity, context)
                if urn:
                    entity.resolved_urn = urn

            elif entity.entity_type == EntityType.CODE_REF:
                # Resolve code to URN
                code_name = entity.metadata.get("normalized_name", entity.text)
                urn = self._code_to_urn(code_name)
                if urn:
                    entity.resolved_urn = urn

        return entities

    def _resolve_article_urn(
        self,
        entity: ExtractedEntity,
        context: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Resolve article reference to URN."""
        from ..utils.urn_pipeline import dict_to_urn

        # Extract article number from text
        match = re.search(
            r'(\d+)(?:-?(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?',
            entity.text,
            re.IGNORECASE
        )
        if not match:
            return None

        articolo = match.group(1)
        estensione = match.group(2).lower() if match.group(2) else None

        # Try to get act type from context or nearby code reference
        tipo_atto = None
        data = None
        numero = None

        if context:
            tipo_atto = context.get("tipo_atto")
            data = context.get("data")
            numero = context.get("numero")

        # If no context, check entity metadata for nearby code
        if not tipo_atto and "nearby_code" in entity.metadata:
            tipo_atto = entity.metadata["nearby_code"]

        if not tipo_atto:
            # Mark as ambiguous - we don't know which act
            entity.is_ambiguous = True
            entity.metadata["ambiguity_reason"] = "unknown_act_type"
            return None

        # Build URN
        urn_data = {
            "tipo_atto": tipo_atto,
            "articolo": articolo,
        }
        if estensione:
            urn_data["estensione"] = estensione
        if data:
            urn_data["data"] = data
        if numero:
            urn_data["numero"] = numero

        try:
            return dict_to_urn(urn_data)
        except Exception as e:
            log.warning("urn_resolution_failed", error=str(e), urn_data=urn_data)
            return None

    def _code_to_urn(self, code_name: str) -> Optional[str]:
        """Convert code name to URN."""
        from ..utils.urn_pipeline import URN_PREFIX

        code_urns = {
            "codice civile": f"{URN_PREFIX}regio.decreto:1942-03-16;262",
            "codice penale": f"{URN_PREFIX}regio.decreto:1930-10-19;1398",
            "codice procedura civile": f"{URN_PREFIX}regio.decreto:1940-10-28;1443",
            "codice procedura penale": f"{URN_PREFIX}decreto.del.presidente.della.repubblica:1988-09-22;447",
            "costituzione": f"{URN_PREFIX}costituzione",
        }
        return code_urns.get(code_name.lower())

    def _mark_ambiguous(
        self, entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Mark entities below confidence threshold as ambiguous."""
        for entity in entities:
            if (
                entity.confidence < self.config.ambiguity_threshold
                and not entity.is_ambiguous
            ):
                entity.is_ambiguous = True
                entity.metadata["ambiguity_reason"] = "low_confidence"
        return entities

    def is_ready(self) -> bool:
        """Check if service is ready for extraction."""
        return True  # Rule-based always available

    @property
    def has_spacy(self) -> bool:
        """Check if spaCy is available."""
        return self._spacy_available
