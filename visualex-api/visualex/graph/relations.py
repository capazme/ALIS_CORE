"""
Relation Extraction & Creation Module
======================================

Extract and create relationships between legal norms in the Knowledge Graph.

This module:
- Parses Italian legal text for citation patterns
- Detects modification relationships
- Extracts jurisprudence references
- Creates appropriate edges in FalkorDB

Reference: Story 2b-3: Relation Extraction & Creation
"""

import re
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Tuple
from enum import Enum

from visualex.graph.schema import EdgeType, NodeType
from visualex.utils.urn_pipeline import (
    parse_urn,
    URNComponents,
    URN_PREFIX,
)

if TYPE_CHECKING:
    from visualex.graph.client import FalkorDBClient

__all__ = [
    "CitationExtractor",
    "RelationCreator",
    "ExtractedCitation",
    "ExtractedModification",
    "ExtractionResult",
    "RelationType",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Relation Type Enum
# =============================================================================


class RelationType(str, Enum):
    """Types of relations that can be extracted from text."""

    # Citation types
    CITA = "cita"
    CITA_ARTICOLO = "cita_articolo"
    CITA_LEGGE = "cita_legge"
    CITA_GIURISPRUDENZA = "cita_giurisprudenza"

    # Modification types
    SOSTITUISCE = "sostituisce"
    ABROGA_TOTALMENTE = "abroga_totalmente"
    ABROGA_PARZIALMENTE = "abroga_parzialmente"
    INTEGRA = "integra"
    SOSPENDE = "sospende"
    PROROGA = "proroga"
    DEROGA_A = "deroga_a"

    # Interpretation
    INTERPRETA = "interpreta"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ExtractedCitation:
    """A citation extracted from text."""

    raw_text: str  # Original matched text
    relation_type: RelationType
    target_urn: Optional[str] = None  # Resolved URN if determinable
    target_article: Optional[str] = None
    target_comma: Optional[str] = None
    target_lettera: Optional[str] = None
    target_act_type: Optional[str] = None
    target_date: Optional[str] = None
    target_number: Optional[str] = None
    context: str = ""  # Surrounding text for context
    start_pos: int = 0  # Position in source text
    end_pos: int = 0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_text": self.raw_text,
            "relation_type": self.relation_type.value,
            "target_urn": self.target_urn,
            "target_article": self.target_article,
            "target_comma": self.target_comma,
            "target_lettera": self.target_lettera,
            "target_act_type": self.target_act_type,
            "target_date": self.target_date,
            "target_number": self.target_number,
            "confidence": self.confidence,
        }


@dataclass
class ExtractedModification:
    """A modification relationship extracted from text."""

    raw_text: str
    relation_type: RelationType
    target_urn: Optional[str] = None
    data_efficacia: Optional[str] = None  # When the modification takes effect
    testo_modificato: Optional[str] = None  # Original text being modified
    testo_nuovo: Optional[str] = None  # New replacement text
    context: str = ""
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_text": self.raw_text,
            "relation_type": self.relation_type.value,
            "target_urn": self.target_urn,
            "data_efficacia": self.data_efficacia,
            "testo_modificato": self.testo_modificato,
            "testo_nuovo": self.testo_nuovo,
            "confidence": self.confidence,
        }


@dataclass
class ExtractionResult:
    """Complete result of relation extraction from a text."""

    source_urn: str
    citations: List[ExtractedCitation] = field(default_factory=list)
    modifications: List[ExtractedModification] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def total_relations(self) -> int:
        """Total number of extracted relations."""
        return len(self.citations) + len(self.modifications)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_urn": self.source_urn,
            "citations": [c.to_dict() for c in self.citations],
            "modifications": [m.to_dict() for m in self.modifications],
            "total_relations": self.total_relations,
            "errors": self.errors,
        }


# =============================================================================
# Citation Extractor
# =============================================================================


class CitationExtractor:
    """
    Extract legal citations from Italian legal text.

    Supports patterns for:
    - Article citations: "ai sensi dell'art. X", "di cui all'articolo X"
    - Law citations: "L. n. 123/2020", "legge 30 dicembre 2020, n. 178"
    - Decree citations: "D.Lgs. n. 50/2016", "decreto legislativo"
    - Jurisprudence: "Cass. n. 12345/2021", "Corte Cost. sent. n. 123/2020"
    - Modification markers: "è sostituito da", "è abrogato"
    """

    # Citation patterns (compiled regex)
    # Order matters: more specific patterns first

    # Article within same act: "art. 123", "articolo 123-bis"
    ARTICLE_SIMPLE_PATTERN = re.compile(
        r"(?:^|[^a-z])(?:art(?:icol[oi])?\.?\s*)(\d+)(?:-([a-z]+))?\b",
        re.IGNORECASE
    )

    # Article with comma: "art. 123, comma 1" or "art. 123, c. 1"
    ARTICLE_COMMA_PATTERN = re.compile(
        r"(?:^|[^a-z])art(?:icol[oi])?\.?\s*(\d+)(?:-([a-z]+))?,?\s*(?:comma|c\.)\s*(\d+)\b",
        re.IGNORECASE
    )

    # Article with comma and lettera: "art. 123, comma 1, lett. a)"
    ARTICLE_FULL_PATTERN = re.compile(
        r"(?:^|[^a-z])art(?:icol[oi])?\.?\s*(\d+)(?:-([a-z]+))?,?\s*(?:comma|c\.)\s*(\d+),?\s*(?:lett(?:era)?\.?|let\.)\s*([a-z])\)",
        re.IGNORECASE
    )

    # Citation with context: "ai sensi dell'art. X" or "di cui all'articolo X"
    # Also handles "dall'art." and "all'art." patterns
    # Handles both straight and curly apostrophes
    CITATION_CONTEXT_PATTERN = re.compile(
        r"(?:"
        r"(?:ai sensi|di cui|secondo|conformemente a|in base a|a norma)(?:\s+del?l?['\u2019]?)?|"  # ai sensi dell'
        r"(?:ex)\s+|"  # ex art
        r"(?:dal?l|al?l|previsto dal?l)['\u2019]"  # dall', all'
        r")\s*"
        r"art(?:icol[oi])?\.?\s*(\d+)(?:-([a-z]+))?",
        re.IGNORECASE
    )

    # Law citation: "L. n. 123/2020" or "legge 30 dicembre 2020, n. 178"
    LAW_SHORT_PATTERN = re.compile(
        r'\bL\.?\s*(?:n\.?\s*)(\d+)/(\d{4})\b',
        re.IGNORECASE
    )

    LAW_FULL_PATTERN = re.compile(
        r'\blegge\s+(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|'
        r'luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4}),?\s*n\.?\s*(\d+)\b',
        re.IGNORECASE
    )

    # Decree citation: "D.Lgs. n. 50/2016", "D.L. n. 18/2020"
    DECREE_PATTERN = re.compile(
        r'\b(D\.?Lgs\.?|D\.?L\.?|D\.?P\.?R\.?|D\.?M\.?|D\.?P\.?C\.?M\.?)\s*(?:n\.?\s*)(\d+)/(\d{4})\b',
        re.IGNORECASE
    )

    # Full decree: "decreto legislativo 18 aprile 2016, n. 50"
    DECREE_FULL_PATTERN = re.compile(
        r'\bdecreto\s+(legislativo|legge|del Presidente della Repubblica|ministeriale)\s+'
        r'(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|'
        r'luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4}),?\s*n\.?\s*(\d+)\b',
        re.IGNORECASE
    )

    # Codice civile/penale reference: "art. 1453 c.c." or "art. 640 c.p."
    # Handle "L'art. X c.c." with apostrophe before art
    CODICE_PATTERN = re.compile(
        r"(?:^|[^a-z])art(?:icol[oi])?\.?\s*(\d+)(?:-([a-z]+))?\s+(c\.c\.|c\.p\.|c\.p\.c\.|c\.p\.p\.)",
        re.IGNORECASE
    )

    # Jurisprudence: "Cass. n. 12345/2021", "Cass. civ. sez. III, n. 123/2021"
    CASSAZIONE_PATTERN = re.compile(
        r'\bCass(?:azione)?\.?\s*(?:civ(?:ile)?\.?|pen(?:ale)?\.?)?\s*'
        r'(?:sez(?:ione)?\.?\s*[IVX]+,?\s*)?'
        r'(?:(?:sent(?:enza)?\.?|ord(?:inanza)?\.?)\s*)?'
        r'(?:n\.?\s*)(\d+)/(\d{4})\b',
        re.IGNORECASE
    )

    # Constitutional Court: "Corte Cost. sent. n. 123/2020"
    CORTE_COST_PATTERN = re.compile(
        r'\bCorte\s+Cost(?:ituzionale)?\.?\s*(?:sent(?:enza)?\.?\s*)?(?:n\.?\s*)(\d+)/(\d{4})\b',
        re.IGNORECASE
    )

    # Modification patterns
    MODIFICATION_PATTERNS = {
        RelationType.SOSTITUISCE: re.compile(
            r'(?:è|sono|viene|vengono)\s+(?:così\s+)?sostitu[it][ato](?:\s+da)?',
            re.IGNORECASE
        ),
        RelationType.ABROGA_TOTALMENTE: re.compile(
            r'(?:è|sono|viene|vengono)\s+abrogat[oaie]',
            re.IGNORECASE
        ),
        RelationType.ABROGA_PARZIALMENTE: re.compile(
            r'(?:è|sono|viene|vengono)\s+(?:parzialmente\s+)?abrogat[oaie]\s+(?:limitatamente|per la parte)',
            re.IGNORECASE
        ),
        RelationType.INTEGRA: re.compile(
            r'(?:è|sono|viene|vengono)\s+integrat[oaie]',
            re.IGNORECASE
        ),
        RelationType.SOSPENDE: re.compile(
            r'(?:è|sono|viene|vengono)\s+sospes[oaie]',
            re.IGNORECASE
        ),
        RelationType.PROROGA: re.compile(
            r'(?:è|sono|viene|vengono)\s+prorogat[oaie]',
            re.IGNORECASE
        ),
        RelationType.DEROGA_A: re.compile(
            r"in deroga\s+(?:a|all['\u2019])",
            re.IGNORECASE
        ),
    }

    # Month name to number mapping
    MONTHS = {
        'gennaio': '01', 'febbraio': '02', 'marzo': '03', 'aprile': '04',
        'maggio': '05', 'giugno': '06', 'luglio': '07', 'agosto': '08',
        'settembre': '09', 'ottobre': '10', 'novembre': '11', 'dicembre': '12',
    }

    # Decree type mapping
    DECREE_TYPES = {
        'd.lgs.': 'decreto.legislativo',
        'd.lgs': 'decreto.legislativo',
        'dlgs': 'decreto.legislativo',
        'd.l.': 'decreto.legge',
        'd.l': 'decreto.legge',
        'dl': 'decreto.legge',
        'd.p.r.': 'decreto.del.presidente.della.repubblica',
        'dpr': 'decreto.del.presidente.della.repubblica',
        'd.m.': 'decreto.ministeriale',
        'dm': 'decreto.ministeriale',
        'd.p.c.m.': 'decreto.del.presidente.del.consiglio.dei.ministri',
        'dpcm': 'decreto.del.presidente.del.consiglio.dei.ministri',
    }

    # Codice mapping
    CODICE_URNS = {
        'c.c.': 'urn:nir:stato:regio.decreto:1942-03-16;262',  # Codice Civile
        'c.p.': 'urn:nir:stato:regio.decreto:1930-10-19;1398',  # Codice Penale
        'c.p.c.': 'urn:nir:stato:regio.decreto:1940-10-28;1443',  # Codice Procedura Civile
        'c.p.p.': 'urn:nir:stato:decreto.del.presidente.della.repubblica:1988-09-22;447',  # Codice Procedura Penale
    }

    def __init__(self, base_urn: Optional[str] = None):
        """
        Initialize the citation extractor.

        Args:
            base_urn: Optional base URN for resolving relative citations
                     (e.g., "art. 5" within the same law)
        """
        self.base_urn = base_urn
        self._base_components: Optional[URNComponents] = None
        if base_urn:
            self._base_components = parse_urn(base_urn)

    def extract(self, text: str, source_urn: str) -> ExtractionResult:
        """
        Extract all citations and modifications from text.

        Args:
            text: The text to analyze
            source_urn: URN of the source article

        Returns:
            ExtractionResult with all extracted relations
        """
        if not text or not text.strip():
            return ExtractionResult(source_urn=source_urn)

        self.base_urn = source_urn
        self._base_components = parse_urn(source_urn)

        result = ExtractionResult(source_urn=source_urn)

        try:
            # Extract citations
            result.citations.extend(self._extract_codice_citations(text))
            result.citations.extend(self._extract_contextual_citations(text))
            result.citations.extend(self._extract_law_citations(text))
            result.citations.extend(self._extract_decree_citations(text))
            result.citations.extend(self._extract_jurisprudence(text))

            # Deduplicate overlapping citations
            result.citations = self._deduplicate_citations(result.citations)

            # Extract modifications
            result.modifications = self._extract_modifications(text)

        except Exception as e:
            logger.error("Error extracting relations from %s: %s", source_urn, e)
            result.errors.append(str(e))

        logger.debug(
            "Extracted %d citations and %d modifications from %s",
            len(result.citations),
            len(result.modifications),
            source_urn
        )

        return result

    def _extract_codice_citations(self, text: str) -> List[ExtractedCitation]:
        """Extract citations to codici (c.c., c.p., etc.)."""
        citations = []

        for match in self.CODICE_PATTERN.finditer(text):
            article_num = match.group(1)
            extension = match.group(2) or ""
            codice = match.group(3).lower()

            base_urn = self.CODICE_URNS.get(codice)
            if base_urn:
                target_urn = f"{base_urn}~art{article_num}{extension}"
            else:
                target_urn = None

            citation = ExtractedCitation(
                raw_text=match.group(0),
                relation_type=RelationType.CITA_ARTICOLO,
                target_urn=target_urn,
                target_article=f"{article_num}{extension}",
                context=self._get_context(text, match.start(), match.end()),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95 if base_urn else 0.7,
            )
            citations.append(citation)

        return citations

    def _extract_contextual_citations(self, text: str) -> List[ExtractedCitation]:
        """Extract citations with context markers (ai sensi, di cui, etc.)."""
        citations = []

        for match in self.CITATION_CONTEXT_PATTERN.finditer(text):
            article_num = match.group(1)
            extension = match.group(2) or ""

            # Try to resolve to full URN if we have base
            target_urn = None
            if self._base_components and self._base_components.is_valid:
                target_urn = self._build_relative_urn(article_num, extension)

            citation = ExtractedCitation(
                raw_text=match.group(0),
                relation_type=RelationType.CITA_ARTICOLO,
                target_urn=target_urn,
                target_article=f"{article_num}{extension}",
                context=self._get_context(text, match.start(), match.end()),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.9 if target_urn else 0.6,
            )
            citations.append(citation)

        return citations

    def _extract_law_citations(self, text: str) -> List[ExtractedCitation]:
        """Extract law citations (L. n. 123/2020, legge...)."""
        citations = []

        # Short form: L. n. 123/2020
        for match in self.LAW_SHORT_PATTERN.finditer(text):
            number = match.group(1)
            year = match.group(2)

            citation = ExtractedCitation(
                raw_text=match.group(0),
                relation_type=RelationType.CITA_LEGGE,
                target_act_type="legge",
                target_number=number,
                target_date=year,  # Only year available
                context=self._get_context(text, match.start(), match.end()),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.8,  # Can't build full URN without date
            )
            citations.append(citation)

        # Full form: legge 30 dicembre 2020, n. 178
        for match in self.LAW_FULL_PATTERN.finditer(text):
            day = match.group(1).zfill(2)
            month = self.MONTHS.get(match.group(2).lower(), '01')
            year = match.group(3)
            number = match.group(4)

            target_date = f"{year}-{month}-{day}"
            target_urn = f"{URN_PREFIX}legge:{target_date};{number}"

            citation = ExtractedCitation(
                raw_text=match.group(0),
                relation_type=RelationType.CITA_LEGGE,
                target_urn=target_urn,
                target_act_type="legge",
                target_date=target_date,
                target_number=number,
                context=self._get_context(text, match.start(), match.end()),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95,
            )
            citations.append(citation)

        return citations

    def _extract_decree_citations(self, text: str) -> List[ExtractedCitation]:
        """Extract decree citations (D.Lgs., D.L., etc.)."""
        citations = []

        # Short form: D.Lgs. n. 50/2016
        for match in self.DECREE_PATTERN.finditer(text):
            decree_abbr = match.group(1).lower().replace(' ', '')
            number = match.group(2)
            year = match.group(3)

            act_type = self.DECREE_TYPES.get(decree_abbr, 'decreto')

            citation = ExtractedCitation(
                raw_text=match.group(0),
                relation_type=RelationType.CITA_LEGGE,
                target_act_type=act_type,
                target_number=number,
                target_date=year,
                context=self._get_context(text, match.start(), match.end()),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.8,
            )
            citations.append(citation)

        # Full form: decreto legislativo 18 aprile 2016, n. 50
        for match in self.DECREE_FULL_PATTERN.finditer(text):
            decree_type = match.group(1).lower()
            day = match.group(2).zfill(2)
            month = self.MONTHS.get(match.group(3).lower(), '01')
            year = match.group(4)
            number = match.group(5)

            # Map to URN act type
            type_mapping = {
                'legislativo': 'decreto.legislativo',
                'legge': 'decreto.legge',
                'del presidente della repubblica': 'decreto.del.presidente.della.repubblica',
                'ministeriale': 'decreto.ministeriale',
            }
            act_type = type_mapping.get(decree_type, 'decreto')

            target_date = f"{year}-{month}-{day}"
            target_urn = f"{URN_PREFIX}{act_type}:{target_date};{number}"

            citation = ExtractedCitation(
                raw_text=match.group(0),
                relation_type=RelationType.CITA_LEGGE,
                target_urn=target_urn,
                target_act_type=act_type,
                target_date=target_date,
                target_number=number,
                context=self._get_context(text, match.start(), match.end()),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95,
            )
            citations.append(citation)

        return citations

    def _extract_jurisprudence(self, text: str) -> List[ExtractedCitation]:
        """Extract jurisprudence citations (Cassazione, Corte Cost., etc.)."""
        citations = []

        # Cassazione
        for match in self.CASSAZIONE_PATTERN.finditer(text):
            number = match.group(1)
            year = match.group(2)

            citation = ExtractedCitation(
                raw_text=match.group(0),
                relation_type=RelationType.CITA_GIURISPRUDENZA,
                target_number=number,
                target_date=year,
                target_act_type="cassazione",
                context=self._get_context(text, match.start(), match.end()),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.85,
            )
            citations.append(citation)

        # Corte Costituzionale
        for match in self.CORTE_COST_PATTERN.finditer(text):
            number = match.group(1)
            year = match.group(2)

            citation = ExtractedCitation(
                raw_text=match.group(0),
                relation_type=RelationType.CITA_GIURISPRUDENZA,
                target_number=number,
                target_date=year,
                target_act_type="corte_costituzionale",
                context=self._get_context(text, match.start(), match.end()),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.9,
            )
            citations.append(citation)

        return citations

    # Pattern to extract effective date from modification context
    DATE_EFFICACIA_PATTERN = re.compile(
        r"(?:a decorrere|con effetto|a partire)\s+dal?\s+"
        r"(\d{1,2})\s*(gennaio|febbraio|marzo|aprile|maggio|giugno|"
        r"luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4})",
        re.IGNORECASE
    )

    # Pattern to extract target article from modification context (before the modification verb)
    TARGET_ARTICLE_PATTERN = re.compile(
        r"(?:l['\u2019]?)?art(?:icol[oi])?\.?\s*(\d+)(?:-([a-z]+))?"
        r"(?:,?\s*(?:comma|c\.)\s*(\d+))?",
        re.IGNORECASE
    )

    def _extract_modifications(self, text: str) -> List[ExtractedModification]:
        """Extract modification relationships from text."""
        modifications = []

        for rel_type, pattern in self.MODIFICATION_PATTERNS.items():
            for match in pattern.finditer(text):
                context = self._get_context(text, match.start(), match.end(), window=150)

                # Extract target article from text before the modification verb
                target_urn = None
                text_before = text[max(0, match.start() - 100):match.start()]
                target_match = self.TARGET_ARTICLE_PATTERN.search(text_before)
                if target_match and self._base_components and self._base_components.is_valid:
                    article_num = target_match.group(1)
                    extension = target_match.group(2) or ""
                    target_urn = self._build_relative_urn(article_num, extension)

                # Extract effective date (data_efficacia) from context
                data_efficacia = None
                date_match = self.DATE_EFFICACIA_PATTERN.search(context)
                if date_match:
                    day = date_match.group(1).zfill(2)
                    month = self.MONTHS.get(date_match.group(2).lower(), '01')
                    year = date_match.group(3)
                    data_efficacia = f"{year}-{month}-{day}"

                modification = ExtractedModification(
                    raw_text=match.group(0),
                    relation_type=rel_type,
                    target_urn=target_urn,
                    data_efficacia=data_efficacia,
                    context=context,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.85 if target_urn else 0.7,
                )
                modifications.append(modification)

        return modifications

    def _build_relative_urn(self, article: str, extension: str = "") -> Optional[str]:
        """Build a URN for an article relative to the base URN."""
        if not self._base_components or not self._base_components.is_valid:
            return None

        # Build new URN with same act but different article
        parts = [URN_PREFIX, self._base_components.tipo_atto]

        if self._base_components.data:
            parts.append(f":{self._base_components.data}")

        if self._base_components.numero:
            parts.append(f";{self._base_components.numero}")

        if self._base_components.allegato:
            parts.append(f":{self._base_components.allegato}")

        parts.append(f"~art{article}{extension}")

        return "".join(parts)

    def _get_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 50
    ) -> str:
        """Get surrounding context for a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def _deduplicate_citations(
        self,
        citations: List[ExtractedCitation]
    ) -> List[ExtractedCitation]:
        """Remove duplicate and overlapping citations, keeping best quality ones.

        Priority: citations with resolved URNs > higher confidence > earlier position
        """
        if not citations:
            return []

        # Sort by:
        # 1. Has target_urn (resolved citations first) - descending (True=1 > False=0)
        # 2. Confidence (higher first) - descending
        # 3. Start position - ascending (earlier first)
        sorted_citations = sorted(
            citations,
            key=lambda c: (
                -(1 if c.target_urn else 0),  # Has URN first
                -c.confidence,                 # Higher confidence
                c.start_pos,                   # Earlier position
            )
        )

        result = []
        covered_ranges: List[Tuple[int, int]] = []

        for citation in sorted_citations:
            # Check if overlapping with any already kept citation
            overlaps = any(
                not (citation.end_pos <= start or citation.start_pos >= end)
                for start, end in covered_ranges
            )

            if not overlaps:
                result.append(citation)
                covered_ranges.append((citation.start_pos, citation.end_pos))

        # Sort result by original text order for cleaner output
        result.sort(key=lambda c: c.start_pos)

        return result


# =============================================================================
# Relation Creator
# =============================================================================


class RelationCreator:
    """
    Create graph edges from extracted relations.

    Handles:
    - CITA edges for citations
    - INTERPRETA edges for jurisprudence
    - Modification edges (SOSTITUISCE, ABROGA, etc.)
    """

    def __init__(self, client: "FalkorDBClient"):
        """
        Initialize relation creator.

        Args:
            client: Connected FalkorDBClient instance
        """
        self.client = client
        self.extractor = CitationExtractor()

    async def create_relations_from_text(
        self,
        source_urn: str,
        text: str,
    ) -> Dict[str, int]:
        """
        Extract and create all relations from article text.

        Args:
            source_urn: URN of the source article
            text: Text to analyze for citations

        Returns:
            Dict with counts of created edges by type
        """
        result = {
            "citations_created": 0,
            "modifications_created": 0,
            "jurisprudence_created": 0,
            "errors": 0,
        }

        # Extract relations
        extraction = self.extractor.extract(text, source_urn)

        # Create citation edges
        for citation in extraction.citations:
            try:
                if await self._create_citation_edge(source_urn, citation):
                    if citation.relation_type == RelationType.CITA_GIURISPRUDENZA:
                        result["jurisprudence_created"] += 1
                    else:
                        result["citations_created"] += 1
            except Exception as e:
                logger.error(
                    "Error creating citation edge from %s: %s",
                    source_urn, e
                )
                result["errors"] += 1

        # Create modification edges
        for modification in extraction.modifications:
            try:
                if await self._create_modification_edge(source_urn, modification):
                    result["modifications_created"] += 1
            except Exception as e:
                logger.error(
                    "Error creating modification edge from %s: %s",
                    source_urn, e
                )
                result["errors"] += 1

        logger.info(
            "Created relations from %s: %d citations, %d modifications, %d jurisprudence",
            source_urn,
            result["citations_created"],
            result["modifications_created"],
            result["jurisprudence_created"],
        )

        return result

    async def _create_citation_edge(
        self,
        source_urn: str,
        citation: ExtractedCitation,
    ) -> bool:
        """
        Create a CITA edge for an extracted citation.

        Returns True if edge was created, False otherwise.
        """
        # Must have a target URN to create edge
        if not citation.target_urn:
            logger.debug(
                "Skipping citation without resolved URN: %s",
                citation.raw_text
            )
            return False

        # Determine edge type based on relation type
        if citation.relation_type == RelationType.CITA_GIURISPRUDENZA:
            # Jurisprudence citations use INTERPRETA edge type per AC3
            edge_type = EdgeType.INTERPRETA
        else:
            edge_type = EdgeType.CITA

        # Edge properties
        properties = {
            "tipo_citazione": citation.relation_type.value,
            "confidence_score": citation.confidence,
            "paragrafo_riferimento": citation.context[:200] if citation.context else None,
        }
        properties = {k: v for k, v in properties.items() if v is not None}

        # Create the edge: source_urn -[CITA]-> target_urn
        try:
            await self.client.create_edge(
                edge_type,
                NodeType.NORMA, source_urn,
                NodeType.NORMA, citation.target_urn,
                from_key="urn",
                to_key="urn",
                properties=properties,
            )
            return True
        except Exception as e:
            # Target might not exist in graph
            logger.debug(
                "Could not create edge to %s: %s",
                citation.target_urn, e
            )
            return False

    async def _create_modification_edge(
        self,
        source_urn: str,
        modification: ExtractedModification,
    ) -> bool:
        """
        Create a modification edge (SOSTITUISCE, ABROGA, etc.).

        Returns True if edge was created, False otherwise.
        """
        # Must have a target URN
        if not modification.target_urn:
            logger.debug(
                "Skipping modification without target URN: %s",
                modification.raw_text
            )
            return False

        # Map relation type to edge type
        edge_type_mapping = {
            RelationType.SOSTITUISCE: EdgeType.SOSTITUISCE,
            RelationType.ABROGA_TOTALMENTE: EdgeType.ABROGA_TOTALMENTE,
            RelationType.ABROGA_PARZIALMENTE: EdgeType.ABROGA_PARZIALMENTE,
            RelationType.INTEGRA: EdgeType.INTEGRA,
            RelationType.SOSPENDE: EdgeType.SOSPENDE,
            RelationType.PROROGA: EdgeType.PROROGA,
            RelationType.DEROGA_A: EdgeType.DEROGA_A,
        }

        edge_type = edge_type_mapping.get(
            modification.relation_type,
            EdgeType.SOSTITUISCE  # Default
        )

        # Edge properties
        properties = {
            "data_efficacia": modification.data_efficacia,
            "testo_modificato": modification.testo_modificato,
            "testo_nuovo": modification.testo_nuovo,
            "confidence_score": modification.confidence,
        }
        properties = {k: v for k, v in properties.items() if v is not None}

        try:
            await self.client.create_edge(
                edge_type,
                NodeType.NORMA, source_urn,
                NodeType.NORMA, modification.target_urn,
                from_key="urn",
                to_key="urn",
                properties=properties,
            )
            return True
        except Exception as e:
            logger.debug(
                "Could not create modification edge to %s: %s",
                modification.target_urn, e
            )
            return False

    async def process_brocardi_relations(
        self,
        article_urn: str,
        brocardi_info: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Create INTERPRETA edges from Brocardi jurisprudence references.

        Args:
            article_urn: URN of the article
            brocardi_info: Brocardi enrichment data with massime

        Returns:
            Dict with counts
        """
        result = {
            "interpreta_edges": 0,
            "atto_nodes": 0,
        }

        massime = brocardi_info.get("massime", [])
        if not massime:
            return result

        for i, massima in enumerate(massime):
            if not massima:
                continue

            # AttoGiudiziario node should already exist from ingestion
            # Just need to ensure the INTERPRETA edge exists
            atto_node_id = f"massima_{article_urn.replace(':', '_')}_{i}"

            try:
                await self.client.create_edge(
                    EdgeType.INTERPRETA,
                    NodeType.ATTO_GIUDIZIARIO, atto_node_id,
                    NodeType.NORMA, article_urn,
                    from_key="node_id",
                    to_key="urn",
                    properties={
                        "fonte_relazione": "brocardi",
                    },
                )
                result["interpreta_edges"] += 1
            except Exception as e:
                logger.debug("Could not create INTERPRETA edge: %s", e)

        return result


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_citations(text: str, source_urn: str) -> ExtractionResult:
    """
    Convenience function to extract all citations from text.

    Args:
        text: Text to analyze
        source_urn: URN of the source article

    Returns:
        ExtractionResult with extracted relations
    """
    extractor = CitationExtractor()
    return extractor.extract(text, source_urn)
