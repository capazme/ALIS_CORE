"""
Precedent Expert for MERL-T Analysis Pipeline.

Implements jurisprudential interpretation from case law:
- Cassazione decisions (highest authority)
- Massime (legal maxims)
- Appellate and tribunal decisions
- Trend analysis and conflict detection

The Art. 12 Preleggi hierarchy places jurisprudence as supporting interpretation,
not as primary source, but courts have established authoritative positions.

Court Authority Hierarchy (for weight calculation):
1. Corte di Cassazione (Sezioni Unite > Sezioni Semplici)
2. Corte d'Appello
3. Tribunale
"""

import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BaseExpert,
    ExpertConfig,
    ExpertContext,
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    ConfidenceFactors,
    FeedbackHook,
    ChunkRetriever,
    LLMService,
)

log = structlog.get_logger()


# Default prompt for jurisprudential analysis
PRECEDENT_PROMPT_TEMPLATE = """Sei un esperto di giurisprudenza italiana.
Il tuo compito è analizzare la seguente domanda giuridica alla luce delle decisioni giurisprudenziali rilevanti.

QUERY: {query}

NORMA DI RIFERIMENTO:
{main_norm}

GIURISPRUDENZA RILEVANTE:
{case_law}

MASSIME:
{massime}

ISTRUZIONI:
1. Identifica le decisioni più autorevoli (privilegiando Cassazione, Sezioni Unite)
2. Analizza l'interpretazione prevalente nella giurisprudenza
3. Distingui tra orientamenti consolidati e in evoluzione
4. Se esistono contrasti giurisprudenziali, presentali esplicitamente
5. Indica la posizione più recente e autorevole

FORMATO OUTPUT:
- Inizia con l'orientamento giurisprudenziale prevalente
- Cita le decisioni chiave con riferimento completo (Corte, data, numero)
- Riporta le massime più significative
- Se c'è conflitto, descrivi le posizioni contrastanti
- Indica eventuali tendenze evolutive recenti
- Fornisci una sintesi dell'interpretazione giurisprudenziale

Rispondi in italiano."""


class CourtAuthority(IntEnum):
    """Court authority levels for weighting."""
    CASSAZIONE_SU = 100  # Sezioni Unite
    CASSAZIONE = 80      # Sezioni Semplici
    APPELLO = 50
    TRIBUNALE = 30
    OTHER = 10


# Court name patterns for authority detection (immutable)
_COURT_PATTERNS: Dict[str, CourtAuthority] = {
    "sezioni unite": CourtAuthority.CASSAZIONE_SU,
    "ss.uu.": CourtAuthority.CASSAZIONE_SU,
    "s.u.": CourtAuthority.CASSAZIONE_SU,
    "cassazione": CourtAuthority.CASSAZIONE,
    "cass.": CourtAuthority.CASSAZIONE,
    "appello": CourtAuthority.APPELLO,
    "app.": CourtAuthority.APPELLO,
    "tribunale": CourtAuthority.TRIBUNALE,
    "trib.": CourtAuthority.TRIBUNALE,
}
COURT_PATTERNS = MappingProxyType(_COURT_PATTERNS)


@dataclass
class CaseDecision:
    """
    Represents a court decision.

    Attributes:
        case_id: Unique identifier
        court: Court name (e.g., "Cassazione civile")
        section: Section (e.g., "Sez. III", "Sezioni Unite")
        date: Decision date
        number: Case number
        massima: Legal maxim/summary
        excerpt: Relevant excerpt from decision
        authority_score: Computed authority score
        relevance_score: Relevance to query
    """

    case_id: str
    court: str
    section: str = ""
    date: str = ""
    number: str = ""
    massima: str = ""
    excerpt: str = ""
    authority_score: float = 0.5
    relevance_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "case_id": self.case_id,
            "court": self.court,
            "section": self.section,
            "date": self.date,
            "number": self.number,
            "massima": self.massima[:200] if self.massima else "",
            "authority_score": round(self.authority_score, 3),
            "relevance_score": round(self.relevance_score, 3),
        }

    @property
    def citation(self) -> str:
        """Generate formal citation."""
        parts = [self.court]
        if self.section:
            parts.append(self.section)
        if self.date:
            parts.append(self.date)
        if self.number:
            parts.append(f"n. {self.number}")
        return ", ".join(parts)


@dataclass
class PrecedentConfig(ExpertConfig):
    """Configuration specific to PrecedentExpert."""

    # Retrieval
    max_case_chunks: int = 10
    max_massime: int = 5
    min_case_score: float = 0.4

    # Authority weighting
    prefer_recent: bool = True
    recent_years_boost: int = 5  # Years to consider "recent"
    recency_boost_factor: float = 0.2  # Max boost for recent decisions (0-1)
    sezioni_unite_boost: float = 0.15  # Boost for Sezioni Unite beyond base authority

    # Conflict detection
    detect_conflicts: bool = True
    min_conflict_indicators: int = 1  # Minimum conflict indicators to flag

    # LLM
    precedent_temperature: float = 0.3
    max_response_tokens: int = 2500

    # Confidence thresholds
    min_cases: int = 1
    high_confidence_threshold: float = 0.7
    no_cases_confidence: float = 0.25

    # F6 Feedback
    enable_f6_feedback: bool = True


class PrecedentExpert(BaseExpert):
    """
    Expert for jurisprudential interpretation from case law.

    Epistemology: How courts have applied the norm
    Focus: Authoritative judicial interpretation and trends

    Output:
    - "Giurisprudenza" section header
    - Key decisions with citations
    - Massime relevant to the query
    - Trend analysis
    - Conflict detection
    - Confidence score

    Example:
        >>> retriever = BridgeTableRetriever(...)
        >>> llm = LLMService(...)
        >>> expert = PrecedentExpert(retriever=retriever, llm_service=llm)
        >>> context = ExpertContext(query_text="Come la Cassazione interpreta l'art. 1453?")
        >>> response = await expert.analyze(context)
        >>> print(response.section_header)
        "Giurisprudenza"
    """

    expert_type = "precedent"
    section_header = "Giurisprudenza"
    description = "Interpretazione giurisprudenziale da Cassazione e corti di merito"

    def __init__(
        self,
        retriever: Optional[ChunkRetriever] = None,
        llm_service: Optional[LLMService] = None,
        config: Optional[PrecedentConfig] = None,
    ):
        """
        Initialize PrecedentExpert.

        Args:
            retriever: Chunk retriever for Bridge Table access
            llm_service: LLM service for analysis
            config: PrecedentExpert configuration
        """
        self._config = config or PrecedentConfig()
        super().__init__(
            retriever=retriever,
            llm_service=llm_service,
            config=self._config,
        )

    async def analyze(self, context: ExpertContext) -> ExpertResponse:
        """
        Analyze query with jurisprudential interpretation.

        Flow:
        1. Retrieve case law chunks (source_type=jurisprudence)
        2. Retrieve massime from Brocardi
        3. Extract and rank case decisions by authority
        4. Detect conflicts between decisions
        5. Generate interpretation via LLM
        6. Compute confidence based on case law availability

        Args:
            context: Input context with query and entities

        Returns:
            ExpertResponse with jurisprudential interpretation
        """
        start_time = time.time()

        log.info(
            "precedent_expert_analyzing",
            query=context.query_text[:50],
            trace_id=context.trace_id,
            norm_refs=len(context.norm_references),
        )

        # Step 1: Retrieve case law chunks
        case_chunks = await self._retrieve_case_law(context)

        # Step 2: Retrieve massime
        massime_chunks = await self._retrieve_massime(context)

        # Step 3: Retrieve main norm context
        main_norm_chunks = await self._retrieve_main_norm(context)

        # Step 4: Extract case decisions with authority ranking
        cases = self._extract_cases(case_chunks + massime_chunks)

        # Step 5: Check if we have enough context
        if not cases and not massime_chunks:
            execution_time = (time.time() - start_time) * 1000
            return self._create_no_cases_response(
                context=context,
                main_norm_chunks=main_norm_chunks,
                execution_time_ms=execution_time,
            )

        # Step 6: Detect conflicts
        has_conflict = False
        conflict_details = ""
        if self._config.detect_conflicts and len(cases) >= 2:
            has_conflict, conflict_details = self._detect_conflicts(cases)

        # Step 7: Build legal sources
        legal_sources = self._build_legal_sources(main_norm_chunks, cases)

        # Step 8: Generate interpretation
        if self.llm_service:
            interpretation, reasoning_steps, tokens = await self._generate_interpretation(
                context, main_norm_chunks, cases, has_conflict, conflict_details
            )
        else:
            interpretation, reasoning_steps, tokens = self._generate_fallback_interpretation(
                context, main_norm_chunks, cases, has_conflict, conflict_details
            )

        # Step 9: Compute confidence
        confidence, factors = self._compute_confidence(cases, has_conflict)

        execution_time = (time.time() - start_time) * 1000

        # Create F6 feedback hook
        feedback_hook = None
        if self._config.enable_f6_feedback:
            feedback_hook = FeedbackHook(
                feedback_type="F6",
                expert_type=self.expert_type,
                response_id=context.trace_id,
                enabled=True,
                correction_options={
                    # Case relevance assessment
                    "case_relevance": [
                        "all_relevant",      # All cases are on point
                        "mostly_relevant",   # Most cases apply
                        "some_irrelevant",   # Some cases don't apply
                        "mostly_irrelevant", # Most cases don't apply
                    ],
                    # Jurisprudence currency
                    "jurisprudence_currency": [
                        "current",           # Cases reflect current law
                        "mostly_current",    # Some dated but still valid
                        "outdated",          # Cases may be superseded
                    ],
                    # Conflict detection accuracy
                    "conflict_detection": [
                        "correctly_detected",    # Conflicts accurately identified
                        "missed_conflict",       # Undetected jurisprudential conflict
                        "false_conflict",        # Reported conflict doesn't exist
                        "no_conflict",           # Correctly reported no conflict
                    ],
                    # Hierarchy respect (Cassazione > Merito)
                    "hierarchy_respect": [
                        "properly_weighted",     # Higher courts given proper weight
                        "underweighted_supreme", # Cassazione underweighted
                        "overweighted_merit",    # Lower courts overweighted
                    ],
                    # Confidence calibration
                    "confidence_calibration": [
                        "well_calibrated",
                        "overconfident",
                        "underconfident",
                    ],
                },
                context_snapshot={
                    "query": context.query_text[:200],
                    "cases_count": len(cases),
                    "has_conflict": has_conflict,
                    "confidence": confidence,
                    "interpretation_preview": interpretation[:300] if interpretation else "",
                    "top_case_courts": [c.court for c in cases[:3]] if cases else [],
                },
            )

        response = ExpertResponse(
            expert_type=self.expert_type,
            section_header=self.section_header,
            interpretation=interpretation,
            legal_basis=legal_sources,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            confidence_factors=factors,
            limitations=self._identify_limitations(cases, has_conflict),
            suggestions=self._generate_suggestions(confidence, has_conflict),
            trace_id=context.trace_id,
            execution_time_ms=execution_time,
            tokens_used=tokens,
            feedback_hook=feedback_hook,
            metadata={
                "cases_found": len(cases),
                "has_conflict": has_conflict,
                "conflict_details": conflict_details if has_conflict else "",
                "top_cases": [c.to_dict() for c in cases[:5]],
                "devils_advocate_flag": has_conflict,  # Flag for Devil's Advocate
            },
        )

        log.info(
            "precedent_expert_completed",
            trace_id=context.trace_id,
            confidence=confidence,
            cases_count=len(cases),
            has_conflict=has_conflict,
            execution_time_ms=execution_time,
        )

        return response

    async def _retrieve_case_law(
        self, context: ExpertContext
    ) -> List[Dict[str, Any]]:
        """
        Retrieve case law chunks from Bridge Table.

        Args:
            context: Input context

        Returns:
            List of jurisprudence chunks
        """
        if not self.retriever:
            return []

        filters = {
            "expert_affinity": "precedent",
            "source_type": "jurisprudence",
        }

        try:
            chunks = await self.retriever.retrieve(
                query=context.query_text,
                query_embedding=context.query_embedding,
                filters=filters,
                limit=self._config.max_case_chunks,
            )

            # Filter by score
            return [
                c for c in (chunks or [])
                if c.get("score", 0) >= self._config.min_case_score
            ]

        except Exception as e:
            log.warning(
                "precedent_case_retrieval_error",
                error=str(e),
                trace_id=context.trace_id,
            )
            return []

    async def _retrieve_massime(
        self, context: ExpertContext
    ) -> List[Dict[str, Any]]:
        """
        Retrieve massime (legal maxims) from Bridge Table.

        Args:
            context: Input context

        Returns:
            List of massime chunks filtered by min_case_score
        """
        if not self.retriever:
            return []

        filters = {
            "source_type": "massima",
        }

        try:
            chunks = await self.retriever.retrieve(
                query=context.query_text,
                query_embedding=context.query_embedding,
                filters=filters,
                limit=self._config.max_massime,
            )
            # Filter by score (same threshold as cases)
            return [
                c for c in (chunks or [])
                if c.get("score", 0) >= self._config.min_case_score
            ]

        except Exception as e:
            log.warning(
                "precedent_massime_retrieval_error",
                error=str(e),
                trace_id=context.trace_id,
            )
            return []

    async def _retrieve_main_norm(
        self, context: ExpertContext
    ) -> List[Dict[str, Any]]:
        """Retrieve main norm context."""
        if context.retrieved_chunks:
            return context.retrieved_chunks

        if not self.retriever:
            return []

        filters = {"source_type": "norm"}
        if context.norm_references:
            filters["urns"] = context.norm_references

        try:
            chunks = await self.retriever.retrieve(
                query=context.query_text,
                query_embedding=context.query_embedding,
                filters=filters,
                limit=self._config.chunk_limit,
            )
            return chunks or []

        except Exception as e:
            log.warning(
                "precedent_norm_retrieval_error",
                error=str(e),
                trace_id=context.trace_id,
            )
            return []

    def _extract_cases(
        self, chunks: List[Dict[str, Any]]
    ) -> List[CaseDecision]:
        """
        Extract and rank case decisions from chunks.

        Args:
            chunks: Retrieved jurisprudence/massime chunks

        Returns:
            List of CaseDecision objects sorted by authority
        """
        cases: List[CaseDecision] = []
        seen_ids: set = set()

        for chunk in chunks:
            case_id = chunk.get("case_id", chunk.get("id", ""))

            # Deduplicate
            if case_id in seen_ids:
                continue
            seen_ids.add(case_id)

            # Extract court info
            court = chunk.get("court", "")
            section = chunk.get("section", "")
            text = chunk.get("text", "")

            # Compute authority score
            authority = self._compute_authority(court, section, chunk)

            case = CaseDecision(
                case_id=case_id,
                court=court or self._infer_court(text),
                section=section,
                date=chunk.get("date", ""),
                number=chunk.get("number", chunk.get("case_number", "")),
                massima=chunk.get("massima", ""),
                excerpt=text[:500] if text else "",
                authority_score=authority,
                relevance_score=chunk.get("score", 0.5),
            )
            cases.append(case)

        # Sort by combined score (authority * relevance)
        cases.sort(
            key=lambda c: c.authority_score * c.relevance_score,
            reverse=True,
        )

        return cases

    def _compute_authority(
        self,
        court: str,
        section: str,
        chunk: Dict[str, Any],
    ) -> float:
        """
        Compute authority score for a case decision.

        Args:
            court: Court name
            section: Section name
            chunk: Full chunk data

        Returns:
            Authority score [0-1]
        """
        base_authority = CourtAuthority.OTHER

        # Check court patterns
        court_lower = court.lower()
        section_lower = section.lower()
        combined = f"{court_lower} {section_lower}"

        for pattern, authority in COURT_PATTERNS.items():
            if pattern in combined:
                if authority > base_authority:
                    base_authority = authority

        # Normalize to [0-1] - use CASSAZIONE as base (Sezioni Unite gets boost)
        authority_score = base_authority / CourtAuthority.CASSAZIONE_SU

        # Apply Sezioni Unite boost (additive, not multiplicative)
        if "sezioni unite" in combined or "ss.uu" in combined or "s.u." in combined:
            authority_score = min(1.0, authority_score + self._config.sezioni_unite_boost)

        # Recency boost (configurable factor)
        if self._config.prefer_recent:
            date_str = chunk.get("date", "")
            recency_boost = self._compute_recency_boost(date_str)
            authority_score = min(1.0, authority_score * (1 + recency_boost * self._config.recency_boost_factor))

        return authority_score

    def _compute_recency_boost(self, date_str: str) -> float:
        """
        Compute recency boost factor.

        Args:
            date_str: Date string (various formats)

        Returns:
            Boost factor [0-1] where 1 = very recent
        """
        if not date_str:
            return 0.0

        try:
            # Try common Italian date formats
            for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y"]:
                try:
                    date = datetime.strptime(date_str, fmt)
                    years_ago = (datetime.now() - date).days / 365
                    if years_ago <= self._config.recent_years_boost:
                        return 1.0 - (years_ago / self._config.recent_years_boost)
                    return 0.0
                except ValueError:
                    continue
        except Exception:
            pass

        return 0.0

    def _infer_court(self, text: str) -> str:
        """
        Infer court from text content using COURT_PATTERNS.

        Args:
            text: Text content to analyze

        Returns:
            Inferred court name string
        """
        text_lower = text.lower()
        best_authority = CourtAuthority.OTHER
        best_court = "Giurisprudenza"

        # Use COURT_PATTERNS for consistent detection
        for pattern, authority in COURT_PATTERNS.items():
            if pattern in text_lower:
                if authority > best_authority:
                    best_authority = authority
                    # Map authority to court name
                    if authority == CourtAuthority.CASSAZIONE_SU:
                        best_court = "Cassazione civile, Sezioni Unite"
                    elif authority == CourtAuthority.CASSAZIONE:
                        best_court = "Cassazione civile"
                    elif authority == CourtAuthority.APPELLO:
                        best_court = "Corte d'Appello"
                    elif authority == CourtAuthority.TRIBUNALE:
                        best_court = "Tribunale"

        return best_court

    def _detect_conflicts(
        self, cases: List[CaseDecision]
    ) -> Tuple[bool, str]:
        """
        Detect conflicts between case decisions.

        Uses pattern matching with configurable minimum indicator threshold.

        Args:
            cases: List of case decisions

        Returns:
            Tuple of (has_conflict, conflict_description)
        """
        # Use specific phrases to avoid false positives (e.g., "contra" in "contratto")
        conflict_indicators = [
            "in senso contrario",
            "contra ",  # Space to avoid "contratto"
            " contra,",  # With comma
            " contra.",  # With period
            "diversamente si è ritenuto",
            "orientamento opposto",
            "contrasto giurisprudenziale",
            "in difformità",
            "critica l'orientamento",
            "non condivide",
            "orientamento minoritario",
        ]

        conflicting_cases: List[str] = []
        indicator_count = 0

        for case in cases:
            text = f"{case.massima} {case.excerpt}".lower()
            case_indicators = sum(1 for ind in conflict_indicators if ind in text)
            if case_indicators > 0:
                conflicting_cases.append(case.citation)
                indicator_count += case_indicators

        # Check against configured minimum threshold
        if indicator_count >= self._config.min_conflict_indicators and conflicting_cases:
            return True, f"Contrasto rilevato in: {', '.join(conflicting_cases[:3])}"

        # Also check if we have Cassazione and lower court disagreeing
        cassazione_cases = [c for c in cases if "cassazione" in c.court.lower()]
        lower_cases = [c for c in cases if "cassazione" not in c.court.lower()]

        if cassazione_cases and lower_cases:
            # Simplified check: if both exist, note potential hierarchy
            return False, ""

        return False, ""

    def _build_legal_sources(
        self,
        main_norm_chunks: List[Dict[str, Any]],
        cases: List[CaseDecision],
    ) -> List[LegalSource]:
        """Build LegalSource list from all sources."""
        sources: List[LegalSource] = []

        # Main norms
        for chunk in main_norm_chunks[:2]:
            sources.append(
                LegalSource(
                    source_type="norm",
                    source_id=chunk.get("urn", chunk.get("id", "")),
                    citation=chunk.get("citation", chunk.get("title", "")),
                    excerpt=chunk.get("text", "")[:300],
                    relevance="Norma di riferimento",
                    relevance_score=0.8,  # Main norm reference
                )
            )

        # Case decisions (sorted by authority already)
        for case in cases:
            sources.append(
                LegalSource(
                    source_type="jurisprudence",
                    source_id=case.case_id,
                    citation=case.citation,
                    excerpt=case.massima or case.excerpt[:300],
                    relevance=f"Autorità: {case.authority_score:.2f}",
                    relevance_score=round(min(1.0, case.authority_score), 3),
                )
            )

        return sources

    async def _generate_interpretation(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        cases: List[CaseDecision],
        has_conflict: bool,
        conflict_details: str,
    ) -> Tuple[str, List[ReasoningStep], int]:
        """Generate interpretation using LLM."""
        main_norm_text = self._format_chunks_for_prompt(main_norm_chunks)
        case_law_text = self._format_cases_for_prompt(cases)
        massime_text = self._format_massime_for_prompt(cases)

        prompt = PRECEDENT_PROMPT_TEMPLATE.format(
            query=context.query_text,
            main_norm=main_norm_text or "Non specificata",
            case_law=case_law_text or "Nessuna giurisprudenza specifica recuperata",
            massime=massime_text or "Nessuna massima disponibile",
        )

        if has_conflict:
            prompt += f"\n\nATTENZIONE: {conflict_details}. Analizza il contrasto."

        try:
            interpretation = await self.llm_service.generate(
                prompt=prompt,
                temperature=self._config.precedent_temperature,
                max_tokens=self._config.max_response_tokens,
            )
            # Approximate token count
            tokens_used = len(prompt.split()) + len(interpretation.split())
        except Exception as e:
            log.exception(
                "precedent_llm_error",
                error=str(e),
                trace_id=context.trace_id,
                exc_info=True,
            )
            interpretation, _, tokens_used = self._generate_fallback_interpretation(
                context, main_norm_chunks, cases, has_conflict, conflict_details
            )

        reasoning_steps = self._build_reasoning_steps(cases, has_conflict)

        return interpretation, reasoning_steps, tokens_used

    def _generate_fallback_interpretation(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        cases: List[CaseDecision],
        has_conflict: bool,
        conflict_details: str,
    ) -> Tuple[str, List[ReasoningStep], int]:
        """Generate interpretation without LLM."""
        parts: List[str] = []

        if cases:
            parts.append("**Giurisprudenza rilevante:**\n")
            for case in cases[:5]:
                parts.append(f"- **{case.citation}**\n")
                if case.massima:
                    parts.append(f"  *Massima:* \"{case.massima[:150]}...\"\n")

        if has_conflict:
            parts.append(f"\n**⚠️ Contrasto giurisprudenziale:**\n{conflict_details}\n")

        if main_norm_chunks:
            parts.append("\n**Norma di riferimento:**\n")
            for chunk in main_norm_chunks[:1]:
                citation = chunk.get("citation", "Norma")
                parts.append(f"- {citation}\n")

        if not parts:
            parts.append(
                "Non sono state reperite decisioni giurisprudenziali rilevanti per questa query. "
                "L'interpretazione giurisprudenziale richiede la consultazione di banche dati specifiche."
            )

        interpretation = "\n".join(parts)
        reasoning_steps = self._build_reasoning_steps(cases, has_conflict)

        return interpretation, reasoning_steps, 0

    def _format_chunks_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format norm chunks for prompt."""
        if not chunks:
            return ""

        formatted: List[str] = []
        for chunk in chunks[:2]:
            citation = chunk.get("citation", chunk.get("title", "Fonte"))
            text = chunk.get("text", "")[:400]
            formatted.append(f"[{citation}]\n{text}\n")

        return "\n".join(formatted)

    def _format_cases_for_prompt(self, cases: List[CaseDecision]) -> str:
        """Format case decisions for prompt."""
        if not cases:
            return ""

        formatted: List[str] = []
        for case in cases[:6]:
            formatted.append(f"[{case.citation}]")
            if case.excerpt:
                formatted.append(f"{case.excerpt[:300]}...")
            formatted.append("")

        return "\n".join(formatted)

    def _format_massime_for_prompt(self, cases: List[CaseDecision]) -> str:
        """Format massime for prompt."""
        massime = [c for c in cases if c.massima]
        if not massime:
            return ""

        formatted: List[str] = []
        for case in massime[:5]:
            formatted.append(f"- {case.citation}: \"{case.massima[:200]}...\"")

        return "\n".join(formatted)

    def _build_reasoning_steps(
        self,
        cases: List[CaseDecision],
        has_conflict: bool,
    ) -> List[ReasoningStep]:
        """Build reasoning chain steps."""
        steps: List[ReasoningStep] = []
        step_num = 1

        if cases:
            case_ids = [c.case_id for c in cases]
            cassazione_count = len([c for c in cases if "cassazione" in c.court.lower()])
            steps.append(
                self._build_reasoning_step(
                    step_number=step_num,
                    description=f"Recuperate {len(cases)} decisioni ({cassazione_count} di Cassazione)",
                    source_ids=case_ids,
                )
            )
            step_num += 1

            if cases[0].authority_score > 0.7:
                steps.append(
                    self._build_reasoning_step(
                        step_number=step_num,
                        description=f"Decisione più autorevole: {cases[0].citation}",
                        source_ids=[cases[0].case_id],
                    )
                )
                step_num += 1

        if has_conflict:
            steps.append(
                self._build_reasoning_step(
                    step_number=step_num,
                    description="Rilevato contrasto giurisprudenziale - analisi delle posizioni",
                )
            )
            step_num += 1

        steps.append(
            self._build_reasoning_step(
                step_number=step_num,
                description="Sintesi dell'interpretazione giurisprudenziale",
            )
        )

        return steps

    def _compute_confidence(
        self,
        cases: List[CaseDecision],
        has_conflict: bool,
    ) -> Tuple[float, ConfidenceFactors]:
        """Compute confidence score based on case law."""
        # Source availability
        cases_threshold = max(2, self._config.min_cases * 2)
        source_availability = min(1.0, len(cases) / cases_threshold)

        # Norm clarity (based on highest authority case)
        if cases:
            norm_clarity = max(c.authority_score for c in cases)
        else:
            norm_clarity = 0.2

        # Definition coverage (based on massime availability)
        massime_count = len([c for c in cases if c.massima])
        definition_coverage = min(1.0, massime_count / max(1, self._config.max_massime))

        # Contextual ambiguity
        if has_conflict:
            contextual_ambiguity = 0.7  # High ambiguity due to conflict
        elif len(cases) >= 3:
            contextual_ambiguity = 0.3  # Multiple consistent cases = low ambiguity
        else:
            contextual_ambiguity = 0.5

        factors = ConfidenceFactors(
            norm_clarity=norm_clarity,
            source_availability=source_availability,
            contextual_ambiguity=contextual_ambiguity,
            definition_coverage=definition_coverage,
        )

        confidence = factors.compute_overall()

        return confidence, factors

    def _create_no_cases_response(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        execution_time_ms: float,
    ) -> ExpertResponse:
        """Create response when no case law is found."""
        interpretation = (
            "Non sono state reperite decisioni giurisprudenziali specifiche "
            "per questa domanda. La norma potrebbe non essere stata oggetto "
            "di contenzioso rilevante, oppure le banche dati potrebbero non coprire la materia."
        )

        if main_norm_chunks:
            interpretation += "\n\n**Norma analizzata:**\n"
            for chunk in main_norm_chunks[:1]:
                citation = chunk.get("citation", "")
                interpretation += f"- {citation}\n"

        feedback_hook = None
        if self._config.enable_f6_feedback:
            feedback_hook = FeedbackHook(
                feedback_type="F6",
                expert_type=self.expert_type,
                response_id=context.trace_id,
                enabled=True,
                correction_options={
                    # No cases assessment
                    "cases_availability": [
                        "correctly_unavailable",  # No cases really exist
                        "cases_exist",            # Cases exist but weren't found
                        "search_failed",          # Technical failure
                    ],
                    # Should this expert have been skipped?
                    "expert_applicability": [
                        "correctly_limited",      # Expert correctly limited
                        "should_have_found",      # Should have found cases
                        "query_not_applicable",   # Query doesn't need jurisprudence
                    ],
                    # Confidence calibration
                    "confidence_calibration": [
                        "well_calibrated",
                        "overconfident",
                        "underconfident",
                    ],
                },
                context_snapshot={
                    "query": context.query_text[:200],
                    "no_cases": True,
                    "main_norm_count": len(main_norm_chunks),
                    "confidence": self._config.no_cases_confidence,
                },
            )

        return ExpertResponse(
            expert_type=self.expert_type,
            section_header=self.section_header,
            interpretation=interpretation,
            legal_basis=[
                LegalSource(
                    source_type="norm",
                    source_id=c.get("urn", ""),
                    citation=c.get("citation", ""),
                    excerpt=c.get("text", "")[:200],
                    relevance="Norma di riferimento",
                    relevance_score=0.5,  # Limited relevance (no cases found)
                )
                for c in main_norm_chunks[:1]
            ],
            confidence=self._config.no_cases_confidence,
            confidence_factors=ConfidenceFactors(
                norm_clarity=0.3,
                source_availability=0.1,
                contextual_ambiguity=0.7,
                definition_coverage=0.1,
            ),
            limitations="Nessuna giurisprudenza rilevante recuperata",
            suggestions="Consultare direttamente banche dati giuridiche (DeJure, ItalGiure)",
            trace_id=context.trace_id,
            execution_time_ms=execution_time_ms,
            feedback_hook=feedback_hook,
            metadata={
                "cases_found": 0,
                "has_conflict": False,
            },
        )

    def _identify_limitations(
        self,
        cases: List[CaseDecision],
        has_conflict: bool,
    ) -> str:
        """Identify limitations of the analysis."""
        limitations: List[str] = []

        if not cases:
            limitations.append("Nessuna giurisprudenza recuperata")

        cassazione_count = len([c for c in cases if "cassazione" in c.court.lower()])
        if cases and cassazione_count == 0:
            limitations.append("Nessuna decisione di Cassazione disponibile")

        if has_conflict:
            limitations.append("Contrasto giurisprudenziale rilevato - interpretazione non univoca")

        if not self.llm_service:
            limitations.append("Analisi eseguita senza supporto LLM")

        return "; ".join(limitations) if limitations else ""

    def _generate_suggestions(
        self,
        confidence: float,
        has_conflict: bool,
    ) -> str:
        """Generate suggestions based on confidence and conflicts."""
        suggestions: List[str] = []

        if confidence < self._config.high_confidence_threshold:
            suggestions.append("Consultare banche dati giuridiche per giurisprudenza più recente")

        if has_conflict:
            suggestions.append("Verificare l'orientamento più recente della Cassazione")
            suggestions.append("Considerare analisi Devil's Advocate per posizioni alternative")

        return "; ".join(suggestions)
