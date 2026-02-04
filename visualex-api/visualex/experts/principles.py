"""
Principles Expert for MERL-T Analysis Pipeline.

Implements teleological interpretation following Art. 12, comma II:
- "...intenzione del legislatore..."
- Ratio legis (purpose of the law)

Teleological interpretation considers:
- RATIO LEGIS: Purpose and objectives of the norm
- PRINCIPI: Underlying legal principles (buona fede, tutela affidamento, etc.)
- COSTITUZIONALE: Constitutional values that inform interpretation
- DOTTRINA: Doctrinal commentary on purpose

Approach:
1. Identify relevant legal principles from the query context
2. Retrieve doctrine chunks with high Principles affinity
3. Analyze ratio legis through LLM synthesis
4. Highlight tensions between principles when present
"""

import time
import structlog
from dataclasses import dataclass, field
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


# Default prompt for principles/teleological interpretation
PRINCIPLES_PROMPT_TEMPLATE = """Sei un esperto di interpretazione teleologica del diritto italiano.
Il tuo compito è analizzare la seguente domanda giuridica considerando la RATIO LEGIS e i PRINCIPI GENERALI che governano la materia.

QUERY: {query}

NORMA DI RIFERIMENTO:
{main_norm}

PRINCIPI GIURIDICI RILEVANTI:
{principles}

DOTTRINA E COMMENTI:
{doctrine}

ISTRUZIONI:
1. Identifica la ratio legis (scopo) della norma
2. Individua i principi generali applicabili (buona fede, tutela affidamento, favor debitoris, etc.)
3. Considera i valori costituzionali che informano l'interpretazione
4. Analizza la dottrina per comprendere l'intenzione del legislatore
5. Evidenzia eventuali tensioni tra principi diversi

FORMATO OUTPUT:
- Inizia con la ratio legis della norma
- Elenca i principi giuridici applicabili con breve definizione
- Spiega come i principi guidano l'interpretazione
- Cita il supporto dottrinale
- Se ci sono posizioni dottrinali discordanti, presentale tutte
- Fornisci una sintesi dell'interpretazione teleologica

Rispondi in italiano."""


# Common Italian legal principles for matching (immutable)
_LEGAL_PRINCIPLES_DICT: Dict[str, Dict[str, Any]] = {
    "buona_fede": {
        "name": "Buona fede",
        "definition": "Obbligo di comportarsi secondo correttezza (art. 1175 c.c.)",
        "articles": ["art. 1175 c.c.", "art. 1337 c.c.", "art. 1375 c.c."],
        "category": "civil",
    },
    "tutela_affidamento": {
        "name": "Tutela dell'affidamento",
        "definition": "Protezione della ragionevole aspettativa delle parti",
        "articles": ["art. 1338 c.c."],
        "category": "civil",
    },
    "favor_debitoris": {
        "name": "Favor debitoris",
        "definition": "In caso di dubbio, interpretazione favorevole al debitore",
        "articles": ["art. 1371 c.c."],
        "category": "civil",
    },
    "conservazione_contratto": {
        "name": "Conservazione del contratto",
        "definition": "Preferenza per l'interpretazione che conserva effetti (art. 1367 c.c.)",
        "articles": ["art. 1367 c.c."],
        "category": "civil",
    },
    "autonomia_privata": {
        "name": "Autonomia privata",
        "definition": "Libertà delle parti di determinare il contenuto contrattuale (art. 1322 c.c.)",
        "articles": ["art. 1322 c.c."],
        "category": "civil",
    },
    "equilibrio_contrattuale": {
        "name": "Equilibrio contrattuale",
        "definition": "Proporzione tra prestazioni corrispettive",
        "articles": ["art. 1448 c.c."],
        "category": "civil",
    },
    "diligenza": {
        "name": "Diligenza del buon padre di famiglia",
        "definition": "Standard di comportamento richiesto (art. 1176 c.c.)",
        "articles": ["art. 1176 c.c."],
        "category": "civil",
    },
    "neminem_laedere": {
        "name": "Neminem laedere",
        "definition": "Obbligo di non arrecare danno ad altri (art. 2043 c.c.)",
        "articles": ["art. 2043 c.c."],
        "category": "civil",
    },
}

# Constitutional principles (enabled via include_constitutional config)
_CONSTITUTIONAL_PRINCIPLES_DICT: Dict[str, Dict[str, Any]] = {
    "uguaglianza": {
        "name": "Principio di uguaglianza",
        "definition": "Tutti i cittadini hanno pari dignità sociale e sono eguali davanti alla legge",
        "articles": ["art. 3 Cost."],
        "category": "constitutional",
    },
    "solidarieta": {
        "name": "Dovere di solidarietà",
        "definition": "Doveri inderogabili di solidarietà politica, economica e sociale",
        "articles": ["art. 2 Cost."],
        "category": "constitutional",
    },
    "libera_iniziativa": {
        "name": "Libertà di iniziativa economica",
        "definition": "L'iniziativa economica privata è libera, nei limiti dell'utilità sociale",
        "articles": ["art. 41 Cost."],
        "category": "constitutional",
    },
    "tutela_risparmio": {
        "name": "Tutela del risparmio",
        "definition": "La Repubblica tutela il risparmio in tutte le sue forme",
        "articles": ["art. 47 Cost."],
        "category": "constitutional",
    },
    "proprieta_privata": {
        "name": "Proprietà privata",
        "definition": "La proprietà privata è riconosciuta e garantita dalla legge con funzione sociale",
        "articles": ["art. 42 Cost."],
        "category": "constitutional",
    },
}

# Immutable exports using MappingProxyType to prevent accidental modification
LEGAL_PRINCIPLES = MappingProxyType(_LEGAL_PRINCIPLES_DICT)
CONSTITUTIONAL_PRINCIPLES = MappingProxyType(_CONSTITUTIONAL_PRINCIPLES_DICT)


@dataclass
class PrinciplesConfig(ExpertConfig):
    """Configuration specific to PrinciplesExpert."""

    # Retrieval
    max_doctrine_chunks: int = 8
    min_doctrine_score: float = 0.4
    include_constitutional: bool = True

    # LLM
    principles_temperature: float = 0.4
    max_response_tokens: int = 2500

    # Confidence thresholds
    min_principles: int = 1
    high_confidence_threshold: float = 0.65
    no_doctrine_confidence: float = 0.35

    # F5 Feedback
    enable_f5_feedback: bool = True


@dataclass
class IdentifiedPrinciple:
    """
    Represents an identified legal principle.

    Attributes:
        principle_id: Internal identifier (e.g., "buona_fede")
        name: Italian name
        definition: Brief definition
        articles: Related articles
        relevance_score: How relevant to the query [0-1]
        source: Where identified from (query, norm, doctrine)
    """

    principle_id: str
    name: str
    definition: str
    articles: List[str] = field(default_factory=list)
    relevance_score: float = 0.5
    source: str = "inference"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "principle_id": self.principle_id,
            "name": self.name,
            "definition": self.definition,
            "articles": self.articles,
            "relevance_score": round(self.relevance_score, 3),
            "source": self.source,
        }


class PrinciplesExpert(BaseExpert):
    """
    Expert for teleological interpretation (Art. 12, comma II disp. prel. c.c.).

    Epistemology: Ratio legis and fundamental principles
    Focus: WHY the norm exists and what values it protects

    Output:
    - "Interpretazione Teleologica" section header
    - Identified principles with definitions
    - Ratio legis explanation
    - Doctrinal support
    - Confidence score

    Example:
        >>> retriever = BridgeTableRetriever(...)
        >>> llm = LLMService(...)
        >>> expert = PrinciplesExpert(retriever=retriever, llm_service=llm)
        >>> context = ExpertContext(query_text="Qual è lo scopo dell'art. 1453 c.c.?")
        >>> response = await expert.analyze(context)
        >>> print(response.section_header)
        "Interpretazione Teleologica"
    """

    expert_type = "principles"
    section_header = "Interpretazione Teleologica"
    description = "Interpretazione teleologica e per principi (art. 12, II disp. prel. c.c.)"

    def __init__(
        self,
        retriever: Optional[ChunkRetriever] = None,
        llm_service: Optional[LLMService] = None,
        config: Optional[PrinciplesConfig] = None,
    ):
        """
        Initialize PrinciplesExpert.

        Args:
            retriever: Chunk retriever for Bridge Table access
            llm_service: LLM service for analysis
            config: PrinciplesExpert configuration
        """
        self._config = config or PrinciplesConfig()
        super().__init__(
            retriever=retriever,
            llm_service=llm_service,
            config=self._config,
        )

    async def analyze(self, context: ExpertContext) -> ExpertResponse:
        """
        Analyze query with teleological interpretation approach.

        Flow:
        1. Identify relevant legal principles from query/context
        2. Retrieve doctrine chunks with Principles affinity
        3. Retrieve main norm context (if not already provided)
        4. Build LLM prompt with principles and doctrine
        5. Generate interpretation
        6. Compute confidence based on doctrine availability

        Args:
            context: Input context with query and entities

        Returns:
            ExpertResponse with teleological interpretation
        """
        start_time = time.time()

        log.info(
            "principles_expert_analyzing",
            query=context.query_text[:50],
            trace_id=context.trace_id,
            has_concepts=bool(context.legal_concepts),
        )

        # Step 1: Identify relevant legal principles
        principles = self._identify_principles(context)

        # Step 2: Retrieve doctrine chunks
        doctrine_chunks = await self._retrieve_doctrine(context)

        # Step 3: Retrieve main norm context
        main_norm_chunks = await self._retrieve_main_norm(context)

        # Step 4: Check if we have enough context
        if not doctrine_chunks and not principles:
            execution_time = (time.time() - start_time) * 1000
            return self._create_no_doctrine_response(
                context=context,
                main_norm_chunks=main_norm_chunks,
                execution_time_ms=execution_time,
            )

        # Step 5: Build legal sources
        legal_sources = self._build_legal_sources(
            main_norm_chunks, doctrine_chunks, principles
        )

        # Step 6: Generate interpretation
        if self.llm_service:
            interpretation, reasoning_steps, tokens = await self._generate_interpretation(
                context, main_norm_chunks, doctrine_chunks, principles
            )
        else:
            interpretation, reasoning_steps, tokens = self._generate_fallback_interpretation(
                context, main_norm_chunks, doctrine_chunks, principles
            )

        # Step 7: Compute confidence
        confidence, factors = self._compute_confidence(
            doctrine_chunks=doctrine_chunks,
            principles=principles,
            context=context,
        )

        execution_time = (time.time() - start_time) * 1000

        # Create F5 feedback hook
        feedback_hook = None
        if self._config.enable_f5_feedback:
            feedback_hook = FeedbackHook(
                feedback_type="F5",
                expert_type=self.expert_type,
                response_id=context.trace_id,
                enabled=True,
                correction_options={
                    # Ratio legis identification quality
                    "ratio_legis_quality": [
                        "excellent",       # Accurately identified ratio legis
                        "good",            # Reasonably identified
                        "partial",         # Partially correct
                        "incorrect",       # Wrong ratio legis
                    ],
                    # Principle relevance
                    "principle_relevance": [
                        "all_relevant",    # All principles apply
                        "mostly_relevant", # Most principles apply
                        "some_irrelevant", # Some don't apply
                        "mostly_irrelevant",
                    ],
                    # Constitutional grounding (if applicable)
                    "constitutional_grounding": [
                        "well_grounded",   # Solid constitutional basis
                        "reasonable",      # Reasonable connection
                        "weak",            # Weak connection
                        "not_applicable",  # No constitutional aspect
                    ],
                    # Doctrinal support
                    "doctrinal_support": [
                        "strong_consensus",    # Doctrine agrees
                        "majority_view",       # Most doctrine agrees
                        "divided",             # Doctrine is split
                        "minority_view",       # Goes against doctrine
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
                    "principles_count": len(principles),
                    "doctrine_chunks_count": len(doctrine_chunks),
                    "confidence": confidence,
                    "interpretation_preview": interpretation[:300] if interpretation else "",
                },
            )

        # Check for doctrinal disagreement
        has_disagreement = self._detect_doctrinal_disagreement(doctrine_chunks)

        response = ExpertResponse(
            expert_type=self.expert_type,
            section_header=self.section_header,
            interpretation=interpretation,
            legal_basis=legal_sources,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            confidence_factors=factors,
            limitations=self._identify_limitations(doctrine_chunks, principles, has_disagreement),
            suggestions=self._generate_suggestions(confidence, context),
            trace_id=context.trace_id,
            execution_time_ms=execution_time,
            tokens_used=tokens,
            feedback_hook=feedback_hook,
            metadata={
                "principles_identified": len(principles),
                "doctrine_chunks": len(doctrine_chunks),
                "has_doctrinal_disagreement": has_disagreement,
                "identified_principles": [p.to_dict() for p in principles],
            },
        )

        log.info(
            "principles_expert_completed",
            trace_id=context.trace_id,
            confidence=confidence,
            principles_count=len(principles),
            doctrine_count=len(doctrine_chunks),
            execution_time_ms=execution_time,
        )

        return response

    def _identify_principles(self, context: ExpertContext) -> List[IdentifiedPrinciple]:
        """
        Identify relevant legal principles from query and context.

        Uses keyword matching against known principles taxonomy.
        Includes constitutional principles if enabled in config.

        Args:
            context: Input context with query and entities

        Returns:
            List of identified principles sorted by relevance
        """
        principles: List[IdentifiedPrinciple] = []
        query_lower = context.query_text.lower()
        concepts = [c.lower() for c in context.legal_concepts]

        # Build combined principles dict based on config
        all_principles: Dict[str, Dict[str, Any]] = dict(LEGAL_PRINCIPLES)
        if self._config.include_constitutional:
            all_principles.update(CONSTITUTIONAL_PRINCIPLES)

        # Check each known principle
        for principle_id, principle_data in all_principles.items():
            # Check query text
            name_lower = principle_data["name"].lower()
            if name_lower in query_lower:
                principles.append(
                    IdentifiedPrinciple(
                        principle_id=principle_id,
                        name=principle_data["name"],
                        definition=principle_data["definition"],
                        articles=principle_data["articles"],
                        relevance_score=0.9,
                        source="query",
                    )
                )
                continue

            # Check legal concepts from NER
            if any(name_lower in concept or concept in name_lower for concept in concepts):
                principles.append(
                    IdentifiedPrinciple(
                        principle_id=principle_id,
                        name=principle_data["name"],
                        definition=principle_data["definition"],
                        articles=principle_data["articles"],
                        relevance_score=0.8,
                        source="ner_concept",
                    )
                )
                continue

            # Check for related keywords
            keywords = self._get_principle_keywords(principle_id)
            if any(kw in query_lower for kw in keywords):
                principles.append(
                    IdentifiedPrinciple(
                        principle_id=principle_id,
                        name=principle_data["name"],
                        definition=principle_data["definition"],
                        articles=principle_data["articles"],
                        relevance_score=0.6,
                        source="keyword_match",
                    )
                )

        # Sort by relevance
        principles.sort(key=lambda p: p.relevance_score, reverse=True)

        log.debug(
            "principles_identified",
            trace_id=context.trace_id,
            count=len(principles),
            principles=[p.name for p in principles],
        )

        return principles

    def _get_principle_keywords(self, principle_id: str) -> List[str]:
        """
        Get related keywords for a principle.

        Args:
            principle_id: The principle identifier

        Returns:
            List of keywords that trigger this principle
        """
        keyword_map = {
            # Civil law principles
            "buona_fede": ["correttezza", "lealtà", "onestà", "fair dealing"],
            "tutela_affidamento": ["affidamento", "aspettativa", "apparenza"],
            "favor_debitoris": ["debitore", "dubbio interpretativo", "1371"],
            "conservazione_contratto": ["conservazione", "nullità parziale", "1367"],
            "autonomia_privata": ["libertà contrattuale", "autonomia", "1322"],
            "equilibrio_contrattuale": ["equilibrio", "squilibrio", "lesione", "1448"],
            "diligenza": ["diligenza", "negligenza", "colpa", "1176"],
            "neminem_laedere": ["danno", "responsabilità", "illecito", "2043"],
            # Constitutional principles
            "uguaglianza": ["uguaglianza", "parità", "discriminazione", "art. 3"],
            "solidarieta": ["solidarietà", "sociale", "art. 2"],
            "libera_iniziativa": ["iniziativa economica", "impresa", "art. 41"],
            "tutela_risparmio": ["risparmio", "credito", "art. 47"],
            "proprieta_privata": ["proprietà", "espropriazione", "art. 42"],
        }

        keywords = keyword_map.get(principle_id)
        if keywords is None:
            log.warning(
                "principle_keywords_missing",
                principle_id=principle_id,
                message="No keywords defined for principle - add to keyword_map",
            )
            return []
        return keywords

    async def _retrieve_doctrine(
        self, context: ExpertContext
    ) -> List[Dict[str, Any]]:
        """Retrieve doctrine chunks from Bridge Table."""
        if not self.retriever:
            return []

        filters = {
            "expert_affinity": "principles",
            "source_type": "doctrine",
        }

        try:
            chunks = await self.retriever.retrieve(
                query=context.query_text,
                query_embedding=context.query_embedding,
                filters=filters,
                limit=self._config.max_doctrine_chunks,
            )

            # Filter by score
            return [
                c for c in (chunks or [])
                if c.get("score", 0) >= self._config.min_doctrine_score
            ]

        except Exception as e:
            log.warning(
                "principles_doctrine_retrieval_error",
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

        filters = {
            "source_type": "norm",
        }

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
                "principles_norm_retrieval_error",
                error=str(e),
                trace_id=context.trace_id,
            )
            return []

    def _build_legal_sources(
        self,
        main_norm_chunks: List[Dict[str, Any]],
        doctrine_chunks: List[Dict[str, Any]],
        principles: List[IdentifiedPrinciple],
    ) -> List[LegalSource]:
        """Build LegalSource list from all sources."""
        sources: List[LegalSource] = []

        # Main norms
        for chunk in main_norm_chunks[:3]:
            sources.append(
                LegalSource(
                    source_type="norm",
                    source_id=chunk.get("urn", chunk.get("id", "")),
                    citation=chunk.get("citation", chunk.get("title", "")),
                    excerpt=chunk.get("text", "")[:400],
                    relevance="Norma di riferimento",
                    relevance_score=0.8,  # Main norm reference
                )
            )

        # Doctrine sources
        for i, chunk in enumerate(doctrine_chunks):
            # Decay score by position (earlier = more relevant)
            position_score = max(0.3, 1.0 - (i * 0.15))
            sources.append(
                LegalSource(
                    source_type="doctrine",
                    source_id=chunk.get("id", ""),
                    citation=chunk.get("citation", chunk.get("author", "Dottrina")),
                    excerpt=chunk.get("text", "")[:300],
                    relevance="Supporto dottrinale",
                    relevance_score=round(position_score, 3),
                )
            )

        # Principles as sources
        for principle in principles:
            sources.append(
                LegalSource(
                    source_type="principle",
                    source_id=principle.principle_id,
                    citation=principle.name,
                    excerpt=principle.definition,
                    relevance=f"Principio giuridico (score: {principle.relevance_score:.2f})",
                    relevance_score=round(min(1.0, principle.relevance_score), 3),
                )
            )

        return sources

    async def _generate_interpretation(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        doctrine_chunks: List[Dict[str, Any]],
        principles: List[IdentifiedPrinciple],
    ) -> Tuple[str, List[ReasoningStep], int]:
        """Generate interpretation using LLM."""
        # Format sections
        main_norm_text = self._format_chunks_for_prompt(main_norm_chunks)
        doctrine_text = self._format_doctrine_for_prompt(doctrine_chunks)
        principles_text = self._format_principles_for_prompt(principles)

        prompt = PRINCIPLES_PROMPT_TEMPLATE.format(
            query=context.query_text,
            main_norm=main_norm_text or "Non specificata",
            principles=principles_text or "Nessun principio specifico identificato",
            doctrine=doctrine_text or "Nessun commento dottrinale disponibile",
        )

        try:
            interpretation = await self.llm_service.generate(
                prompt=prompt,
                temperature=self._config.principles_temperature,
                max_tokens=self._config.max_response_tokens,
            )
            # Approximate token count (word-based, not actual LLM tokens)
            # TODO: Use proper tokenizer for accurate count when LLM provider supports it
            tokens_used = len(prompt.split()) + len(interpretation.split())
        except Exception as e:
            log.exception(
                "principles_llm_error",
                error=str(e),
                trace_id=context.trace_id,
                exc_info=True,
            )
            interpretation, _, tokens_used = self._generate_fallback_interpretation(
                context, main_norm_chunks, doctrine_chunks, principles
            )

        reasoning_steps = self._build_reasoning_steps(
            main_norm_chunks, doctrine_chunks, principles
        )

        return interpretation, reasoning_steps, tokens_used

    def _generate_fallback_interpretation(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        doctrine_chunks: List[Dict[str, Any]],
        principles: List[IdentifiedPrinciple],
    ) -> Tuple[str, List[ReasoningStep], int]:
        """Generate interpretation without LLM."""
        parts: List[str] = []

        if principles:
            parts.append("**Principi giuridici applicabili:**\n")
            for p in principles[:5]:
                parts.append(f"- **{p.name}**: {p.definition}\n")
                if p.articles:
                    parts.append(f"  *Riferimenti:* {', '.join(p.articles)}\n")

        if doctrine_chunks:
            parts.append("\n**Supporto dottrinale:**\n")
            for chunk in doctrine_chunks[:3]:
                citation = chunk.get("citation", chunk.get("author", "Dottrina"))
                text = chunk.get("text", "")[:200]
                parts.append(f"- *{citation}*: \"{text}...\"\n")

        if main_norm_chunks:
            parts.append("\n**Norma di riferimento:**\n")
            for chunk in main_norm_chunks[:1]:
                citation = chunk.get("citation", "Norma")
                parts.append(f"- {citation}\n")

        if not parts:
            parts.append(
                "Non sono stati identificati principi o dottrina specifici per questa query. "
                "L'interpretazione teleologica richiede un'analisi più approfondita."
            )

        interpretation = "\n".join(parts)
        reasoning_steps = self._build_reasoning_steps(
            main_norm_chunks, doctrine_chunks, principles
        )

        return interpretation, reasoning_steps, 0

    def _format_chunks_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format norm chunks for inclusion in LLM prompt.

        Args:
            chunks: List of chunk dictionaries with citation, text, etc.

        Returns:
            Formatted string with citations and text excerpts (max 3 chunks)
        """
        if not chunks:
            return ""

        formatted: List[str] = []
        for chunk in chunks[:3]:
            citation = chunk.get("citation", chunk.get("title", "Fonte"))
            text = chunk.get("text", "")[:400]
            formatted.append(f"[{citation}]\n{text}\n")

        return "\n".join(formatted)

    def _format_doctrine_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format doctrine chunks for inclusion in LLM prompt.

        Args:
            chunks: List of doctrine chunk dictionaries with author, text, etc.

        Returns:
            Formatted string with author citations and excerpts (max 5 chunks)
        """
        if not chunks:
            return ""

        formatted: List[str] = []
        for chunk in chunks[:5]:
            author = chunk.get("author", chunk.get("citation", "Autore non specificato"))
            text = chunk.get("text", "")[:400]
            formatted.append(f"[{author}]\n{text}\n")

        return "\n".join(formatted)

    def _format_principles_for_prompt(self, principles: List[IdentifiedPrinciple]) -> str:
        """
        Format identified principles for inclusion in LLM prompt.

        Args:
            principles: List of identified legal principles

        Returns:
            Formatted string with principle names, definitions, and article refs (max 5)
        """
        if not principles:
            return ""

        formatted: List[str] = []
        for p in principles[:5]:
            articles = ", ".join(p.articles) if p.articles else "N/A"
            formatted.append(f"- {p.name}: {p.definition} (Rif.: {articles})")

        return "\n".join(formatted)

    def _build_reasoning_steps(
        self,
        main_norm_chunks: List[Dict[str, Any]],
        doctrine_chunks: List[Dict[str, Any]],
        principles: List[IdentifiedPrinciple],
    ) -> List[ReasoningStep]:
        """Build reasoning chain steps."""
        steps: List[ReasoningStep] = []
        step_num = 1

        if principles:
            principle_names = [p.name for p in principles]
            steps.append(
                self._build_reasoning_step(
                    step_number=step_num,
                    description=f"Identificati {len(principles)} principi giuridici: {', '.join(principle_names[:3])}",
                    source_ids=[p.principle_id for p in principles],
                )
            )
            step_num += 1

        if doctrine_chunks:
            doctrine_ids = [c.get("id", "") for c in doctrine_chunks]
            steps.append(
                self._build_reasoning_step(
                    step_number=step_num,
                    description=f"Recuperati {len(doctrine_chunks)} commenti dottrinali",
                    source_ids=doctrine_ids,
                )
            )
            step_num += 1

        if main_norm_chunks:
            norm_ids = [c.get("urn", c.get("id", "")) for c in main_norm_chunks]
            steps.append(
                self._build_reasoning_step(
                    step_number=step_num,
                    description=f"Analizzata norma di riferimento ({len(main_norm_chunks)} chunks)",
                    source_ids=norm_ids,
                )
            )
            step_num += 1

        steps.append(
            self._build_reasoning_step(
                step_number=step_num,
                description="Sintesi teleologica secondo art. 12, II disp. prel. c.c. (ratio legis)",
            )
        )

        return steps

    def _detect_doctrinal_disagreement(
        self, doctrine_chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Detect if doctrine sources show disagreement.

        Simple heuristic: check for contrasting indicators in text.
        """
        if len(doctrine_chunks) < 2:
            return False

        disagreement_indicators = [
            "tuttavia",
            "contra",
            "diversamente",
            "in senso opposto",
            "critica",
            "non condivide",
            "contraria",
        ]

        for chunk in doctrine_chunks:
            text_lower = chunk.get("text", "").lower()
            if any(ind in text_lower for ind in disagreement_indicators):
                return True

        return False

    def _compute_confidence(
        self,
        doctrine_chunks: List[Dict[str, Any]],
        principles: List[IdentifiedPrinciple],
        context: ExpertContext,
    ) -> Tuple[float, ConfidenceFactors]:
        """
        Compute confidence score based on available sources.

        Args:
            doctrine_chunks: Retrieved doctrine chunks
            principles: Identified legal principles
            context: Input context

        Returns:
            Tuple of (confidence_score, confidence_factors)
        """
        # Source availability - use min_principles from config for threshold
        doctrine_score = min(1.0, len(doctrine_chunks) / self._config.max_doctrine_chunks)
        # Good coverage = at least min_principles * 2 (e.g., 2 principles if min is 1)
        principles_threshold = max(2, self._config.min_principles * 2)
        principles_score = min(1.0, len(principles) / principles_threshold)
        source_availability = (doctrine_score + principles_score) / 2

        # Norm clarity - based on principles identification
        if principles:
            avg_relevance = sum(p.relevance_score for p in principles) / len(principles)
            norm_clarity = avg_relevance
        else:
            norm_clarity = 0.3

        # Definition coverage - how well we cover the teleological aspect
        if doctrine_chunks and principles:
            definition_coverage = 0.8
        elif doctrine_chunks or principles:
            definition_coverage = 0.5
        else:
            definition_coverage = 0.2

        # Contextual ambiguity
        if len(principles) >= 2 and self._detect_doctrinal_disagreement(doctrine_chunks):
            contextual_ambiguity = 0.6  # Multiple views = more ambiguity
        elif principles:
            contextual_ambiguity = 0.3
        else:
            contextual_ambiguity = 0.7

        factors = ConfidenceFactors(
            norm_clarity=norm_clarity,
            source_availability=source_availability,
            contextual_ambiguity=contextual_ambiguity,
            definition_coverage=definition_coverage,
        )

        confidence = factors.compute_overall()

        return confidence, factors

    def _create_no_doctrine_response(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        execution_time_ms: float,
    ) -> ExpertResponse:
        """Create response when no doctrine or principles are found."""
        interpretation = (
            "Non sono stati identificati principi giuridici specifici o commenti dottrinali "
            "per fornire un'interpretazione teleologica affidabile. "
            "L'analisi della ratio legis richiede fonti dottrinali o principi applicabili."
        )

        if main_norm_chunks:
            interpretation += "\n\n**Norma analizzata:**\n"
            for chunk in main_norm_chunks[:1]:
                citation = chunk.get("citation", "")
                interpretation += f"- {citation}\n"

        feedback_hook = None
        if self._config.enable_f5_feedback:
            feedback_hook = FeedbackHook(
                feedback_type="F5",
                expert_type=self.expert_type,
                response_id=context.trace_id,
                enabled=True,
                correction_options={
                    # No doctrine assessment
                    "doctrine_availability": [
                        "correctly_unavailable",  # Doctrine really doesn't exist
                        "doctrine_exists",        # Doctrine exists but wasn't found
                        "search_failed",          # Technical failure
                    ],
                    # Should this expert have been skipped?
                    "expert_applicability": [
                        "correctly_limited",      # Expert correctly limited
                        "should_have_found",      # Should have found something
                        "query_not_applicable",   # Query doesn't need this expert
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
                    "no_doctrine": True,
                    "main_norm_count": len(main_norm_chunks),
                    "confidence": self._config.no_doctrine_confidence,
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
                    relevance_score=0.5,  # Limited relevance (no doctrine found)
                )
                for c in main_norm_chunks[:1]
            ],
            confidence=self._config.no_doctrine_confidence,
            confidence_factors=ConfidenceFactors(
                norm_clarity=0.4,
                source_availability=0.2,
                contextual_ambiguity=0.6,
                definition_coverage=0.2,
            ),
            limitations="Dottrina e principi specifici non disponibili per questa query",
            suggestions="Specificare il principio giuridico di interesse o consultare fonti dottrinali",
            trace_id=context.trace_id,
            execution_time_ms=execution_time_ms,
            feedback_hook=feedback_hook,
            metadata={
                "no_doctrine": True,
                "principles_identified": 0,
            },
        )

    def _identify_limitations(
        self,
        doctrine_chunks: List[Dict[str, Any]],
        principles: List[IdentifiedPrinciple],
        has_disagreement: bool,
    ) -> str:
        """Identify limitations of the analysis."""
        limitations: List[str] = []

        if not doctrine_chunks:
            limitations.append("Nessun commento dottrinale recuperato")

        if not principles:
            limitations.append("Nessun principio giuridico specifico identificato")

        if has_disagreement:
            limitations.append("Posizioni dottrinali discordanti rilevate")

        if not self.llm_service:
            limitations.append("Analisi eseguita senza supporto LLM")

        return "; ".join(limitations) if limitations else ""

    def _generate_suggestions(self, confidence: float, context: ExpertContext) -> str:
        """Generate suggestions based on confidence level."""
        if confidence >= self._config.high_confidence_threshold:
            return ""

        suggestions: List[str] = []

        if not context.legal_concepts:
            suggestions.append("Specificare i concetti giuridici rilevanti per migliorare l'identificazione dei principi")

        if confidence < 0.4:
            suggestions.append("Consultare direttamente la dottrina di riferimento per la materia")

        return "; ".join(suggestions)
