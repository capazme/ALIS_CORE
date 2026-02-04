"""
Literal Expert for MERL-T Analysis Pipeline.

Implements literal/textual interpretation following Art. 12, comma I, disp. prel. c.c.:
"Nell'applicare la legge non si può ad essa attribuire altro senso
che quello fatto palese dal significato proprio delle parole..."

The literal interpretation is the primary hermeneutic canon:
- Focus on the EXACT TEXT of the norm
- "Significato proprio delle parole" = technical-legal meaning if exists,
  otherwise common meaning
- Principle: "in claris non fit interpretatio"

Approach:
1. Retrieve exact norm text via Bridge Table (expert_affinity[Literal])
2. Identify technical terms and legal definitions
3. Analyze syntactic structure
4. Follow internal references (normative cross-references)
5. Produce interpretation based on textual data
"""

import time
import structlog
from dataclasses import dataclass
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


# Default prompt for literal interpretation
LITERAL_PROMPT_TEMPLATE = """Sei un esperto di interpretazione letterale del diritto italiano.
Il tuo compito è analizzare la seguente domanda giuridica basandoti ESCLUSIVAMENTE sul testo delle norme fornite.

QUERY: {query}

NORME RILEVANTI:
{norm_texts}

DEFINIZIONI LEGALI:
{definitions}

ISTRUZIONI:
1. Analizza il significato letterale delle parole usate nelle norme
2. Identifica eventuali termini tecnici con significato giuridico specifico
3. Considera la struttura sintattica e la punteggiatura
4. Se il testo è chiaro, applica il principio "in claris non fit interpretatio"
5. Cita sempre gli articoli specifici che supportano la tua interpretazione

FORMATO OUTPUT:
- Inizia con una sintesi della risposta
- Elenca i punti normativi rilevanti con citazioni esatte
- Fornisci una spiegazione in linguaggio accessibile
- Indica eventuali ambiguità testuali

Rispondi in italiano."""


@dataclass
class LiteralConfig(ExpertConfig):
    """Configuration specific to LiteralExpert."""

    # Chunk retrieval
    include_definitions: bool = True
    definition_limit: int = 5

    # LLM
    literal_temperature: float = 0.2  # Lower for more deterministic
    max_response_tokens: int = 1500

    # Confidence thresholds
    min_norm_chunks: int = 1
    high_confidence_threshold: float = 0.7

    # F3 Feedback
    enable_f3_feedback: bool = True  # Enable F3 feedback hooks for RLCF


class LiteralExpert(BaseExpert):
    """
    Expert for literal interpretation (Art. 12, I disp. prel. c.c.).

    Epistemology: Legal positivism
    Focus: What the law SAYS (text-based interpretation)

    Output:
    - "Interpretazione Letterale" section header
    - Relevant norm text with URN links
    - Plain-language explanation
    - Confidence score (0.0-1.0)
    - Processing time

    Example:
        >>> retriever = BridgeTableRetriever(...)
        >>> llm = LLMService(...)
        >>> expert = LiteralExpert(retriever=retriever, llm_service=llm)
        >>> context = ExpertContext(query_text="Cos'è la risoluzione del contratto?")
        >>> response = await expert.analyze(context)
        >>> print(response.section_header)
        "Interpretazione Letterale"
    """

    expert_type = "literal"
    section_header = "Interpretazione Letterale"
    description = "Interpretazione letterale (art. 12, I disp. prel. c.c.)"

    def __init__(
        self,
        retriever: Optional[ChunkRetriever] = None,
        llm_service: Optional[LLMService] = None,
        config: Optional[LiteralConfig] = None,
    ):
        """
        Initialize LiteralExpert.

        Args:
            retriever: Chunk retriever for Bridge Table access
            llm_service: LLM service for analysis
            config: LiteralExpert configuration
        """
        self._config = config or LiteralConfig()
        super().__init__(
            retriever=retriever,
            llm_service=llm_service,
            config=self._config,
        )

    async def analyze(self, context: ExpertContext) -> ExpertResponse:
        """
        Analyze query with literal interpretation approach.

        Flow:
        1. Retrieve norm text chunks via Bridge Table (expert_affinity[Literal])
        2. Retrieve legal definitions if needed
        3. Build LLM prompt with norm texts
        4. Generate interpretation
        5. Compute confidence based on source quality

        Args:
            context: Input context with query and entities

        Returns:
            ExpertResponse with literal interpretation
        """
        start_time = time.time()

        log.info(
            "literal_expert_analyzing",
            query=context.query_text[:50],
            trace_id=context.trace_id,
            has_norm_refs=bool(context.norm_references),
            has_concepts=bool(context.legal_concepts),
        )

        # Step 1: Retrieve relevant chunks
        norm_chunks, definition_chunks = await self._retrieve_chunks(context)

        # Step 2: Check if we have enough data
        if not norm_chunks and not context.retrieved_chunks:
            execution_time = (time.time() - start_time) * 1000
            return self._create_low_confidence_response(
                context=context,
                reason="Nessuna norma rilevante trovata nel database.",
                suggestion="Specificare l'articolo o la legge di riferimento (es. 'art. 1453 c.c.').",
                execution_time_ms=execution_time,
            )

        # Use pre-retrieved chunks if available
        all_norm_chunks = norm_chunks or context.retrieved_chunks

        # Step 3: Build legal sources
        legal_sources = self._build_legal_sources(all_norm_chunks, definition_chunks)

        # Step 4: Generate interpretation
        if self.llm_service:
            interpretation, reasoning_steps, tokens = await self._generate_interpretation(
                context, all_norm_chunks, definition_chunks
            )
        else:
            interpretation, reasoning_steps, tokens = self._generate_fallback_interpretation(
                context, all_norm_chunks, definition_chunks
            )

        # Step 5: Compute confidence
        confidence, factors = self._compute_confidence(
            norm_chunks=all_norm_chunks,
            definition_chunks=definition_chunks,
            context=context,
        )

        execution_time = (time.time() - start_time) * 1000

        # Create F3 feedback hook for RLCF (LiteralExpert = F3)
        feedback_hook = None
        if self._config.enable_f3_feedback:
            feedback_hook = FeedbackHook(
                feedback_type="F3",
                expert_type=self.expert_type,
                response_id=context.trace_id,
                enabled=True,
                correction_options={
                    # Interpretation quality assessment
                    "interpretation_quality": [
                        "excellent",      # Accurate, complete, well-structured
                        "good",           # Mostly accurate, minor issues
                        "fair",           # Some inaccuracies or gaps
                        "poor",           # Significant errors or omissions
                    ],
                    # Source relevance assessment
                    "source_relevance": [
                        "all_relevant",   # All cited sources are pertinent
                        "mostly_relevant", # Most sources are pertinent
                        "some_irrelevant", # Some sources don't apply
                        "mostly_irrelevant", # Most sources don't apply
                    ],
                    # Confidence calibration feedback
                    "confidence_calibration": [
                        "well_calibrated", # Confidence matches actual quality
                        "overconfident",   # Confidence too high for quality
                        "underconfident",  # Confidence too low for quality
                    ],
                    # Textual interpretation accuracy (Art. 12 disp. prel.)
                    "textual_accuracy": [
                        "faithful",        # Faithful to normative text
                        "reasonable",      # Reasonable interpretation
                        "stretched",       # Interpretation stretches the text
                        "incorrect",       # Misinterprets the text
                    ],
                    # Missing elements
                    "missing_elements": [
                        "none",            # Nothing important missing
                        "minor_details",   # Some minor details missing
                        "key_articles",    # Important articles missing
                        "fundamental",     # Fundamental concepts missing
                    ],
                },
                context_snapshot={
                    "query": context.query_text[:200],
                    "sources_count": len(legal_sources),
                    "source_urns": [s.source_id for s in legal_sources[:5]],
                    "confidence": confidence,
                    "confidence_factors": factors.to_dict(),
                    "interpretation_preview": interpretation[:300] if interpretation else "",
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
            limitations=self._identify_limitations(all_norm_chunks, context),
            suggestions=self._generate_suggestions(confidence, context),
            trace_id=context.trace_id,
            execution_time_ms=execution_time,
            tokens_used=tokens,
            feedback_hook=feedback_hook,
            metadata={
                "norm_chunks_count": len(all_norm_chunks),
                "definition_chunks_count": len(definition_chunks),
            },
        )

        log.info(
            "literal_expert_completed",
            trace_id=context.trace_id,
            confidence=confidence,
            sources_count=len(legal_sources),
            execution_time_ms=execution_time,
        )

        return response

    async def _retrieve_chunks(
        self, context: ExpertContext
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve norm and definition chunks from Bridge Table.

        Args:
            context: Input context

        Returns:
            Tuple of (norm_chunks, definition_chunks)
        """
        if not self.retriever:
            log.warning("literal_expert_no_retriever", trace_id=context.trace_id)
            return [], []

        # Build filters for literal interpretation
        norm_filters = {
            "expert_affinity": "literal",
            "source_type": "norm",
        }

        # Add URN filters if we have norm references
        if context.norm_references:
            norm_filters["urns"] = context.norm_references

        # Retrieve norm chunks
        norm_chunks = await self.retriever.retrieve(
            query=context.query_text,
            query_embedding=context.query_embedding,
            filters=norm_filters,
            limit=self._config.chunk_limit,
        )

        # Retrieve definition chunks if enabled
        definition_chunks: List[Dict[str, Any]] = []
        if self._config.include_definitions:
            definition_filters = {
                "expert_affinity": "literal",
                "source_type": "definition",
            }

            # Add concept filters if we have legal concepts
            if context.legal_concepts:
                definition_filters["concepts"] = context.legal_concepts

            definition_chunks = await self.retriever.retrieve(
                query=context.query_text,
                query_embedding=context.query_embedding,
                filters=definition_filters,
                limit=self._config.definition_limit,
            )

        log.debug(
            "literal_chunks_retrieved",
            trace_id=context.trace_id,
            norm_count=len(norm_chunks),
            definition_count=len(definition_chunks),
        )

        return norm_chunks, definition_chunks

    def _build_legal_sources(
        self,
        norm_chunks: List[Dict[str, Any]],
        definition_chunks: List[Dict[str, Any]],
    ) -> List[LegalSource]:
        """Build LegalSource list from retrieved chunks."""
        sources: List[LegalSource] = []

        # Compute max score for normalization
        max_score = max(
            (chunk.get("score", 0) for chunk in norm_chunks),
            default=1.0,
        ) or 1.0

        for chunk in norm_chunks:
            raw_score = chunk.get("score", 0)
            normalized = min(1.0, raw_score / max_score) if max_score > 0 else 0.0
            sources.append(
                LegalSource(
                    source_type="norm",
                    source_id=chunk.get("urn", chunk.get("id", "")),
                    citation=chunk.get("citation", chunk.get("title", "")),
                    excerpt=chunk.get("text", "")[:500],
                    relevance=f"Score: {raw_score:.2f}",
                    relevance_score=round(normalized, 3),
                )
            )

        for chunk in definition_chunks:
            raw_score = chunk.get("score", 0)
            normalized = min(1.0, raw_score / max_score) if max_score > 0 else 0.5
            sources.append(
                LegalSource(
                    source_type="definition",
                    source_id=chunk.get("urn", chunk.get("id", "")),
                    citation=chunk.get("citation", chunk.get("title", "")),
                    excerpt=chunk.get("text", "")[:300],
                    relevance="Definizione legale",
                    relevance_score=round(normalized, 3),
                )
            )

        return sources

    async def _generate_interpretation(
        self,
        context: ExpertContext,
        norm_chunks: List[Dict[str, Any]],
        definition_chunks: List[Dict[str, Any]],
    ) -> Tuple[str, List[ReasoningStep], int]:
        """
        Generate interpretation using LLM.

        Args:
            context: Input context
            norm_chunks: Retrieved norm chunks
            definition_chunks: Retrieved definition chunks

        Returns:
            Tuple of (interpretation, reasoning_steps, tokens_used)
        """
        # Format norm texts
        norm_texts = self._format_chunks_for_prompt(norm_chunks)
        definitions = self._format_chunks_for_prompt(definition_chunks)

        # Build prompt
        prompt = LITERAL_PROMPT_TEMPLATE.format(
            query=context.query_text,
            norm_texts=norm_texts or "Nessuna norma specifica trovata.",
            definitions=definitions or "Nessuna definizione specifica trovata.",
        )

        # Generate with LLM
        try:
            interpretation = await self.llm_service.generate(
                prompt=prompt,
                temperature=self._config.literal_temperature,
                max_tokens=self._config.max_response_tokens,
            )
            tokens_used = len(prompt.split()) + len(interpretation.split())  # Approximation
        except Exception as e:
            log.exception(
                "literal_llm_error",
                error=str(e),
                trace_id=context.trace_id,
                exc_info=True,
            )
            interpretation, _, tokens_used = self._generate_fallback_interpretation(
                context, norm_chunks, definition_chunks
            )

        # Build reasoning steps
        reasoning_steps = self._build_reasoning_steps(norm_chunks, definition_chunks)

        return interpretation, reasoning_steps, tokens_used

    def _generate_fallback_interpretation(
        self,
        context: ExpertContext,
        norm_chunks: List[Dict[str, Any]],
        definition_chunks: List[Dict[str, Any]],
    ) -> Tuple[str, List[ReasoningStep], int]:
        """
        Generate interpretation without LLM (fallback mode).

        Returns:
            Tuple of (interpretation, reasoning_steps, tokens_used)
        """
        parts: List[str] = []

        if norm_chunks:
            parts.append("**Testo normativo rilevante:**\n")
            for i, chunk in enumerate(norm_chunks[:3], 1):
                citation = chunk.get("citation", "Norma")
                text = chunk.get("text", "")[:400]
                parts.append(f"{i}. {citation}:\n\"{text}...\"\n")

        if definition_chunks:
            parts.append("\n**Definizioni legali:**\n")
            for chunk in definition_chunks[:2]:
                citation = chunk.get("citation", "Definizione")
                text = chunk.get("text", "")[:200]
                parts.append(f"- {citation}: {text}\n")

        if not parts:
            parts.append("Non sono state trovate norme specifiche per rispondere alla domanda.")

        interpretation = "\n".join(parts)
        reasoning_steps = self._build_reasoning_steps(norm_chunks, definition_chunks)

        return interpretation, reasoning_steps, 0

    def _format_chunks_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for inclusion in prompt."""
        if not chunks:
            return ""

        formatted: List[str] = []
        for chunk in chunks:
            citation = chunk.get("citation", chunk.get("title", "Fonte"))
            text = chunk.get("text", "")
            urn = chunk.get("urn", "")

            if urn:
                formatted.append(f"[{citation}] (URN: {urn})\n{text}\n")
            else:
                formatted.append(f"[{citation}]\n{text}\n")

        return "\n".join(formatted)

    def _build_reasoning_steps(
        self,
        norm_chunks: List[Dict[str, Any]],
        definition_chunks: List[Dict[str, Any]],
    ) -> List[ReasoningStep]:
        """Build reasoning chain steps."""
        steps: List[ReasoningStep] = []

        if norm_chunks:
            norm_urns = [c.get("urn", c.get("id", "")) for c in norm_chunks]
            steps.append(
                self._build_reasoning_step(
                    step_number=1,
                    description=f"Recuperati {len(norm_chunks)} testi normativi rilevanti via Bridge Table",
                    source_ids=norm_urns,
                )
            )

        if definition_chunks:
            def_urns = [c.get("urn", c.get("id", "")) for c in definition_chunks]
            steps.append(
                self._build_reasoning_step(
                    step_number=len(steps) + 1,
                    description=f"Identificate {len(definition_chunks)} definizioni legali pertinenti",
                    source_ids=def_urns,
                )
            )

        steps.append(
            self._build_reasoning_step(
                step_number=len(steps) + 1,
                description="Analisi del significato letterale delle parole secondo art. 12 disp. prel. c.c.",
            )
        )

        return steps

    def _compute_confidence(
        self,
        norm_chunks: List[Dict[str, Any]],
        definition_chunks: List[Dict[str, Any]],
        context: ExpertContext,
    ) -> Tuple[float, ConfidenceFactors]:
        """
        Compute confidence score based on source quality.

        Args:
            norm_chunks: Retrieved norm chunks
            definition_chunks: Retrieved definition chunks
            context: Input context

        Returns:
            Tuple of (confidence_score, confidence_factors)
        """
        # Source availability
        source_availability = min(1.0, len(norm_chunks) / self._config.min_norm_chunks)

        # Norm clarity based on chunk scores
        if norm_chunks:
            avg_score = sum(c.get("score", 0.5) for c in norm_chunks) / len(norm_chunks)
            norm_clarity = min(1.0, avg_score)
        else:
            norm_clarity = 0.2

        # Definition coverage
        if context.legal_concepts:
            covered = len(definition_chunks) / max(1, len(context.legal_concepts))
            definition_coverage = min(1.0, covered)
        else:
            definition_coverage = 0.5 if definition_chunks else 0.3

        # Contextual ambiguity (lower is better)
        # High if no specific norm references were found
        if context.norm_references and norm_chunks:
            contextual_ambiguity = 0.2
        elif norm_chunks:
            contextual_ambiguity = 0.4
        else:
            contextual_ambiguity = 0.8

        factors = ConfidenceFactors(
            norm_clarity=norm_clarity,
            source_availability=source_availability,
            contextual_ambiguity=contextual_ambiguity,
            definition_coverage=definition_coverage,
        )

        confidence = factors.compute_overall()

        return confidence, factors

    def _identify_limitations(
        self,
        norm_chunks: List[Dict[str, Any]],
        context: ExpertContext,
    ) -> str:
        """Identify limitations of the analysis."""
        limitations: List[str] = []

        if not norm_chunks:
            limitations.append("Nessuna norma specifica trovata nel database")

        if context.norm_references and not any(
            c.get("urn") in context.norm_references for c in norm_chunks
        ):
            limitations.append("I riferimenti normativi specificati non sono stati trovati")

        if not self.llm_service:
            limitations.append("Analisi eseguita senza supporto LLM")

        return "; ".join(limitations) if limitations else ""

    def _generate_suggestions(self, confidence: float, context: ExpertContext) -> str:
        """Generate suggestions for user based on confidence level."""
        if confidence >= self._config.high_confidence_threshold:
            return ""

        suggestions: List[str] = []

        if not context.norm_references:
            suggestions.append(
                "Specificare il riferimento normativo esatto (es. 'art. 1453 c.c.')"
            )

        if confidence < 0.3:
            suggestions.append(
                "La domanda potrebbe richiedere un'interpretazione più approfondita "
                "(sistematica o teleologica)"
            )

        return "; ".join(suggestions)
