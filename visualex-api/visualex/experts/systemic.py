"""
Systemic Expert for MERL-T Analysis Pipeline.

Implements systematic and historical interpretation following Art. 12, comma I + Art. 14:
- Art. 12, I: "...secondo la connessione di esse [parole]..."
- Art. 14: Historical interpretation principles

Systematic interpretation considers:
- CONNESSIONE: How the norm fits within the legal system
- STORICO: Evolution of the norm over time (modifications, abrogations)
- TOPOGRAFICO: Position of the norm (book, title, chapter, section)

Approach:
1. Place the norm in its systematic context (code, special law)
2. Analyze relationships with related norms (cross-references, derogations, exceptions)
3. Reconstruct historical evolution (previous versions, modifications)
4. Consider systematic ratio (coherence of the legal order)
"""

import time
import structlog
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

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


# Default prompt for systemic interpretation
SYSTEMIC_PROMPT_TEMPLATE = """Sei un esperto di interpretazione sistematica del diritto italiano.
Il tuo compito è analizzare la seguente domanda giuridica considerando il CONTESTO NORMATIVO in cui si inserisce la norma.

QUERY: {query}

NORMA PRINCIPALE:
{main_norm}

NORME CORRELATE (via RIFERIMENTO, MODIFICA, ATTUA):
{related_norms}

EVOLUZIONE STORICA:
{historical_context}

ISTRUZIONI:
1. Analizza come la norma si inserisce nel sistema giuridico complessivo
2. Considera le relazioni con altre norme (rinvii, deroghe, eccezioni)
3. Valuta l'evoluzione storica e le modifiche intervenute
4. Identifica principi generali che governano la materia
5. Verifica la coerenza sistematica dell'interpretazione

FORMATO OUTPUT:
- Inizia con la collocazione sistematica della norma
- Descrivi le connessioni rilevanti con altre norme
- Spiega come il contesto normativo influenza l'interpretazione
- Indica eventuali modifiche storiche significative
- Fornisci una sintesi dell'interpretazione sistematica

Rispondi in italiano."""


@dataclass
class GraphRelation:
    """
    Represents a relationship in the Knowledge Graph.

    Attributes:
        relation_type: Type of relationship (RIFERIMENTO, MODIFICA, ATTUA, etc.)
        source_urn: Source norm URN
        target_urn: Target norm URN
        metadata: Additional metadata (date, context, etc.)
    """

    relation_type: str
    source_urn: str
    target_urn: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "relation_type": self.relation_type,
            "source_urn": self.source_urn,
            "target_urn": self.target_urn,
            "metadata": self.metadata,
        }


@dataclass
class SystemicConfig(ExpertConfig):
    """Configuration specific to SystemicExpert."""

    # Graph traversal
    max_traversal_depth: int = 2
    max_related_norms: int = 10
    include_historical: bool = True

    # Relation type weights for traversal
    relation_weights: Dict[str, float] = None

    # LLM
    systemic_temperature: float = 0.3
    max_response_tokens: int = 2000

    # Confidence thresholds
    min_relations: int = 1
    high_confidence_threshold: float = 0.7
    isolated_norm_confidence: float = 0.3

    # F4 Feedback
    enable_f4_feedback: bool = True

    def __post_init__(self):
        """Set default relation weights if not provided."""
        if self.relation_weights is None:
            self.relation_weights = {
                "riferimento": 0.9,
                "modifica": 0.95,
                "abroga": 0.90,
                "attua": 0.85,
                "deroga": 0.85,
                "rinvia": 0.80,
                "disciplina": 0.75,
                "default": 0.5,
            }


@runtime_checkable
class GraphTraverser(Protocol):
    """
    Protocol for Knowledge Graph traversal.

    Implementations should connect to FalkorDB or similar graph database.
    """

    async def get_neighbors(
        self,
        urn: str,
        relation_types: Optional[List[str]] = None,
        depth: int = 1,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring norms in the graph.

        Args:
            urn: URN of the starting norm
            relation_types: Filter by relationship types
            depth: Maximum traversal depth
            limit: Maximum number of results

        Returns:
            List of neighbor dictionaries with node and edge info
        """
        ...

    async def get_modifications(
        self,
        urn: str,
        as_of_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get modification history for a norm.

        Args:
            urn: URN of the norm
            as_of_date: Optional date filter

        Returns:
            List of modification records
        """
        ...


class SystemicExpert(BaseExpert):
    """
    Expert for systematic interpretation (Art. 12, I + Art. 14 disp. prel. c.c.).

    Epistemology: Systemic coherence of the legal order
    Focus: How the norm FITS within the legal system

    Output:
    - "Interpretazione Sistematica" section header
    - Graph of related norms (mini visualization data)
    - Synthesis of systemic context
    - Key cross-references with explanations
    - Confidence score

    Example:
        >>> retriever = BridgeTableRetriever(...)
        >>> graph = FalkorDBGraphTraverser(...)
        >>> llm = LLMService(...)
        >>> expert = SystemicExpert(retriever=retriever, graph_traverser=graph, llm_service=llm)
        >>> context = ExpertContext(query_text="Come si collega l'art. 1453 c.c. ad altre norme?")
        >>> response = await expert.analyze(context)
        >>> print(response.section_header)
        "Interpretazione Sistematica"
    """

    expert_type = "systemic"
    section_header = "Interpretazione Sistematica"
    description = "Interpretazione sistematica e storica (art. 12, I + art. 14 disp. prel. c.c.)"

    # Relation types for traversal
    SYSTEMIC_RELATION_TYPES = [
        "riferimento",
        "modifica",
        "modificato_da",
        "abroga",
        "abrogato_da",
        "attua",
        "deroga",
        "rinvia",
    ]

    def __init__(
        self,
        retriever: Optional[ChunkRetriever] = None,
        graph_traverser: Optional[GraphTraverser] = None,
        llm_service: Optional[LLMService] = None,
        config: Optional[SystemicConfig] = None,
    ):
        """
        Initialize SystemicExpert.

        Args:
            retriever: Chunk retriever for Bridge Table access
            graph_traverser: Graph traverser for Knowledge Graph
            llm_service: LLM service for analysis
            config: SystemicExpert configuration
        """
        self._config = config or SystemicConfig()
        super().__init__(
            retriever=retriever,
            llm_service=llm_service,
            config=self._config,
        )
        self.graph_traverser = graph_traverser

    async def analyze(self, context: ExpertContext) -> ExpertResponse:
        """
        Analyze query with systematic interpretation approach.

        Flow:
        1. Retrieve main norm chunks via Bridge Table
        2. Traverse Knowledge Graph to find related norms
        3. Gather historical modifications
        4. Build LLM prompt with systemic context
        5. Generate interpretation
        6. Compute confidence based on graph connectivity

        Args:
            context: Input context with query and entities

        Returns:
            ExpertResponse with systematic interpretation
        """
        start_time = time.time()

        log.info(
            "systemic_expert_analyzing",
            query=context.query_text[:50],
            trace_id=context.trace_id,
            has_norm_refs=bool(context.norm_references),
            has_graph=self.graph_traverser is not None,
        )

        # Step 1: Retrieve main norm chunks
        main_norm_chunks = await self._retrieve_main_norm(context)

        # Step 2: Traverse graph to find related norms
        related_norms, relations = await self._traverse_graph(context, main_norm_chunks)

        # Step 3: Get historical modifications
        historical_context = await self._get_historical_context(context, main_norm_chunks)

        # Step 4: Check if norm is isolated
        if not related_norms and not context.retrieved_chunks:
            execution_time = (time.time() - start_time) * 1000
            return self._create_isolated_norm_response(
                context=context,
                main_norm_chunks=main_norm_chunks,
                execution_time_ms=execution_time,
            )

        # Step 5: Build legal sources
        legal_sources = self._build_legal_sources(
            main_norm_chunks, related_norms, relations
        )

        # Step 6: Generate interpretation
        if self.llm_service:
            interpretation, reasoning_steps, tokens = await self._generate_interpretation(
                context, main_norm_chunks, related_norms, historical_context
            )
        else:
            interpretation, reasoning_steps, tokens = self._generate_fallback_interpretation(
                context, main_norm_chunks, related_norms, relations
            )

        # Step 7: Compute confidence
        confidence, factors = self._compute_confidence(
            main_norm_chunks=main_norm_chunks,
            related_norms=related_norms,
            relations=relations,
            context=context,
        )

        execution_time = (time.time() - start_time) * 1000

        # Create F4 feedback hook
        feedback_hook = None
        if self._config.enable_f4_feedback:
            feedback_hook = FeedbackHook(
                feedback_type="F4",
                expert_type=self.expert_type,
                response_id=context.trace_id,
                enabled=True,
                correction_options={
                    # Systemic interpretation quality
                    "systemic_insight": [
                        "excellent",       # Deep systemic understanding
                        "good",            # Solid systemic analysis
                        "superficial",     # Surface-level connections
                        "misleading",      # Incorrect systemic view
                    ],
                    # Graph coverage assessment
                    "graph_coverage": [
                        "comprehensive",   # Found all relevant connections
                        "adequate",        # Found main connections
                        "incomplete",      # Missing important connections
                        "poor",            # Most connections missing
                    ],
                    # Cross-reference relevance
                    "crossref_relevance": [
                        "all_relevant",    # All cross-refs are pertinent
                        "mostly_relevant", # Most cross-refs are pertinent
                        "some_irrelevant", # Some cross-refs don't apply
                        "mostly_irrelevant", # Most cross-refs don't apply
                    ],
                    # Confidence calibration
                    "confidence_calibration": [
                        "well_calibrated",
                        "overconfident",
                        "underconfident",
                    ],
                    # Isolated norm assessment
                    "isolation_assessment": [
                        "correctly_isolated",   # Norm is truly isolated
                        "false_isolation",      # Connections exist but missed
                        "correctly_connected",  # Found real connections
                        "spurious_connections", # Found irrelevant connections
                    ],
                },
                context_snapshot={
                    "query": context.query_text[:200],
                    "main_norm_count": len(main_norm_chunks),
                    "related_norms_count": len(related_norms),
                    "relations_count": len(relations),
                    "confidence": confidence,
                    "interpretation_preview": interpretation[:300] if interpretation else "",
                    "isolated_norm": len(related_norms) == 0,
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
            limitations=self._identify_limitations(related_norms, relations, context),
            suggestions=self._generate_suggestions(confidence, context),
            trace_id=context.trace_id,
            execution_time_ms=execution_time,
            tokens_used=tokens,
            feedback_hook=feedback_hook,
            metadata={
                "main_norm_chunks": len(main_norm_chunks),
                "related_norms_count": len(related_norms),
                "relations_count": len(relations),
                "isolated_norm": len(related_norms) == 0,
                "graph_data": self._build_graph_visualization_data(relations),
            },
        )

        log.info(
            "systemic_expert_completed",
            trace_id=context.trace_id,
            confidence=confidence,
            related_norms=len(related_norms),
            execution_time_ms=execution_time,
        )

        return response

    async def _retrieve_main_norm(
        self, context: ExpertContext
    ) -> List[Dict[str, Any]]:
        """Retrieve main norm chunks from Bridge Table."""
        if not self.retriever:
            return context.retrieved_chunks or []

        filters = {
            "expert_affinity": "systemic",
            "source_type": "norm",
        }

        if context.norm_references:
            filters["urns"] = context.norm_references

        chunks = await self.retriever.retrieve(
            query=context.query_text,
            query_embedding=context.query_embedding,
            filters=filters,
            limit=self._config.chunk_limit,
        )

        return chunks or context.retrieved_chunks or []

    async def _traverse_graph(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[GraphRelation]]:
        """
        Traverse Knowledge Graph to find related norms.

        Args:
            context: Input context
            main_norm_chunks: Main norm chunks

        Returns:
            Tuple of (related_norm_chunks, graph_relations)
        """
        if not self.graph_traverser:
            log.debug("systemic_no_graph_traverser", trace_id=context.trace_id)
            return [], []

        related_norms: List[Dict[str, Any]] = []
        relations: List[GraphRelation] = []
        seen_urns: set = set()  # Track seen URNs to avoid duplicates

        # Get URNs to traverse from
        urns_to_traverse = set()
        for chunk in main_norm_chunks:
            if urn := chunk.get("urn"):
                urns_to_traverse.add(urn)
        for urn in context.norm_references:
            urns_to_traverse.add(urn)

        # Traverse from each URN
        for urn in urns_to_traverse:
            try:
                neighbors = await self.graph_traverser.get_neighbors(
                    urn=urn,
                    relation_types=self.SYSTEMIC_RELATION_TYPES,
                    depth=self._config.max_traversal_depth,
                    limit=self._config.max_related_norms,
                )

                for neighbor in neighbors:
                    target_urn = neighbor.get("urn", "")

                    # Extract relation info
                    relation = GraphRelation(
                        relation_type=neighbor.get("relation_type", "unknown"),
                        source_urn=urn,
                        target_urn=target_urn,
                        metadata=neighbor.get("edge_metadata") or {},
                    )
                    relations.append(relation)

                    # Extract norm info (deduplicated)
                    if target_urn and target_urn not in seen_urns:
                        if "text" in neighbor or "citation" in neighbor:
                            related_norms.append(neighbor)
                            seen_urns.add(target_urn)

            except Exception as e:
                log.warning(
                    "systemic_graph_traversal_error",
                    urn=urn,
                    error=str(e),
                    trace_id=context.trace_id,
                )

        log.debug(
            "systemic_graph_traversed",
            trace_id=context.trace_id,
            urns_traversed=len(urns_to_traverse),
            related_found=len(related_norms),
            relations_found=len(relations),
        )

        return related_norms, relations

    async def _get_historical_context(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Get historical modification context for norms."""
        if not self._config.include_historical or not self.graph_traverser:
            return []

        historical: List[Dict[str, Any]] = []

        for chunk in main_norm_chunks:
            if urn := chunk.get("urn"):
                try:
                    modifications = await self.graph_traverser.get_modifications(urn)
                    historical.extend(modifications)
                except Exception as e:
                    log.warning(
                        "systemic_historical_error",
                        urn=urn,
                        error=str(e),
                        trace_id=context.trace_id,
                    )

        return historical

    def _build_legal_sources(
        self,
        main_norm_chunks: List[Dict[str, Any]],
        related_norms: List[Dict[str, Any]],
        relations: List[GraphRelation],
    ) -> List[LegalSource]:
        """Build LegalSource list from all sources, ranked by relation weight."""
        sources: List[LegalSource] = []

        # Main norms (highest priority)
        for chunk in main_norm_chunks:
            sources.append(
                LegalSource(
                    source_type="norm",
                    source_id=chunk.get("urn", chunk.get("id", "")),
                    citation=chunk.get("citation", chunk.get("title", "")),
                    excerpt=chunk.get("text", "")[:400],
                    relevance="Norma principale",
                    relevance_score=1.0,  # Main norm = highest relevance
                )
            )

        # Related norms with relation context and weights
        weighted_norms: List[Tuple[float, Dict[str, Any], str]] = []

        for norm in related_norms:
            # Find the relation for this norm
            norm_urn = norm.get("urn", "")
            relation_type = "correlata"
            for rel in relations:
                if rel.target_urn == norm_urn:
                    relation_type = rel.relation_type
                    break

            # Get relation weight for ranking
            weight = self._config.relation_weights.get(
                relation_type,
                self._config.relation_weights.get("default", 0.5)
            )
            weighted_norms.append((weight, norm, relation_type))

        # Sort by weight (descending) to prioritize more relevant relations
        weighted_norms.sort(key=lambda x: x[0], reverse=True)

        for weight, norm, relation_type in weighted_norms:
            norm_urn = norm.get("urn", "")
            sources.append(
                LegalSource(
                    source_type="norm",
                    source_id=norm_urn,
                    citation=norm.get("citation", norm.get("title", "")),
                    excerpt=norm.get("text", "")[:300],
                    relevance=f"Connessa via {relation_type.upper()} (peso: {weight:.2f})",
                    relevance_score=round(min(1.0, weight), 3),
                )
            )

        return sources

    async def _generate_interpretation(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        related_norms: List[Dict[str, Any]],
        historical_context: List[Dict[str, Any]],
    ) -> Tuple[str, List[ReasoningStep], int]:
        """Generate interpretation using LLM."""
        # Format sections
        main_norm_text = self._format_chunks_for_prompt(main_norm_chunks)
        related_text = self._format_chunks_for_prompt(related_norms)
        historical_text = self._format_historical_for_prompt(historical_context)

        prompt = SYSTEMIC_PROMPT_TEMPLATE.format(
            query=context.query_text,
            main_norm=main_norm_text or "Non specificata",
            related_norms=related_text or "Nessuna norma correlata trovata",
            historical_context=historical_text or "Nessuna evoluzione storica rilevante",
        )

        try:
            interpretation = await self.llm_service.generate(
                prompt=prompt,
                temperature=self._config.systemic_temperature,
                max_tokens=self._config.max_response_tokens,
            )
            tokens_used = len(prompt.split()) + len(interpretation.split())
        except Exception as e:
            log.exception(
                "systemic_llm_error",
                error=str(e),
                trace_id=context.trace_id,
                exc_info=True,
            )
            interpretation, _, tokens_used = self._generate_fallback_interpretation(
                context, main_norm_chunks, related_norms, []
            )

        reasoning_steps = self._build_reasoning_steps(
            main_norm_chunks, related_norms, historical_context
        )

        return interpretation, reasoning_steps, tokens_used

    def _generate_fallback_interpretation(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        related_norms: List[Dict[str, Any]],
        relations: List[GraphRelation],
    ) -> Tuple[str, List[ReasoningStep], int]:
        """Generate interpretation without LLM."""
        parts: List[str] = []

        if main_norm_chunks:
            parts.append("**Norma principale:**\n")
            for chunk in main_norm_chunks[:2]:
                citation = chunk.get("citation", "Norma")
                text = chunk.get("text", "")[:300]
                parts.append(f"- {citation}: \"{text}...\"\n")

        if related_norms:
            parts.append("\n**Norme correlate nel sistema:**\n")
            for norm in related_norms[:5]:
                citation = norm.get("citation", "Norma correlata")
                parts.append(f"- {citation}\n")

        if relations:
            parts.append("\n**Relazioni sistematiche:**\n")
            relation_summary: Dict[str, int] = {}
            for rel in relations:
                relation_summary[rel.relation_type] = relation_summary.get(rel.relation_type, 0) + 1
            for rel_type, count in relation_summary.items():
                parts.append(f"- {rel_type.upper()}: {count} connessioni\n")

        if not parts:
            parts.append("Non sono state trovate connessioni sistematiche significative.")

        interpretation = "\n".join(parts)
        reasoning_steps = self._build_reasoning_steps(main_norm_chunks, related_norms, [])

        return interpretation, reasoning_steps, 0

    def _format_chunks_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for inclusion in prompt."""
        if not chunks:
            return ""

        formatted: List[str] = []
        for chunk in chunks[:5]:  # Limit to avoid prompt overflow
            citation = chunk.get("citation", chunk.get("title", "Fonte"))
            text = chunk.get("text", "")[:500]
            urn = chunk.get("urn", "")

            if urn:
                formatted.append(f"[{citation}] (URN: {urn})\n{text}\n")
            else:
                formatted.append(f"[{citation}]\n{text}\n")

        return "\n".join(formatted)

    def _format_historical_for_prompt(self, historical: List[Dict[str, Any]]) -> str:
        """Format historical context for prompt."""
        if not historical:
            return ""

        formatted: List[str] = []
        for mod in historical[:3]:
            date = mod.get("data_effetto", mod.get("date", ""))
            mod_type = mod.get("tipo_modifica", mod.get("type", "modifica"))
            source = mod.get("norma_modificante", mod.get("source", ""))

            formatted.append(f"- {date}: {mod_type} da {source}")

        return "\n".join(formatted)

    def _build_reasoning_steps(
        self,
        main_norm_chunks: List[Dict[str, Any]],
        related_norms: List[Dict[str, Any]],
        historical_context: List[Dict[str, Any]],
    ) -> List[ReasoningStep]:
        """Build reasoning chain steps."""
        steps: List[ReasoningStep] = []

        if main_norm_chunks:
            step_num = 1
            main_urns = [c.get("urn", c.get("id", "")) for c in main_norm_chunks]
            steps.append(
                self._build_reasoning_step(
                    step_number=step_num,
                    description=f"Identificata norma principale ({len(main_norm_chunks)} chunks)",
                    source_ids=main_urns,
                )
            )

        if related_norms:
            step_num = len(steps) + 1
            related_urns = [n.get("urn", n.get("id", "")) for n in related_norms]
            steps.append(
                self._build_reasoning_step(
                    step_number=step_num,
                    description=f"Traversato Knowledge Graph: {len(related_norms)} norme correlate",
                    source_ids=related_urns,
                )
            )

        if historical_context:
            step_num = len(steps) + 1
            steps.append(
                self._build_reasoning_step(
                    step_number=step_num,
                    description=f"Analizzata evoluzione storica: {len(historical_context)} modifiche",
                )
            )

        steps.append(
            self._build_reasoning_step(
                step_number=len(steps) + 1,
                description="Sintesi sistematica secondo art. 12, I + art. 14 disp. prel. c.c.",
            )
        )

        return steps

    def _compute_confidence(
        self,
        main_norm_chunks: List[Dict[str, Any]],
        related_norms: List[Dict[str, Any]],
        relations: List[GraphRelation],
        context: ExpertContext,
    ) -> Tuple[float, ConfidenceFactors]:
        """Compute confidence score based on graph connectivity."""
        # Source availability
        total_sources = len(main_norm_chunks) + len(related_norms)
        source_availability = min(1.0, total_sources / 5)

        # Norm clarity - based on main norm presence
        if main_norm_chunks:
            avg_score = sum(c.get("score", 0.5) for c in main_norm_chunks) / len(main_norm_chunks)
            norm_clarity = min(1.0, avg_score)
        else:
            norm_clarity = 0.2

        # Systemic coverage - based on relations found
        if relations:
            systemic_score = min(1.0, len(relations) / self._config.min_relations)
            definition_coverage = systemic_score
        else:
            definition_coverage = 0.2

        # Contextual ambiguity
        if related_norms and len(relations) >= self._config.min_relations:
            contextual_ambiguity = 0.2
        elif related_norms:
            contextual_ambiguity = 0.4
        else:
            contextual_ambiguity = 0.7  # Isolated norm

        factors = ConfidenceFactors(
            norm_clarity=norm_clarity,
            source_availability=source_availability,
            contextual_ambiguity=contextual_ambiguity,
            definition_coverage=definition_coverage,
        )

        confidence = factors.compute_overall()

        return confidence, factors

    def _create_isolated_norm_response(
        self,
        context: ExpertContext,
        main_norm_chunks: List[Dict[str, Any]],
        execution_time_ms: float,
    ) -> ExpertResponse:
        """Create response for isolated norm with few connections."""
        interpretation = (
            "La norma appare relativamente isolata nel sistema giuridico, "
            "con poche connessioni sistematiche rilevanti. "
            "L'interpretazione sistematica risulta quindi limitata."
        )

        if main_norm_chunks:
            interpretation += "\n\n**Norma analizzata:**\n"
            for chunk in main_norm_chunks[:1]:
                citation = chunk.get("citation", "")
                interpretation += f"- {citation}\n"

        feedback_hook = None
        if self._config.enable_f4_feedback:
            feedback_hook = FeedbackHook(
                feedback_type="F4",
                expert_type=self.expert_type,
                response_id=context.trace_id,
                enabled=True,
                correction_options={
                    # Isolation assessment is critical for this response type
                    "isolation_assessment": [
                        "correctly_isolated",   # Norm is truly isolated
                        "false_isolation",      # Connections exist but missed
                    ],
                    # Graph coverage for isolated norms
                    "graph_coverage": [
                        "graph_incomplete",     # Graph data is missing connections
                        "norm_truly_isolated",  # Norm has no systemic connections
                        "search_failed",        # Technical failure in search
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
                    "main_norm_count": len(main_norm_chunks),
                    "isolated_norm": True,
                    "confidence": self._config.isolated_norm_confidence,
                    "interpretation_preview": interpretation[:300],
                },
            )

        # Build limitations message based on state
        if not self.graph_traverser:
            limitations = "Knowledge Graph non disponibile"
        else:
            limitations = "Poche connessioni sistematiche trovate nel Knowledge Graph"

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
                    relevance="Norma principale (isolata)",
                    relevance_score=0.5,  # Isolated norm has reduced relevance
                )
                for c in main_norm_chunks[:1]
            ],
            confidence=self._config.isolated_norm_confidence,
            confidence_factors=ConfidenceFactors(
                norm_clarity=0.5,
                source_availability=0.3,
                contextual_ambiguity=0.7,
                definition_coverage=0.2,
            ),
            limitations=limitations,
            suggestions="Verificare se la norma ha connessioni non ancora indicizzate nel sistema",
            trace_id=context.trace_id,
            execution_time_ms=execution_time_ms,
            feedback_hook=feedback_hook,
            metadata={
                "isolated_norm": True,
                "main_norm_chunks": len(main_norm_chunks),
            },
        )

    def _build_graph_visualization_data(
        self, relations: List[GraphRelation]
    ) -> Dict[str, Any]:
        """Build data for graph visualization in UI."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for rel in relations:
            # Add source node
            if rel.source_urn not in nodes:
                nodes[rel.source_urn] = {
                    "id": rel.source_urn,
                    "type": "norm",
                }
            # Add target node
            if rel.target_urn not in nodes:
                nodes[rel.target_urn] = {
                    "id": rel.target_urn,
                    "type": "norm",
                }
            # Add edge
            edges.append({
                "source": rel.source_urn,
                "target": rel.target_urn,
                "type": rel.relation_type,
            })

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
        }

    def _identify_limitations(
        self,
        related_norms: List[Dict[str, Any]],
        relations: List[GraphRelation],
        context: ExpertContext,
    ) -> str:
        """Identify limitations of the analysis."""
        limitations: List[str] = []

        if not self.graph_traverser:
            limitations.append("Knowledge Graph non disponibile per traversal")

        if not related_norms:
            limitations.append("Nessuna norma correlata trovata")

        if not relations:
            limitations.append("Nessuna relazione sistematica identificata")

        if not self.llm_service:
            limitations.append("Analisi eseguita senza supporto LLM")

        return "; ".join(limitations) if limitations else ""

    def _generate_suggestions(self, confidence: float, context: ExpertContext) -> str:
        """Generate suggestions based on confidence level."""
        if confidence >= self._config.high_confidence_threshold:
            return ""

        suggestions: List[str] = []

        if not context.norm_references:
            suggestions.append("Specificare il riferimento normativo per migliorare la ricerca sistematica")

        if confidence < 0.4:
            suggestions.append("Considerare l'interpretazione letterale come punto di partenza")

        return "; ".join(suggestions)
