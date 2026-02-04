"""
Expert Router for MERL-T Analysis Pipeline.

Routes queries to appropriate Experts following Art. 12 Preleggi hierarchy:
Literal → Systemic → Principles → Precedent

Query types supported:
- DEFINITION: "Cos'è...", "Definizione di..."
- INTERPRETATION: "Come interpretare...", "Significato di..."
- COMPARISON: "Differenza tra...", "Confronto..."
- CASE_ANALYSIS: "Nel caso in cui...", "Posso..."

The router is trainable via RLCF F2 feedback.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import structlog

from ..ner import ExtractionResult, EntityType

log = structlog.get_logger()


class QueryType(str, Enum):
    """Types of legal queries."""

    DEFINITION = "DEFINITION"  # Definitional queries
    INTERPRETATION = "INTERPRETATION"  # Interpretive queries
    COMPARISON = "COMPARISON"  # Comparative analysis
    CASE_ANALYSIS = "CASE_ANALYSIS"  # Case-specific application


class ExpertType(str, Enum):
    """Art. 12 Preleggi expert types."""

    LITERAL = "literal"  # Interpretazione letterale
    SYSTEMIC = "systemic"  # Interpretazione sistematica
    PRINCIPLES = "principles"  # Ratio legis / Principi
    PRECEDENT = "precedent"  # Giurisprudenza


@dataclass
class ExpertWeight:
    """Weight assignment for an Expert."""

    expert: ExpertType
    weight: float  # 0.0 - 1.0
    is_primary: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "expert": self.expert.value,
            "weight": round(self.weight, 3),
            "is_primary": self.is_primary,
            "skip_reason": self.skip_reason,
        }


@dataclass
class RoutingDecision:
    """
    Routing decision for a query.

    Contains weights for each Expert and rationale for F2 feedback.
    """

    query_type: QueryType
    expert_weights: List[ExpertWeight]
    confidence: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "query_type": self.query_type.value,
            "expert_weights": [w.to_dict() for w in self.expert_weights],
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "metadata": self.metadata,
            "activation_order": self.get_activation_order(),
        }

    def get_activation_order(self, threshold: float = 0.1) -> List[str]:
        """
        Get Expert activation order (Art. 12 Preleggi sequence).

        Returns Experts with weight >= threshold, ordered by weight desc.
        """
        active = [w for w in self.expert_weights if w.weight >= threshold]
        sorted_active = sorted(active, key=lambda x: x.weight, reverse=True)
        return [w.expert.value for w in sorted_active]

    def get_primary_expert(self) -> Optional[ExpertType]:
        """Get the primary Expert (highest weight)."""
        if not self.expert_weights:
            return None
        primary = max(self.expert_weights, key=lambda x: x.weight)
        return primary.expert if primary.weight > 0 else None


# Query classification patterns
QUERY_PATTERNS = {
    QueryType.DEFINITION: [
        r"cos['\s]?[eè]\s",
        r"definizione\s+di",
        r"cosa\s+(si\s+)?intende\s+per",
        r"significato\s+di",
        r"nozione\s+di",
        r"che\s+cos['\s]?[eè]",
    ],
    QueryType.INTERPRETATION: [
        r"come\s+(si\s+)?interpreta",
        r"interpretazione\s+di",
        r"senso\s+di",
        r"portata\s+di",
        r"ambito\s+di\s+applicazione",
        r"come\s+va\s+inteso",
    ],
    QueryType.COMPARISON: [
        r"differenza\s+tra",
        r"confronto\s+(tra|fra)",
        r"distinzione\s+tra",
        r"quale\s+differenza",
        r"rispetto\s+a",
        r"a\s+differenza\s+di",
    ],
    QueryType.CASE_ANALYSIS: [
        r"nel\s+caso\s+(in\s+cui)?",
        r"posso\s+",
        r"[eè]\s+possibile\s+",
        r"se\s+.+\s+(posso|devo|ho|sono)",
        r"qualora\s+",
        r"laddove\s+",
        r"nell'ipotesi\s+(in\s+cui)?",
    ],
}

# Default weights by query type (Art. 12 Preleggi compliant)
DEFAULT_WEIGHTS = {
    QueryType.DEFINITION: {
        ExpertType.LITERAL: 0.60,
        ExpertType.SYSTEMIC: 0.20,
        ExpertType.PRINCIPLES: 0.10,
        ExpertType.PRECEDENT: 0.10,
    },
    QueryType.INTERPRETATION: {
        ExpertType.LITERAL: 0.35,
        ExpertType.SYSTEMIC: 0.25,
        ExpertType.PRINCIPLES: 0.20,
        ExpertType.PRECEDENT: 0.20,
    },
    QueryType.COMPARISON: {
        ExpertType.LITERAL: 0.30,
        ExpertType.SYSTEMIC: 0.35,
        ExpertType.PRINCIPLES: 0.20,
        ExpertType.PRECEDENT: 0.15,
    },
    QueryType.CASE_ANALYSIS: {
        ExpertType.LITERAL: 0.25,
        ExpertType.SYSTEMIC: 0.20,
        ExpertType.PRINCIPLES: 0.25,
        ExpertType.PRECEDENT: 0.30,
    },
}


@dataclass
class RouterConfig:
    """Configuration for Expert Router."""

    weight_threshold: float = 0.1  # Min weight to activate Expert
    confidence_threshold: float = 0.5  # Min confidence for routing
    boost_factor: float = 1.2  # Boost factor for entity-based adjustments
    max_boost: float = 0.8  # Maximum weight after boost
    record_rationale: bool = True  # Record rationale for F2 feedback


class ExpertRouter:
    """
    Routes queries to appropriate Experts following Art. 12 Preleggi.

    The router:
    1. Classifies query type (DEFINITION, INTERPRETATION, etc.)
    2. Determines Expert activation order and weights
    3. Adjusts weights based on NER entities
    4. Records routing rationale for F2 feedback
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        """
        Initialize Expert Router.

        Args:
            config: Router configuration
        """
        self.config = config or RouterConfig()
        self._compiled_patterns = {
            query_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for query_type, patterns in QUERY_PATTERNS.items()
        }
        log.info("expert_router_initialized")

    async def route(
        self,
        query: str,
        ner_result: Optional[ExtractionResult] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route query to appropriate Experts.

        Args:
            query: User query text
            ner_result: NER extraction result (optional)
            context: Additional context (optional)

        Returns:
            RoutingDecision with Expert weights and rationale
        """
        # Step 1: Classify query type
        query_type, type_confidence = self._classify_query(query)

        # Step 2: Get base weights for query type
        base_weights = DEFAULT_WEIGHTS.get(
            query_type, DEFAULT_WEIGHTS[QueryType.INTERPRETATION]
        ).copy()

        # Step 3: Adjust weights based on NER entities
        adjusted_weights = self._adjust_for_entities(base_weights, ner_result)

        # Step 4: Adjust based on keywords
        final_weights = self._adjust_for_keywords(adjusted_weights, query)

        # Step 5: Normalize weights
        final_weights = self._normalize_weights(final_weights)

        # Step 6: Build Expert weight list
        expert_weights = self._build_expert_weights(final_weights, query_type)

        # Step 7: Build rationale for F2 feedback
        rationale = self._build_rationale(query_type, ner_result, final_weights)

        decision = RoutingDecision(
            query_type=query_type,
            expert_weights=expert_weights,
            confidence=type_confidence,
            rationale=rationale,
            metadata={
                "original_query": query,
                "ner_entity_count": len(ner_result.entities) if ner_result else 0,
            },
        )

        log.info(
            "routing_decision_made",
            query_type=query_type.value,
            confidence=type_confidence,
            primary_expert=decision.get_primary_expert(),
        )

        return decision

    def _classify_query(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify query type based on patterns.

        Returns:
            Tuple of (QueryType, confidence)
        """
        query_lower = query.lower()
        scores: Dict[QueryType, float] = {}

        for query_type, patterns in self._compiled_patterns.items():
            matches = sum(1 for p in patterns if p.search(query_lower))
            if matches > 0:
                scores[query_type] = matches / len(patterns)

        if not scores:
            # Default to INTERPRETATION for general queries
            return QueryType.INTERPRETATION, 0.5

        best_type = max(scores, key=lambda k: scores[k])
        confidence = min(scores[best_type] * 2, 1.0)

        return best_type, confidence

    def _adjust_for_entities(
        self,
        weights: Dict[ExpertType, float],
        ner_result: Optional[ExtractionResult],
    ) -> Dict[ExpertType, float]:
        """Adjust weights based on NER entities."""
        if not ner_result:
            return weights

        adjusted = weights.copy()
        boost = self.config.boost_factor
        max_w = self.config.max_boost

        # Norm references → boost Literal
        if ner_result.article_refs:
            adjusted[ExpertType.LITERAL] = min(
                adjusted[ExpertType.LITERAL] * boost, max_w
            )

        # Legal concepts → check for principles-related terms
        concept_texts = [c.text.lower() for c in ner_result.legal_concepts]
        principles_terms = ["principio", "diritto", "libertà", "tutela", "garanzia"]
        if any(t in " ".join(concept_texts) for t in principles_terms):
            adjusted[ExpertType.PRINCIPLES] = min(
                adjusted[ExpertType.PRINCIPLES] * boost, max_w
            )

        # Temporal references → boost Systemic (historical context)
        if ner_result.temporal_refs:
            adjusted[ExpertType.SYSTEMIC] = min(
                adjusted[ExpertType.SYSTEMIC] * (boost * 0.8), max_w
            )

        return adjusted

    def _adjust_for_keywords(
        self,
        weights: Dict[ExpertType, float],
        query: str,
    ) -> Dict[ExpertType, float]:
        """Adjust weights based on specific keywords."""
        adjusted = weights.copy()
        query_lower = query.lower()
        boost = self.config.boost_factor
        max_w = self.config.max_boost

        # Systemic keywords
        if any(kw in query_lower for kw in ["storico", "evoluzione", "modifica", "sistema"]):
            adjusted[ExpertType.SYSTEMIC] = min(
                adjusted[ExpertType.SYSTEMIC] * boost, max_w
            )

        # Principles keywords
        if any(kw in query_lower for kw in ["ratio", "scopo", "finalità", "intenzione"]):
            adjusted[ExpertType.PRINCIPLES] = min(
                adjusted[ExpertType.PRINCIPLES] * boost, max_w
            )

        # Literal keywords
        if any(kw in query_lower for kw in ["letterale", "testuale", "parola", "testo"]):
            adjusted[ExpertType.LITERAL] = min(
                adjusted[ExpertType.LITERAL] * boost, max_w
            )

        # Precedent keywords
        if any(kw in query_lower for kw in ["giurisprudenza", "cassazione", "sentenza", "prassi"]):
            adjusted[ExpertType.PRECEDENT] = min(
                adjusted[ExpertType.PRECEDENT] * boost, max_w
            )

        return adjusted

    def _normalize_weights(
        self, weights: Dict[ExpertType, float]
    ) -> Dict[ExpertType, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total == 0:
            # Equal distribution if all zero
            return {exp: 0.25 for exp in ExpertType}
        return {exp: w / total for exp, w in weights.items()}

    def _build_expert_weights(
        self,
        weights: Dict[ExpertType, float],
        query_type: QueryType,
    ) -> List[ExpertWeight]:
        """Build ExpertWeight list with primary flag and skip reasons."""
        result: List[ExpertWeight] = []
        threshold = self.config.weight_threshold

        # Find primary Expert
        primary_exp = max(weights, key=lambda k: weights[k])

        # Art. 12 Preleggi order: Literal → Systemic → Principles → Precedent
        for expert in [
            ExpertType.LITERAL,
            ExpertType.SYSTEMIC,
            ExpertType.PRINCIPLES,
            ExpertType.PRECEDENT,
        ]:
            weight = weights.get(expert, 0.0)
            is_primary = (expert == primary_exp)

            skip_reason = None
            if weight < threshold:
                skip_reason = f"Weight {weight:.2f} below threshold {threshold}"

            result.append(
                ExpertWeight(
                    expert=expert,
                    weight=weight,
                    is_primary=is_primary,
                    skip_reason=skip_reason,
                )
            )

        return result

    def _build_rationale(
        self,
        query_type: QueryType,
        ner_result: Optional[ExtractionResult],
        weights: Dict[ExpertType, float],
    ) -> str:
        """Build routing rationale for F2 feedback."""
        if not self.config.record_rationale:
            return ""

        parts = [f"Query classificata come {query_type.value}."]

        # Primary Expert
        primary = max(weights, key=lambda k: weights[k])
        parts.append(
            f"Expert primario: {primary.value} (peso: {weights[primary]:.2f})."
        )

        # Entity-based adjustments
        if ner_result:
            if ner_result.article_refs:
                parts.append(
                    f"Rilevati {len(ner_result.article_refs)} riferimenti normativi."
                )
            if ner_result.legal_concepts:
                parts.append(
                    f"Rilevati {len(ner_result.legal_concepts)} concetti giuridici."
                )

        # Activation order
        active_experts = [
            exp.value for exp, w in weights.items()
            if w >= self.config.weight_threshold
        ]
        parts.append(f"Expert attivati: {', '.join(active_experts)}.")

        return " ".join(parts)

    def route_sync(
        self,
        query: str,
        ner_result: Optional[ExtractionResult] = None,
    ) -> RoutingDecision:
        """
        Synchronous version of route().

        Note: Prefer using the async route() method. This is provided
        for compatibility with synchronous code paths only.
        Not recommended for use in async contexts.
        """
        import asyncio
        return asyncio.run(self.route(query, ner_result))
