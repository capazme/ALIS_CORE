"""
Gating Network for MERL-T Analysis Pipeline.

Combines Expert outputs with learned weights following the
Art. 12 Preleggi interpretation hierarchy.

The Gating Network:
1. Receives ExpertResponse from multiple Experts
2. Combines interpretations using configurable weights
3. Produces aggregated output with traceability
4. Handles conflicts between Experts
5. Integrates with RLCF feedback loop (F7)

Aggregation strategies:
- weighted_average: Weighted combination of all experts
- best_confidence: Uses only highest-confidence expert
- consensus: Finds common ground between experts
- ensemble: Preserves all perspectives separately

Example:
    >>> gating = GatingNetwork()
    >>> responses = [literal_response, systemic_response, precedent_response]
    >>> weights = {"literal": 0.5, "systemic": 0.3, "precedent": 0.2}
    >>> aggregated = await gating.aggregate(responses, weights)
"""

import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Optional

from .base import (
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    ConfidenceFactors,
    FeedbackHook,
    LLMService,
)

log = structlog.get_logger()


class AggregationMethod(str, Enum):
    """Available aggregation methods."""

    WEIGHTED_AVERAGE = "weighted_average"
    BEST_CONFIDENCE = "best_confidence"
    CONSENSUS = "consensus"
    ENSEMBLE = "ensemble"


# Default weights based on Art. 12 Preleggi hierarchy (immutable)
_DEFAULT_EXPERT_WEIGHTS: Dict[str, float] = {
    "literal": 0.35,  # Primary: Interpretazione letterale
    "systemic": 0.30,  # Secondary: Interpretazione sistematica
    "principles": 0.20,  # Tertiary: Principi / Ratio legis
    "precedent": 0.15,  # Supporting: Giurisprudenza
}
DEFAULT_EXPERT_WEIGHTS = MappingProxyType(_DEFAULT_EXPERT_WEIGHTS)


# User profile weight modifiers (for RLCF future integration)
USER_PROFILE_MODIFIERS: Dict[str, Dict[str, float]] = {
    "analysis": {  # 🔍 Analisi - full trace, prefer multiple experts
        "literal": 1.0,
        "systemic": 1.0,
        "principles": 1.0,
        "precedent": 1.0,
    },
    "quick": {  # ⚡ Quick - prefer highest confidence
        "literal": 1.2,
        "systemic": 0.9,
        "principles": 0.8,
        "precedent": 0.8,
    },
    "academic": {  # 📚 Academic - emphasize principles and precedent
        "literal": 0.9,
        "systemic": 1.0,
        "principles": 1.3,
        "precedent": 1.2,
    },
}


@dataclass
class ExpertContribution:
    """
    Contribution from a single Expert to the aggregated response.

    Attributes:
        expert_type: Type of expert (literal, systemic, etc.)
        interpretation: Expert's interpretation text
        confidence: Original confidence score
        weight: Assigned weight in aggregation
        weighted_confidence: confidence * weight
        selected: Whether this was the primary expert (for best_confidence)
    """

    expert_type: str
    interpretation: str
    confidence: float
    weight: float
    weighted_confidence: float
    selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "expert_type": self.expert_type,
            "interpretation_preview": self.interpretation[:200] if self.interpretation else "",
            "confidence": round(self.confidence, 3),
            "weight": round(self.weight, 3),
            "weighted_confidence": round(self.weighted_confidence, 3),
            "selected": self.selected,
        }


@dataclass
class AggregatedResponse:
    """
    Aggregated response from multiple Experts.

    Attributes:
        synthesis: Unified synthesis of all interpretations
        expert_contributions: Individual contributions from each expert
        combined_legal_basis: Deduplicated legal sources
        combined_reasoning: Merged reasoning steps
        confidence: Aggregated confidence score
        confidence_breakdown: Per-expert weighted confidence
        conflicts: Detected conflicts between experts
        aggregation_method: Method used for aggregation
        trace_id: Tracing identifier
        execution_time_ms: Processing time
        feedback_hook: F7 feedback hook for RLCF
    """

    synthesis: str
    expert_contributions: Dict[str, ExpertContribution]
    combined_legal_basis: List[LegalSource] = field(default_factory=list)
    combined_reasoning: List[ReasoningStep] = field(default_factory=list)
    confidence: float = 0.5
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    aggregation_method: str = "weighted_average"
    trace_id: str = ""
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    feedback_hook: Optional[FeedbackHook] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API response."""
        return {
            "synthesis": self.synthesis,
            "expert_contributions": {
                k: v.to_dict() for k, v in self.expert_contributions.items()
            },
            "combined_legal_basis": [lb.to_dict() for lb in self.combined_legal_basis],
            "combined_reasoning": [rs.to_dict() for rs in self.combined_reasoning],
            "confidence": round(self.confidence, 3),
            "confidence_breakdown": {
                k: round(v, 3) for k, v in self.confidence_breakdown.items()
            },
            "conflicts": self.conflicts,
            "aggregation_method": self.aggregation_method,
            "trace_id": self.trace_id,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timestamp": self.timestamp,
            "feedback_hook": self.feedback_hook.to_dict() if self.feedback_hook else None,
            "metadata": self.metadata,
        }


@dataclass
class GatingConfig:
    """
    Configuration for GatingNetwork.

    Attributes:
        method: Default aggregation method
        default_weights: Default expert weights
        confidence_divergence_threshold: Threshold for flagging confidence conflicts
        source_overlap_threshold: Threshold for flagging source divergence
        max_legal_sources: Maximum combined legal sources
        max_reasoning_steps: Maximum combined reasoning steps
        enable_f7_feedback: Enable F7 feedback hook
        llm_temperature: Temperature for LLM synthesis
        max_synthesis_tokens: Max tokens for synthesis
    """

    method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    default_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EXPERT_WEIGHTS))
    confidence_divergence_threshold: float = 0.4
    source_overlap_threshold: float = 0.2
    max_legal_sources: int = 10
    max_reasoning_steps: int = 15
    enable_f7_feedback: bool = True
    llm_temperature: float = 0.3
    max_synthesis_tokens: int = 2000


# Synthesis prompt template
GATING_SYNTHESIS_PROMPT = """Sei un giurista esperto. Sintetizza le seguenti interpretazioni
da diversi approcci ermeneutici (Art. 12 Disposizioni Preliminari c.c.) in una risposta coerente.

{expert_sections}

ISTRUZIONI:
1. Integra le diverse prospettive in modo coerente
2. Rispetta la gerarchia interpretativa (letterale > sistematica > teleologica > giurisprudenziale)
3. Se ci sono divergenze, spiega le diverse posizioni
4. Cita le fonti più rilevanti
5. Fornisci una conclusione chiara

Rispondi in italiano con una sintesi strutturata."""


class GatingNetwork:
    """
    Network for aggregating responses from multiple Experts.

    Supports multiple aggregation strategies:
    - weighted_average: Combines with configurable weights
    - best_confidence: Selects highest-confidence expert
    - consensus: Finds common ground
    - ensemble: Preserves all perspectives

    Integrates with RLCF via F7 feedback hook for weight learning.

    Example:
        >>> gating = GatingNetwork(method=AggregationMethod.WEIGHTED_AVERAGE)
        >>> responses = [literal_resp, systemic_resp, precedent_resp]
        >>> weights = {"literal": 0.5, "systemic": 0.3, "precedent": 0.2}
        >>> result = await gating.aggregate(responses, weights)
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        config: Optional[GatingConfig] = None,
    ):
        """
        Initialize GatingNetwork.

        Args:
            llm_service: LLM service for synthesis (optional)
            config: Gating configuration
        """
        self._config = config or GatingConfig()
        self.llm_service = llm_service

        log.info(
            "gating_network_initialized",
            method=self._config.method.value,
        )

    async def aggregate(
        self,
        responses: List[ExpertResponse],
        weights: Optional[Dict[str, float]] = None,
        trace_id: str = "",
        user_profile: Optional[str] = None,
    ) -> AggregatedResponse:
        """
        Aggregate Expert responses with learned/configured weights.

        Args:
            responses: List of ExpertResponse from activated experts
            weights: Optional override weights (defaults to config weights)
            trace_id: Tracing identifier
            user_profile: User profile for weight adjustment (future)

        Returns:
            AggregatedResponse with unified synthesis
        """
        start_time = time.time()

        if not responses:
            return self._create_empty_response(trace_id)

        log.info(
            "gating_aggregating",
            expert_count=len(responses),
            method=self._config.method.value,
            trace_id=trace_id,
        )

        # Use provided weights or defaults
        base_weights = dict(weights) if weights else dict(self._config.default_weights)

        # Apply user profile modifier if provided
        if user_profile and user_profile in USER_PROFILE_MODIFIERS:
            modifiers = USER_PROFILE_MODIFIERS[user_profile]
            base_weights = {
                k: v * modifiers.get(k, 1.0)
                for k, v in base_weights.items()
            }
            log.debug(
                "gating_user_profile_applied",
                profile=user_profile,
                trace_id=trace_id,
            )

        # Normalize weights for present experts only
        normalized_weights = self._normalize_weights(responses, base_weights)

        # Execute aggregation based on method
        method = self._config.method
        if method == AggregationMethod.WEIGHTED_AVERAGE:
            result = await self._aggregate_weighted(responses, normalized_weights, trace_id)
        elif method == AggregationMethod.BEST_CONFIDENCE:
            result = await self._aggregate_best(responses, trace_id)
        elif method == AggregationMethod.CONSENSUS:
            result = await self._aggregate_consensus(responses, normalized_weights, trace_id)
        elif method == AggregationMethod.ENSEMBLE:
            result = await self._aggregate_ensemble(responses, normalized_weights, trace_id)
        else:
            # Fallback to weighted average
            result = await self._aggregate_weighted(responses, normalized_weights, trace_id)

        # Add timing and metadata
        result.execution_time_ms = (time.time() - start_time) * 1000
        result.trace_id = trace_id

        # Add F7 feedback hook
        if self._config.enable_f7_feedback:
            result.feedback_hook = FeedbackHook(
                feedback_type="F7",
                expert_type="gating",
                response_id=trace_id,
                enabled=True,
                correction_options={
                    # Weight appropriateness
                    "weight_appropriateness": [
                        "appropriate",        # Weights correctly balanced
                        "literal_overweight", # Literal expert overweighted
                        "literal_underweight", # Literal expert underweighted
                        "systemic_overweight",
                        "systemic_underweight",
                        "principles_overweight",
                        "principles_underweight",
                        "precedent_overweight",
                        "precedent_underweight",
                    ],
                    # Conflict detection accuracy
                    "conflict_detection": [
                        "correctly_identified",   # All conflicts found
                        "missed_conflict",        # Undetected expert conflict
                        "false_conflict",         # Reported conflict that doesn't exist
                        "no_conflicts",           # Correctly reported no conflicts
                    ],
                    # Aggregation method appropriateness
                    "aggregation_method": [
                        "method_appropriate",     # Right method for situation
                        "should_use_weighted",    # Should have used weighted avg
                        "should_use_max",         # Should have used max confidence
                        "should_use_bayesian",    # Should have used Bayesian
                    ],
                    # Combined confidence calibration
                    "combined_confidence": [
                        "well_calibrated",
                        "overconfident",
                        "underconfident",
                    ],
                },
                context_snapshot={
                    "expert_count": len(responses),
                    "expert_confidences": {r.expert_type: r.confidence for r in responses},
                    "weights_used": normalized_weights,
                    "combined_confidence": result.confidence,
                    "conflicts": result.conflicts,
                    "aggregation_method": method.value,
                },
            )

        # Add metadata for RLCF
        result.metadata = {
            "expert_count": len(responses),
            "weights_used": normalized_weights,
            "aggregation_method": method.value,
            "total_sources": len(result.combined_legal_basis),
        }

        log.info(
            "gating_completed",
            method=method.value,
            confidence=result.confidence,
            conflicts=len(result.conflicts),
            execution_time_ms=result.execution_time_ms,
        )

        return result

    def _normalize_weights(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalize weights for present experts.

        Args:
            responses: Expert responses
            weights: Base weights

        Returns:
            Normalized weights summing to 1.0
        """
        present_experts = {r.expert_type for r in responses}
        filtered = {k: v for k, v in weights.items() if k in present_experts}

        total = sum(filtered.values())
        if total > 0:
            return {k: v / total for k, v in filtered.items()}

        # Equal weights if no weights match
        return {exp: 1.0 / len(present_experts) for exp in present_experts}

    async def _aggregate_weighted(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
        trace_id: str,
    ) -> AggregatedResponse:
        """
        Aggregate with weighted average.

        Combines all expert outputs weighted by configured/learned weights.
        """
        # Build contributions
        contributions: Dict[str, ExpertContribution] = {}
        for resp in responses:
            w = weights.get(resp.expert_type, 0.0)
            contributions[resp.expert_type] = ExpertContribution(
                expert_type=resp.expert_type,
                interpretation=resp.interpretation,
                confidence=resp.confidence,
                weight=w,
                weighted_confidence=resp.confidence * w,
            )

        # Compute aggregated confidence
        weighted_confidence = sum(c.weighted_confidence for c in contributions.values())
        confidence_breakdown = {
            exp: c.weighted_confidence for exp, c in contributions.items()
        }

        # Combine legal basis (deduplicate by source_id, prioritize by weight)
        combined_basis = self._combine_legal_sources(responses, weights)

        # Combine reasoning steps
        combined_reasoning = self._combine_reasoning_steps(responses, weights)

        # Detect conflicts
        conflicts = self._detect_conflicts(responses)

        # Generate synthesis
        if self.llm_service:
            synthesis = await self._synthesize_with_llm(responses, weights)
        else:
            synthesis = self._synthesize_simple(responses, weights)

        return AggregatedResponse(
            synthesis=synthesis,
            expert_contributions=contributions,
            combined_legal_basis=combined_basis[:self._config.max_legal_sources],
            combined_reasoning=combined_reasoning[:self._config.max_reasoning_steps],
            confidence=weighted_confidence,
            confidence_breakdown=confidence_breakdown,
            conflicts=conflicts,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE.value,
            trace_id=trace_id,
        )

    async def _aggregate_best(
        self,
        responses: List[ExpertResponse],
        trace_id: str,
    ) -> AggregatedResponse:
        """
        Aggregate using only the highest-confidence expert.

        Selects the expert with highest confidence score.
        """
        best = max(responses, key=lambda r: r.confidence)

        contributions: Dict[str, ExpertContribution] = {}
        for resp in responses:
            is_selected = resp.expert_type == best.expert_type
            contributions[resp.expert_type] = ExpertContribution(
                expert_type=resp.expert_type,
                interpretation=resp.interpretation,
                confidence=resp.confidence,
                weight=1.0 if is_selected else 0.0,
                weighted_confidence=resp.confidence if is_selected else 0.0,
                selected=is_selected,
            )

        return AggregatedResponse(
            synthesis=best.interpretation,
            expert_contributions=contributions,
            combined_legal_basis=best.legal_basis[:self._config.max_legal_sources],
            combined_reasoning=best.reasoning_steps[:self._config.max_reasoning_steps],
            confidence=best.confidence,
            confidence_breakdown={best.expert_type: best.confidence},
            conflicts=[],
            aggregation_method=AggregationMethod.BEST_CONFIDENCE.value,
            trace_id=trace_id,
        )

    async def _aggregate_consensus(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
        trace_id: str,
    ) -> AggregatedResponse:
        """
        Aggregate by finding consensus between experts.

        Focuses on sources cited by multiple experts.
        """
        # Find sources cited by multiple experts
        source_counts: Dict[str, Dict[str, Any]] = {}
        for resp in responses:
            for lb in resp.legal_basis:
                if lb.source_id not in source_counts:
                    source_counts[lb.source_id] = {
                        "source": lb,
                        "experts": [],
                        "count": 0,
                    }
                source_counts[lb.source_id]["experts"].append(resp.expert_type)
                source_counts[lb.source_id]["count"] += 1

        # Sources with consensus (cited by 2+ experts)
        consensus_sources = [
            s["source"] for s in source_counts.values() if s["count"] >= 2
        ]

        # Build contributions
        contributions: Dict[str, ExpertContribution] = {}
        for resp in responses:
            w = weights.get(resp.expert_type, 0.0)
            contributions[resp.expert_type] = ExpertContribution(
                expert_type=resp.expert_type,
                interpretation=resp.interpretation,
                confidence=resp.confidence,
                weight=w,
                weighted_confidence=resp.confidence * w,
            )

        # Confidence based on consensus level
        if consensus_sources:
            consensus_confidence = min(
                len(consensus_sources) / max(len(source_counts), 1) + 0.3,
                1.0,
            )
        else:
            consensus_confidence = 0.4

        # Synthesis focusing on consensus
        if self.llm_service:
            synthesis = await self._synthesize_with_llm(responses, weights, focus="consensus")
        else:
            synthesis = self._synthesize_consensus(responses, consensus_sources)

        return AggregatedResponse(
            synthesis=synthesis,
            expert_contributions=contributions,
            combined_legal_basis=consensus_sources[:self._config.max_legal_sources],
            combined_reasoning=[],
            confidence=consensus_confidence,
            confidence_breakdown={r.expert_type: r.confidence for r in responses},
            conflicts=self._detect_conflicts(responses),
            aggregation_method=AggregationMethod.CONSENSUS.value,
            trace_id=trace_id,
        )

    async def _aggregate_ensemble(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
        trace_id: str,
    ) -> AggregatedResponse:
        """
        Aggregate preserving all expert perspectives.

        Maintains each interpretation as a separate section.
        """
        contributions: Dict[str, ExpertContribution] = {}
        sections: List[str] = []

        # Sort by weight (highest first)
        sorted_responses = sorted(
            responses,
            key=lambda r: weights.get(r.expert_type, 0),
            reverse=True,
        )

        for resp in sorted_responses:
            w = weights.get(resp.expert_type, 0.0)
            contributions[resp.expert_type] = ExpertContribution(
                expert_type=resp.expert_type,
                interpretation=resp.interpretation,
                confidence=resp.confidence,
                weight=w,
                weighted_confidence=resp.confidence * w,
            )

            header = resp.section_header or resp.expert_type.title()
            sections.append(f"## {header}\n{resp.interpretation}")

        synthesis = "\n\n".join(sections)

        # Combine all sources
        all_basis = self._combine_legal_sources(responses, weights)

        # Average confidence
        avg_confidence = sum(r.confidence for r in responses) / len(responses)

        return AggregatedResponse(
            synthesis=synthesis,
            expert_contributions=contributions,
            combined_legal_basis=all_basis[:self._config.max_legal_sources + 5],
            combined_reasoning=[],
            confidence=avg_confidence,
            confidence_breakdown={r.expert_type: r.confidence for r in responses},
            conflicts=self._detect_conflicts(responses),
            aggregation_method=AggregationMethod.ENSEMBLE.value,
            trace_id=trace_id,
        )

    def _combine_legal_sources(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
    ) -> List[LegalSource]:
        """Combine legal sources from all experts, deduplicated."""
        combined: List[LegalSource] = []
        seen_ids: set = set()

        # Sort responses by weight (higher weight = priority)
        sorted_responses = sorted(
            responses,
            key=lambda r: weights.get(r.expert_type, 0),
            reverse=True,
        )

        for resp in sorted_responses:
            for lb in resp.legal_basis:
                if lb.source_id not in seen_ids:
                    combined.append(lb)
                    seen_ids.add(lb.source_id)

        return combined

    def _combine_reasoning_steps(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
    ) -> List[ReasoningStep]:
        """Combine reasoning steps from all experts."""
        combined: List[ReasoningStep] = []
        step_num = 1

        sorted_responses = sorted(
            responses,
            key=lambda r: weights.get(r.expert_type, 0),
            reverse=True,
        )

        for resp in sorted_responses:
            for rs in resp.reasoning_steps:
                combined.append(
                    ReasoningStep(
                        step_number=step_num,
                        description=f"[{resp.expert_type}] {rs.description}",
                        sources=rs.sources,
                    )
                )
                step_num += 1

        return combined

    def _detect_conflicts(self, responses: List[ExpertResponse]) -> List[str]:
        """
        Detect conflicts between expert interpretations.

        Checks for:
        - Significant confidence divergence
        - Low source overlap
        """
        conflicts: List[str] = []

        if len(responses) < 2:
            return conflicts

        # Confidence divergence
        confidences = [r.confidence for r in responses]
        confidence_spread = max(confidences) - min(confidences)

        if confidence_spread > self._config.confidence_divergence_threshold:
            high = max(responses, key=lambda r: r.confidence)
            low = min(responses, key=lambda r: r.confidence)
            conflicts.append(
                f"Divergenza significativa: {high.expert_type} ({high.confidence:.2f}) "
                f"vs {low.expert_type} ({low.confidence:.2f})"
            )

        # Source overlap check
        source_sets = [
            {lb.source_id for lb in r.legal_basis}
            for r in responses
            if r.legal_basis
        ]

        if len(source_sets) >= 2:
            common = source_sets[0].intersection(*source_sets[1:])
            total = source_sets[0].union(*source_sets[1:])
            if total and len(common) / len(total) < self._config.source_overlap_threshold:
                conflicts.append("Fonti giuridiche poco sovrapposte tra expert")

        return conflicts

    def _synthesize_simple(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
    ) -> str:
        """Generate simple synthesis without LLM."""
        sections = ["# Sintesi Multi-Expert\n"]

        sorted_responses = sorted(
            responses,
            key=lambda r: weights.get(r.expert_type, 0),
            reverse=True,
        )

        for resp in sorted_responses:
            w = weights.get(resp.expert_type, 0)
            header = resp.section_header or resp.expert_type.title()
            sections.append(
                f"## {header} (peso: {w:.2f}, confidenza: {resp.confidence:.2f})"
            )
            # Truncate interpretation
            interp = resp.interpretation[:500]
            if len(resp.interpretation) > 500:
                interp += "..."
            sections.append(interp)
            sections.append("")

        sections.append(
            "\n*Nota: Sintesi generata senza AI - combinazione meccanica delle interpretazioni*"
        )

        return "\n".join(sections)

    def _synthesize_consensus(
        self,
        responses: List[ExpertResponse],
        consensus_sources: List[LegalSource],
    ) -> str:
        """Generate consensus-focused synthesis without LLM."""
        sections = ["# Punti di Consenso\n"]

        if consensus_sources:
            sections.append("## Fonti su cui gli Expert concordano:")
            for lb in consensus_sources[:5]:
                excerpt = lb.excerpt[:200] if lb.excerpt else ""
                sections.append(f"- {lb.citation}: {excerpt}...")
        else:
            sections.append("Nessuna fonte citata da più Expert.")

        sections.append("\n## Interpretazioni:")
        for resp in responses:
            header = resp.section_header or resp.expert_type.title()
            interp = resp.interpretation[:300]
            sections.append(f"- **{header}**: {interp}...")

        return "\n".join(sections)

    async def _synthesize_with_llm(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
        focus: str = "balanced",
    ) -> str:
        """Generate synthesis using LLM."""
        if not self.llm_service:
            return self._synthesize_simple(responses, weights)

        prompt = self._build_synthesis_prompt(responses, weights, focus)

        try:
            synthesis = await self.llm_service.generate(
                prompt=prompt,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.max_synthesis_tokens,
            )
            return synthesis

        except Exception as e:
            log.warning(
                "gating_llm_synthesis_error",
                error=str(e),
            )
            return self._synthesize_simple(responses, weights)

    def _build_synthesis_prompt(
        self,
        responses: List[ExpertResponse],
        weights: Dict[str, float],
        focus: str,
    ) -> str:
        """Build prompt for LLM synthesis."""
        expert_sections: List[str] = []

        sorted_responses = sorted(
            responses,
            key=lambda r: weights.get(r.expert_type, 0),
            reverse=True,
        )

        for resp in sorted_responses:
            w = weights.get(resp.expert_type, 0)
            header = resp.section_header or resp.expert_type.upper()
            expert_sections.append(
                f"## {header} (peso: {w:.2f}, confidenza: {resp.confidence:.2f})\n"
                f"{resp.interpretation}\n"
            )
            if resp.legal_basis:
                citations = ", ".join(lb.citation for lb in resp.legal_basis[:3])
                expert_sections.append(f"Fonti: {citations}\n")

        prompt = GATING_SYNTHESIS_PROMPT.format(
            expert_sections="\n".join(expert_sections)
        )

        if focus == "consensus":
            prompt += "\nFOCUS: Evidenzia i punti di accordo tra gli expert."

        return prompt

    def _create_empty_response(self, trace_id: str) -> AggregatedResponse:
        """Create response when no expert outputs are available."""
        return AggregatedResponse(
            synthesis="Nessuna risposta da aggregare - gli Expert non hanno prodotto output.",
            expert_contributions={},
            confidence=0.0,
            trace_id=trace_id,
            metadata={"expert_count": 0},
        )
