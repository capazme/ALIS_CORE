"""
Synthesizer for MERL-T Analysis Pipeline.

Produces coherent final responses from combined Expert outputs.
Takes GatingNetwork aggregated output and generates user-facing response
with profile-aware formatting.

The Synthesizer:
1. Receives AggregatedResponse from GatingNetwork
2. Generates unified response answering user's question
3. Formats output based on user profile
4. Handles Expert disagreements with explicit notes
5. Supports progressive disclosure (collapsed sections)
6. Integrates with RLCF via F7 feedback hook

User profiles:
- consulenza: Summary + key conclusion (⚡ Consultazione)
- ricerca: Summary + expandable Expert sections (📖 Ricerca)
- analisi: Full trace with all Expert outputs (🔍 Analisi)
- contributore: Full trace + feedback options (🎓 Contributore)

Example:
    >>> synthesizer = Synthesizer(llm_service=my_llm)
    >>> final = await synthesizer.synthesize(
    ...     query="Cos'è la risoluzione del contratto?",
    ...     aggregated=gating_output,
    ...     user_profile="ricerca",
    ... )
"""

import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import (
    LegalSource,
    FeedbackHook,
    LLMService,
)
from .gating import AggregatedResponse, ExpertContribution

log = structlog.get_logger()


class UserProfile(str, Enum):
    """User profiles for response formatting."""

    CONSULENZA = "consulenza"  # ⚡ Summary + key conclusion
    RICERCA = "ricerca"  # 📖 Summary + expandable sections
    ANALISI = "analisi"  # 🔍 Full trace
    CONTRIBUTORE = "contributore"  # 🎓 Full trace + feedback


class SynthesisMode(str, Enum):
    """Synthesis mode based on agreement level."""

    CONVERGENT = "convergent"  # Experts agree - unified response
    DIVERGENT = "divergent"  # Experts disagree - present alternatives


@dataclass
class AccordionSection:
    """
    Collapsible section for progressive disclosure.

    Attributes:
        expert_type: Type of expert (literal, systemic, etc.)
        header: Section header
        content: Full content (collapsed by default)
        confidence: Expert confidence score
        is_expanded: Whether section starts expanded
    """

    expert_type: str
    header: str
    content: str
    confidence: float
    is_expanded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "expert_type": self.expert_type,
            "header": self.header,
            "content": self.content,
            "confidence": round(self.confidence, 3),
            "is_expanded": self.is_expanded,
        }


@dataclass
class SynthesizedResponse:
    """
    Final synthesized response for display.

    Attributes:
        main_answer: Primary answer (always visible)
        expert_accordion: Collapsible expert sections
        source_links: Clickable URN links
        confidence_indicator: Overall confidence display
        synthesis_mode: convergent or divergent
        has_disagreement: Whether experts disagreed
        disagreement_note: Explanation of disagreement
        devils_advocate_flag: Whether to activate Devil's Advocate
        feedback_hook: F7 feedback opportunity
        user_profile: Profile used for formatting
        trace_id: Tracing identifier
        execution_time_ms: Processing time
    """

    main_answer: str
    expert_accordion: List[AccordionSection] = field(default_factory=list)
    source_links: List[Dict[str, str]] = field(default_factory=list)
    confidence_indicator: str = ""
    confidence_value: float = 0.5
    synthesis_mode: str = "convergent"
    has_disagreement: bool = False
    disagreement_note: str = ""
    devils_advocate_flag: bool = False
    feedback_hook: Optional[FeedbackHook] = None
    user_profile: str = "ricerca"
    trace_id: str = ""
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API response."""
        return {
            "main_answer": self.main_answer,
            "expert_accordion": [s.to_dict() for s in self.expert_accordion],
            "source_links": self.source_links,
            "confidence_indicator": self.confidence_indicator,
            "confidence_value": round(self.confidence_value, 3),
            "synthesis_mode": self.synthesis_mode,
            "has_disagreement": self.has_disagreement,
            "disagreement_note": self.disagreement_note,
            "devils_advocate_flag": self.devils_advocate_flag,
            "feedback_hook": self.feedback_hook.to_dict() if self.feedback_hook else None,
            "user_profile": self.user_profile,
            "trace_id": self.trace_id,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SynthesizerConfig:
    """
    Configuration for Synthesizer.

    Attributes:
        default_profile: Default user profile
        enable_f7_feedback: Enable F7 feedback for eligible profiles
        disagreement_threshold: Confidence divergence to flag disagreement
        high_confidence_threshold: Threshold for "alta" indicator
        medium_confidence_threshold: Threshold for "media" indicator
        max_sources_display: Maximum source links to display
        llm_temperature: Temperature for LLM synthesis
        max_synthesis_tokens: Max tokens for synthesis
    """

    default_profile: UserProfile = UserProfile.RICERCA
    enable_f7_feedback: bool = True
    disagreement_threshold: float = 0.4
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.5
    max_sources_display: int = 8
    llm_temperature: float = 0.3
    max_synthesis_tokens: int = 2500


# Synthesis prompt template
SYNTHESIS_PROMPT_TEMPLATE = """Sei un esperto giurista italiano. Genera una risposta chiara e completa
basata sulle interpretazioni fornite dagli Expert del sistema MERL-T.

QUERY ORIGINALE:
{query}

INTERPRETAZIONI DEGLI EXPERT:
{expert_interpretations}

FONTI PRINCIPALI:
{sources}

ISTRUZIONI:
1. Rispondi direttamente alla domanda dell'utente
2. Integra le diverse prospettive in modo coerente
3. Mantieni la sequenza Art. 12 Preleggi (letterale → sistematica → teleologica → giurisprudenziale)
4. Cita le fonti normative quando rilevante
5. Se ci sono divergenze tra expert, menzionale brevemente
6. Sii chiaro e professionale

{profile_instruction}

Rispondi in italiano."""


# Profile-specific instructions
PROFILE_INSTRUCTIONS = {
    UserProfile.CONSULENZA: (
        "FORMATO: Fornisci una risposta SINTETICA (max 300 parole) con la conclusione principale. "
        "Evita dettagli tecnici non necessari."
    ),
    UserProfile.RICERCA: (
        "FORMATO: Fornisci una risposta STRUTTURATA con: "
        "1) Risposta diretta, 2) Fondamento normativo, 3) Punti chiave. "
        "Circa 500-800 parole."
    ),
    UserProfile.ANALISI: (
        "FORMATO: Fornisci un'ANALISI COMPLETA che includa: "
        "1) Risposta, 2) Analisi letterale, 3) Analisi sistematica, "
        "4) Ratio legis, 5) Giurisprudenza rilevante. Sii esaustivo."
    ),
    UserProfile.CONTRIBUTORE: (
        "FORMATO: Fornisci un'ANALISI COMPLETA e CRITICA che includa: "
        "1) Risposta, 2) Tutte le prospettive interpretative, "
        "3) Eventuali punti di discussione, 4) Domande aperte. "
        "Questo output sarà oggetto di feedback da parte di esperti."
    ),
}


class Synthesizer:
    """
    Synthesizer for producing final user-facing responses.

    Takes AggregatedResponse from GatingNetwork and generates
    formatted output based on user profile.

    Example:
        >>> synthesizer = Synthesizer(llm_service=my_llm)
        >>> result = await synthesizer.synthesize(
        ...     query="Cos'è la risoluzione?",
        ...     aggregated=gating_output,
        ...     user_profile="analisi",
        ... )
        >>> print(result.main_answer)
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        config: Optional[SynthesizerConfig] = None,
    ):
        """
        Initialize Synthesizer.

        Args:
            llm_service: LLM service for synthesis (optional)
            config: Synthesizer configuration
        """
        self._config = config or SynthesizerConfig()
        self.llm_service = llm_service

        log.info(
            "synthesizer_initialized",
            default_profile=self._config.default_profile.value,
        )

    async def synthesize(
        self,
        query: str,
        aggregated: AggregatedResponse,
        user_profile: Optional[str] = None,
        trace_id: str = "",
    ) -> SynthesizedResponse:
        """
        Synthesize final response from aggregated Expert outputs.

        Args:
            query: Original user query
            aggregated: AggregatedResponse from GatingNetwork
            user_profile: User profile for formatting
            trace_id: Tracing identifier

        Returns:
            SynthesizedResponse ready for display
        """
        start_time = time.time()

        # Resolve profile
        profile = self._resolve_profile(user_profile)

        log.info(
            "synthesizer_processing",
            query=query[:50],
            profile=profile.value,
            experts=len(aggregated.expert_contributions),
            trace_id=trace_id,
        )

        # Detect disagreement
        has_disagreement = self._detect_disagreement(aggregated)
        synthesis_mode = SynthesisMode.DIVERGENT if has_disagreement else SynthesisMode.CONVERGENT

        # Generate main answer
        if self.llm_service:
            main_answer = await self._generate_llm_answer(
                query, aggregated, profile, has_disagreement
            )
        else:
            main_answer = self._generate_fallback_answer(
                aggregated, profile, has_disagreement
            )

        # Build accordion sections based on profile
        accordion = self._build_accordion(aggregated, profile)

        # Build source links
        source_links = self._build_source_links(aggregated)

        # Confidence indicator
        confidence_indicator = self._compute_confidence_indicator(aggregated.confidence)

        # Disagreement note
        disagreement_note = ""
        if has_disagreement:
            disagreement_note = self._generate_disagreement_note(aggregated)

        # F7 feedback hook (only for eligible profiles)
        feedback_hook = None
        if self._config.enable_f7_feedback and profile in [
            UserProfile.ANALISI,
            UserProfile.CONTRIBUTORE,
        ]:
            feedback_hook = FeedbackHook(
                feedback_type="F7",
                expert_type="synthesizer",
                response_id=trace_id,
                enabled=True,
            )

        execution_time = (time.time() - start_time) * 1000

        response = SynthesizedResponse(
            main_answer=main_answer,
            expert_accordion=accordion,
            source_links=source_links,
            confidence_indicator=confidence_indicator,
            confidence_value=aggregated.confidence,
            synthesis_mode=synthesis_mode.value,
            has_disagreement=has_disagreement,
            disagreement_note=disagreement_note,
            devils_advocate_flag=has_disagreement,  # Flag Devil's Advocate
            feedback_hook=feedback_hook,
            user_profile=profile.value,
            trace_id=trace_id,
            execution_time_ms=execution_time,
            metadata={
                "gating_method": aggregated.aggregation_method,
                "expert_count": len(aggregated.expert_contributions),
                "source_count": len(aggregated.combined_legal_basis),
                "conflicts_detected": len(aggregated.conflicts),
            },
        )

        log.info(
            "synthesizer_completed",
            profile=profile.value,
            mode=synthesis_mode.value,
            has_disagreement=has_disagreement,
            execution_time_ms=execution_time,
            trace_id=trace_id,
        )

        return response

    def _resolve_profile(self, user_profile: Optional[str]) -> UserProfile:
        """Resolve user profile from string."""
        if not user_profile:
            return self._config.default_profile

        try:
            return UserProfile(user_profile.lower())
        except ValueError:
            log.warning(
                "synthesizer_unknown_profile",
                profile=user_profile,
                using_default=self._config.default_profile.value,
            )
            return self._config.default_profile

    def _detect_disagreement(self, aggregated: AggregatedResponse) -> bool:
        """
        Detect if experts significantly disagreed.

        Uses conflicts from GatingNetwork plus confidence divergence.
        """
        # Check explicit conflicts
        if aggregated.conflicts:
            return True

        # Check confidence divergence
        if len(aggregated.expert_contributions) >= 2:
            confidences = [
                c.confidence for c in aggregated.expert_contributions.values()
            ]
            divergence = max(confidences) - min(confidences)
            if divergence > self._config.disagreement_threshold:
                return True

        return False

    async def _generate_llm_answer(
        self,
        query: str,
        aggregated: AggregatedResponse,
        profile: UserProfile,
        has_disagreement: bool,
    ) -> str:
        """Generate main answer using LLM."""
        prompt = self._build_synthesis_prompt(
            query, aggregated, profile, has_disagreement
        )

        try:
            answer = await self.llm_service.generate(
                prompt=prompt,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.max_synthesis_tokens,
            )
            return answer

        except Exception as e:
            log.warning(
                "synthesizer_llm_error",
                error=str(e),
            )
            return self._generate_fallback_answer(aggregated, profile, has_disagreement)

    def _build_synthesis_prompt(
        self,
        query: str,
        aggregated: AggregatedResponse,
        profile: UserProfile,
        has_disagreement: bool,
    ) -> str:
        """Build prompt for LLM synthesis."""
        # Format expert interpretations
        expert_sections = []
        for exp_type, contrib in sorted(
            aggregated.expert_contributions.items(),
            key=lambda x: x[1].weight,
            reverse=True,
        ):
            expert_sections.append(
                f"## {exp_type.upper()} (confidenza: {contrib.confidence:.0%})\n"
                f"{contrib.interpretation}"
            )

        # Format sources
        source_text = []
        for lb in aggregated.combined_legal_basis[:6]:
            source_text.append(f"- {lb.citation}")

        # Get profile instruction
        profile_instruction = PROFILE_INSTRUCTIONS.get(profile, "")

        if has_disagreement:
            profile_instruction += (
                "\n\nNOTA: Gli Expert hanno rilevato alcune DIVERGENZE interpretative. "
                "Menziona brevemente le diverse posizioni nella risposta."
            )

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            query=query,
            expert_interpretations="\n\n".join(expert_sections),
            sources="\n".join(source_text) if source_text else "Nessuna fonte specifica",
            profile_instruction=profile_instruction,
        )

        return prompt

    def _generate_fallback_answer(
        self,
        aggregated: AggregatedResponse,
        profile: UserProfile,
        has_disagreement: bool,
    ) -> str:
        """Generate answer without LLM."""
        parts = []

        # Use gating synthesis if available
        if aggregated.synthesis:
            parts.append(aggregated.synthesis)
        else:
            # Build from contributions
            parts.append("# Risposta\n")
            for exp_type, contrib in sorted(
                aggregated.expert_contributions.items(),
                key=lambda x: x[1].weight,
                reverse=True,
            ):
                if contrib.weight > 0.15 or profile in [UserProfile.ANALISI, UserProfile.CONTRIBUTORE]:
                    parts.append(f"\n**{exp_type.title()}** (confidenza: {contrib.confidence:.0%})")
                    interp = contrib.interpretation[:400] if profile == UserProfile.CONSULENZA else contrib.interpretation
                    if len(contrib.interpretation) > 400 and profile == UserProfile.CONSULENZA:
                        interp += "..."
                    parts.append(interp)

        if has_disagreement:
            parts.append(
                "\n\n*Nota: Sono state rilevate divergenze tra le interpretazioni degli Expert.*"
            )

        parts.append(
            "\n\n*Risposta generata senza sintesi AI - combinazione delle interpretazioni expert.*"
        )

        return "\n".join(parts)

    def _build_accordion(
        self,
        aggregated: AggregatedResponse,
        profile: UserProfile,
    ) -> List[AccordionSection]:
        """Build collapsible expert sections based on profile."""
        sections = []

        # Consulenza: no accordion (summary only)
        if profile == UserProfile.CONSULENZA:
            return sections

        # Sort by weight
        sorted_contribs = sorted(
            aggregated.expert_contributions.items(),
            key=lambda x: x[1].weight,
            reverse=True,
        )

        for exp_type, contrib in sorted_contribs:
            # Map expert type to header
            headers = {
                "literal": "Interpretazione Letterale",
                "systemic": "Interpretazione Sistematica",
                "principles": "Interpretazione Teleologica",
                "precedent": "Giurisprudenza",
            }
            header = headers.get(exp_type, exp_type.title())

            # Expand based on profile
            is_expanded = profile in [UserProfile.ANALISI, UserProfile.CONTRIBUTORE]

            sections.append(
                AccordionSection(
                    expert_type=exp_type,
                    header=header,
                    content=contrib.interpretation,
                    confidence=contrib.confidence,
                    is_expanded=is_expanded,
                )
            )

        return sections

    def _build_source_links(
        self,
        aggregated: AggregatedResponse,
    ) -> List[Dict[str, str]]:
        """Build clickable source links."""
        links = []
        seen = set()

        for lb in aggregated.combined_legal_basis[: self._config.max_sources_display]:
            if lb.source_id in seen:
                continue
            seen.add(lb.source_id)

            links.append({
                "source_id": lb.source_id,
                "citation": lb.citation,
                "source_type": lb.source_type,
                "urn": lb.source_id if lb.source_id.startswith("urn:") else "",
            })

        return links

    def _compute_confidence_indicator(self, confidence: float) -> str:
        """Compute human-readable confidence indicator."""
        if confidence >= self._config.high_confidence_threshold:
            return "alta"
        elif confidence >= self._config.medium_confidence_threshold:
            return "media"
        else:
            return "bassa"

    def _generate_disagreement_note(self, aggregated: AggregatedResponse) -> str:
        """Generate note explaining disagreement."""
        parts = ["**Divergenze rilevate:**"]

        if aggregated.conflicts:
            for conflict in aggregated.conflicts[:3]:
                parts.append(f"- {conflict}")
        else:
            # Compute from confidence divergence
            if len(aggregated.expert_contributions) >= 2:
                sorted_exp = sorted(
                    aggregated.expert_contributions.items(),
                    key=lambda x: x[1].confidence,
                    reverse=True,
                )
                high_exp, high_contrib = sorted_exp[0]
                low_exp, low_contrib = sorted_exp[-1]
                parts.append(
                    f"- Differenza di confidenza: {high_exp} ({high_contrib.confidence:.0%}) "
                    f"vs {low_exp} ({low_contrib.confidence:.0%})"
                )

        parts.append(
            "\n*Si consiglia di valutare criticamente le diverse prospettive.*"
        )

        return "\n".join(parts)
