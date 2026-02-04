"""
Tests for Expert orchestration components.

Tests per:
- ExpertRouter: Routing delle query
- GatingNetwork: Aggregazione risposte
- MultiExpertOrchestrator: Orchestrazione completa
"""

import pytest
from typing import List

from merlt.experts import (
    ExpertRouter,
    RoutingDecision,
    GatingNetwork,
    AggregatedResponse,
    MultiExpertOrchestrator,
    OrchestratorConfig,
    ExpertContext,
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    # Nuove classi per Synthesizer
    AdaptiveSynthesizer,
    SynthesisConfig,
    SynthesisResult,
    SynthesisMode,
)
from merlt.experts.gating import GatingConfig, AggregationMethod, ExpertContribution


# ============================================================================
# ExpertRouter Tests
# ============================================================================

class TestRoutingDecision:
    """Test per RoutingDecision."""

    def test_create(self):
        """Crea decisione."""
        decision = RoutingDecision(
            expert_weights={"literal": 0.6, "systemic": 0.3, "precedent": 0.1},
            query_type="definitional",
            confidence=0.8
        )

        assert decision.query_type == "definitional"
        assert decision.confidence == 0.8
        assert decision.expert_weights["literal"] == 0.6

    def test_get_selected_experts(self):
        """Seleziona expert sopra soglia."""
        decision = RoutingDecision(
            expert_weights={"literal": 0.6, "systemic": 0.3, "precedent": 0.1}
        )

        # Soglia default 0.3
        selected = decision.get_selected_experts(threshold=0.3)

        assert len(selected) == 2
        assert selected[0] == ("literal", 0.6)
        assert selected[1] == ("systemic", 0.3)

    def test_get_selected_experts_low_threshold(self):
        """Seleziona con soglia bassa."""
        decision = RoutingDecision(
            expert_weights={"literal": 0.6, "systemic": 0.3, "precedent": 0.1}
        )

        selected = decision.get_selected_experts(threshold=0.05)

        assert len(selected) == 3


class TestExpertRouter:
    """Test per ExpertRouter."""

    def test_init(self):
        """Inizializza router."""
        router = ExpertRouter()

        assert router.query_weights is not None
        assert "definitional" in router.query_weights
        assert "general" in router.query_weights

    @pytest.mark.asyncio
    async def test_route_definitional(self):
        """Route query definitional."""
        router = ExpertRouter()
        context = ExpertContext(query_text="Cos'è la legittima difesa?")

        decision = await router.route(context)

        assert decision.query_type == "definitional"
        # Literal dovrebbe avere peso alto per query definitional
        assert decision.expert_weights["literal"] > 0.3

    @pytest.mark.asyncio
    async def test_route_constitutional(self):
        """Route query constitutional."""
        router = ExpertRouter()
        context = ExpertContext(
            query_text="Quali diritti fondamentali sono coinvolti nell'art. 3 Cost.?"
        )

        decision = await router.route(context)

        assert decision.query_type == "constitutional"
        # Principles dovrebbe avere peso alto
        assert decision.expert_weights["principles"] > 0.3

    @pytest.mark.asyncio
    async def test_route_jurisprudential(self):
        """Route query giurisprudenziale."""
        router = ExpertRouter()
        context = ExpertContext(
            query_text="Qual è l'orientamento della Cassazione sul danno biologico?"
        )

        decision = await router.route(context)

        assert decision.query_type == "jurisprudential"
        assert decision.expert_weights["precedent"] > 0.4

    @pytest.mark.asyncio
    async def test_route_systemic(self):
        """Route query sistematica."""
        router = ExpertRouter()
        context = ExpertContext(
            query_text="Qual è la relazione tra art. 1453 e 1455 c.c. nel sistema?"
        )

        decision = await router.route(context)

        # Dovrebbe riconoscere come systemic
        assert decision.expert_weights["systemic"] > 0.2

    @pytest.mark.asyncio
    async def test_route_general(self):
        """Route query generica."""
        router = ExpertRouter()
        context = ExpertContext(query_text="Spiegami questo articolo")

        decision = await router.route(context)

        # Query generica = distribuzione bilanciata
        assert decision.query_type == "general"
        assert all(w > 0.1 for w in decision.expert_weights.values())

    @pytest.mark.asyncio
    async def test_adjust_for_entities(self):
        """Aggiusta pesi per entità."""
        router = ExpertRouter()
        context = ExpertContext(
            query_text="test",
            entities={
                "norm_references": ["urn:norma:cc:art1321"],
                "legal_concepts": ["principio di buona fede"]
            }
        )

        decision = await router.route(context)

        # Con riferimenti normativi, literal dovrebbe essere boosted
        assert decision.expert_weights["literal"] > 0


# ============================================================================
# GatingNetwork Tests
# ============================================================================

class TestAggregatedResponse:
    """Test per AggregatedResponse."""

    def test_create(self):
        """Crea response aggregata."""
        response = AggregatedResponse(
            synthesis="Sintesi test",
            expert_contributions={
                "literal": ExpertContribution(
                    expert_type="literal",
                    interpretation="Test",
                    confidence=0.8,
                    weight=1.0,
                    weighted_confidence=0.8,
                )
            },
            confidence=0.8
        )

        assert response.synthesis == "Sintesi test"
        assert response.confidence == 0.8

    def test_to_dict(self):
        """Serializza in dizionario."""
        response = AggregatedResponse(
            synthesis="Test",
            expert_contributions={},
            conflicts=["Conflitto 1"]
        )

        data = response.to_dict()

        assert data["synthesis"] == "Test"
        assert data["conflicts"] == ["Conflitto 1"]


class TestGatingNetwork:
    """Test per GatingNetwork."""

    def test_init_default(self):
        """Inizializza con default."""
        gating = GatingNetwork()

        assert gating._config.method == AggregationMethod.WEIGHTED_AVERAGE

    def test_init_custom_method(self):
        """Inizializza con metodo custom."""
        gating = GatingNetwork(config=GatingConfig(method=AggregationMethod.ENSEMBLE))

        assert gating._config.method == AggregationMethod.ENSEMBLE

    @pytest.mark.asyncio
    async def test_aggregate_empty(self):
        """Aggrega lista vuota."""
        gating = GatingNetwork()

        result = await gating.aggregate([], {})

        assert result.confidence == 0.0
        assert "Nessuna risposta" in result.synthesis

    @pytest.mark.asyncio
    async def test_aggregate_single(self):
        """Aggrega singola risposta."""
        gating = GatingNetwork()

        responses = [
            ExpertResponse(
                expert_type="literal",
                interpretation="Test interpretation",
                confidence=0.8
            )
        ]

        result = await gating.aggregate(
            responses,
            {"literal": 1.0},
            trace_id="test"
        )

        assert result.confidence > 0
        assert "literal" in result.expert_contributions

    @pytest.mark.asyncio
    async def test_aggregate_multiple(self):
        """Aggrega multiple risposte."""
        gating = GatingNetwork()

        responses = [
            ExpertResponse(
                expert_type="literal",
                interpretation="Interpretazione letterale",
                confidence=0.8,
                legal_basis=[
                    LegalSource(source_type="norm", source_id="s1", citation="Art. 1")
                ]
            ),
            ExpertResponse(
                expert_type="systemic",
                interpretation="Interpretazione sistematica",
                confidence=0.7,
                legal_basis=[
                    LegalSource(source_type="norm", source_id="s2", citation="Art. 2")
                ]
            )
        ]

        result = await gating.aggregate(
            responses,
            {"literal": 0.6, "systemic": 0.4},
            trace_id="test"
        )

        assert len(result.expert_contributions) == 2
        assert result.confidence > 0
        assert len(result.combined_legal_basis) == 2

    @pytest.mark.asyncio
    async def test_aggregate_best_confidence(self):
        """Aggrega con metodo best_confidence."""
        gating = GatingNetwork(config=GatingConfig(method=AggregationMethod.BEST_CONFIDENCE))

        responses = [
            ExpertResponse(expert_type="literal", interpretation="Low", confidence=0.3),
            ExpertResponse(expert_type="systemic", interpretation="High", confidence=0.9),
        ]

        result = await gating.aggregate(responses, {}, trace_id="test")

        assert result.aggregation_method == "best_confidence"
        assert result.synthesis == "High"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_aggregate_ensemble(self):
        """Aggrega con metodo ensemble."""
        gating = GatingNetwork(config=GatingConfig(method=AggregationMethod.ENSEMBLE))

        responses = [
            ExpertResponse(expert_type="literal", interpretation="Letterale", confidence=0.8),
            ExpertResponse(expert_type="principles", interpretation="Teleologica", confidence=0.7),
        ]

        result = await gating.aggregate(
            responses,
            {"literal": 0.5, "principles": 0.5},
            trace_id="test"
        )

        assert result.aggregation_method == "ensemble"
        assert "Literal" in result.synthesis
        assert "Principles" in result.synthesis

    @pytest.mark.asyncio
    async def test_detect_conflicts(self):
        """Rileva conflitti."""
        gating = GatingNetwork()

        responses = [
            ExpertResponse(
                expert_type="literal",
                interpretation="A",
                confidence=0.9,
                legal_basis=[LegalSource("norm", "s1", "Art. 1")]
            ),
            ExpertResponse(
                expert_type="systemic",
                interpretation="B",
                confidence=0.3,  # Grande divergenza
                legal_basis=[LegalSource("norm", "s2", "Art. 2")]  # Fonti diverse
            ),
        ]

        result = await gating.aggregate(responses, {"literal": 0.5, "systemic": 0.5})

        # Dovrebbe rilevare divergenza di confidenza
        assert len(result.conflicts) > 0


# ============================================================================
# MultiExpertOrchestrator Tests
# ============================================================================

class TestOrchestratorConfig:
    """Test per OrchestratorConfig."""

    def test_defaults(self):
        """Verifica default."""
        config = OrchestratorConfig()

        assert config.selection_threshold == 0.2
        assert config.max_experts == 4
        assert config.parallel_execution is True
        assert config.timeout_seconds == 30.0

    def test_custom(self):
        """Config custom."""
        config = OrchestratorConfig(
            max_experts=2,
            parallel_execution=False,
            timeout_seconds=60.0
        )

        assert config.max_experts == 2
        assert config.parallel_execution is False
        assert config.timeout_seconds == 60.0


def _create_synthesizer() -> AdaptiveSynthesizer:
    """Helper per creare AdaptiveSynthesizer per test."""
    config = SynthesisConfig(
        convergent_threshold=0.5,
        resolvability_weight=0.3,
        include_disagreement_explanation=True,
        max_alternatives=3,
    )
    return AdaptiveSynthesizer(config=config, ai_service=None)


class TestMultiExpertOrchestrator:
    """Test per MultiExpertOrchestrator."""

    def test_init_default(self):
        """Inizializza con synthesizer (OBBLIGATORIO)."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        assert len(orchestrator.list_experts()) == 4
        assert "literal" in orchestrator.list_experts()
        assert "systemic" in orchestrator.list_experts()

    def test_init_with_config(self):
        """Inizializza con config."""
        synthesizer = _create_synthesizer()
        config = OrchestratorConfig(max_experts=2)
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer, config=config)

        assert orchestrator.config.max_experts == 2

    def test_get_expert(self):
        """Ottiene expert per tipo."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        literal = orchestrator.get_expert("literal")
        assert literal is not None
        assert literal.expert_type == "literal"

        none = orchestrator.get_expert("nonexistent")
        assert none is None

    @pytest.mark.asyncio
    async def test_process_simple(self):
        """Processa query semplice - ritorna SynthesisResult."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        response = await orchestrator.process("Cos'è la legittima difesa?")

        assert isinstance(response, SynthesisResult)
        assert response.execution_time_ms > 0
        assert len(response.expert_contributions) > 0

    @pytest.mark.asyncio
    async def test_process_with_entities(self):
        """Processa con entità."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        response = await orchestrator.process(
            query="Art. 52 c.p.",
            entities={
                "norm_references": ["urn:norma:cp:art52"],
                "legal_concepts": ["legittima difesa"]
            }
        )

        assert isinstance(response, SynthesisResult)

    @pytest.mark.asyncio
    async def test_process_with_routing(self):
        """Processa e ritorna routing."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        response, routing = await orchestrator.process_with_routing(
            "Qual è l'orientamento della Cassazione?"
        )

        assert isinstance(response, SynthesisResult)
        assert isinstance(routing, RoutingDecision)
        assert routing.query_type == "jurisprudential"

    @pytest.mark.asyncio
    async def test_run_single_expert(self):
        """Esegue singolo expert."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        response = await orchestrator.run_single_expert(
            "literal",
            "Test query"
        )

        assert isinstance(response, ExpertResponse)
        assert response.expert_type == "literal"

    @pytest.mark.asyncio
    async def test_run_single_expert_not_found(self):
        """Expert non trovato."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        response = await orchestrator.run_single_expert(
            "nonexistent",
            "Test"
        )

        assert response.confidence == 0.0
        assert "non trovato" in response.interpretation

    @pytest.mark.asyncio
    async def test_max_experts_limit(self):
        """Rispetta limite max_experts."""
        synthesizer = _create_synthesizer()
        config = OrchestratorConfig(max_experts=2)
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer, config=config)

        response = await orchestrator.process("Test generale")

        # Dovrebbe avere al massimo 2 contributi
        assert len(response.expert_contributions) <= 2

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Esecuzione parallela."""
        synthesizer = _create_synthesizer()
        config = OrchestratorConfig(parallel_execution=True)
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer, config=config)

        response = await orchestrator.process("Test")

        # Verifica che ha eseguito (tempo ragionevole per parallelo)
        assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_synthesis_mode_in_result(self):
        """Verifica che SynthesisResult includa synthesis_mode."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        response = await orchestrator.process("Cos'è la legittima difesa?")

        assert isinstance(response, SynthesisResult)
        assert response.mode in [SynthesisMode.CONVERGENT, SynthesisMode.DIVERGENT]

    @pytest.mark.asyncio
    async def test_disagreement_analysis_in_result(self):
        """Verifica che SynthesisResult includa disagreement_analysis."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        response = await orchestrator.process("Cos'è la responsabilità extracontrattuale?")

        assert isinstance(response, SynthesisResult)
        # disagreement_analysis può essere None o DisagreementAnalysis
        # ma il campo deve esistere
        assert hasattr(response, "disagreement_analysis")


# ============================================================================
# Integration Tests
# ============================================================================

class TestOrchestrationIntegration:
    """Test di integrazione per l'intero sistema."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Pipeline completo: query -> routing -> experts -> synthesizer."""
        # Usa threshold basso per avere più expert
        synthesizer = _create_synthesizer()
        config = OrchestratorConfig(selection_threshold=0.1)
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer, config=config)

        # Query complessa che dovrebbe attivare più expert
        response = await orchestrator.process(
            query="Cos'è il danno ingiusto secondo l'art. 2043 c.c.?",
            entities={
                "norm_references": ["urn:norma:cc:art2043"],
                "legal_concepts": ["danno ingiusto", "responsabilità civile"]
            }
        )

        # Verifica response - ora è SynthesisResult
        assert isinstance(response, SynthesisResult)
        assert response.synthesis != ""
        assert len(response.expert_contributions) >= 1  # Almeno 1
        assert response.execution_time_ms > 0
        # Nuovi campi del SynthesisResult
        assert response.mode in [SynthesisMode.CONVERGENT, SynthesisMode.DIVERGENT]

    @pytest.mark.asyncio
    async def test_different_query_types(self):
        """Verifica routing per diversi tipi di query."""
        synthesizer = _create_synthesizer()
        orchestrator = MultiExpertOrchestrator(synthesizer=synthesizer)

        queries = [
            ("Cos'è il contratto?", "definitional"),
            ("Orientamento Cassazione sul danno", "jurisprudential"),
            ("Diritti costituzionali alla salute", "constitutional"),
        ]

        for query, expected_type in queries:
            response, routing = await orchestrator.process_with_routing(query)

            assert routing.query_type == expected_type
            assert response.confidence > 0


# ============================================================================
# AdaptiveSynthesizer Tests
# ============================================================================

class TestAdaptiveSynthesizer:
    """Test per AdaptiveSynthesizer."""

    def test_init_default(self):
        """Inizializza con default."""
        synthesizer = AdaptiveSynthesizer()

        assert synthesizer.config is not None
        assert synthesizer.config.convergent_threshold > 0

    def test_init_with_config(self):
        """Inizializza con config custom."""
        config = SynthesisConfig(
            convergent_threshold=0.7,
            resolvability_weight=0.5,
            include_disagreement_explanation=False,
        )
        synthesizer = AdaptiveSynthesizer(config=config)

        assert synthesizer.config.convergent_threshold == 0.7
        assert synthesizer.config.resolvability_weight == 0.5

    @pytest.mark.asyncio
    async def test_synthesize_empty_responses(self):
        """Sintetizza lista vuota."""
        synthesizer = _create_synthesizer()

        result = await synthesizer.synthesize(
            query="Test",
            responses=[],
            weights={},
        )

        assert isinstance(result, SynthesisResult)
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_single_response(self):
        """Sintetizza singola risposta - convergent."""
        synthesizer = _create_synthesizer()

        responses = [
            ExpertResponse(
                expert_type="literal",
                interpretation="Test interpretation",
                confidence=0.8,
            )
        ]

        result = await synthesizer.synthesize(
            query="Test",
            responses=responses,
            weights={"literal": 1.0},
        )

        assert isinstance(result, SynthesisResult)
        assert result.mode == SynthesisMode.CONVERGENT
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_synthesize_multiple_agreeing_responses(self):
        """Sintetizza risposte concordi - dovrebbe essere convergent."""
        synthesizer = _create_synthesizer()

        responses = [
            ExpertResponse(
                expert_type="literal",
                interpretation="La legittima difesa è...",
                confidence=0.8,
            ),
            ExpertResponse(
                expert_type="systemic",
                interpretation="La legittima difesa nel sistema...",
                confidence=0.75,
            ),
        ]

        result = await synthesizer.synthesize(
            query="Cos'è la legittima difesa?",
            responses=responses,
            weights={"literal": 0.6, "systemic": 0.4},
        )

        assert isinstance(result, SynthesisResult)
        # Con risposte simili, dovrebbe essere convergent
        assert len(result.expert_contributions) == 2

    @pytest.mark.asyncio
    async def test_synthesis_result_has_all_fields(self):
        """Verifica che SynthesisResult abbia tutti i campi richiesti."""
        synthesizer = _create_synthesizer()

        responses = [
            ExpertResponse(
                expert_type="literal",
                interpretation="Test",
                confidence=0.8,
            )
        ]

        result = await synthesizer.synthesize(
            query="Test",
            responses=responses,
            weights={"literal": 1.0},
        )

        # Verifica tutti i campi di SynthesisResult
        assert hasattr(result, "synthesis")
        assert hasattr(result, "mode")
        assert hasattr(result, "confidence")
        assert hasattr(result, "expert_contributions")
        assert hasattr(result, "combined_legal_basis")
        assert hasattr(result, "disagreement_analysis")
        assert hasattr(result, "alternatives")
        assert hasattr(result, "explanation")
        assert hasattr(result, "execution_time_ms")
