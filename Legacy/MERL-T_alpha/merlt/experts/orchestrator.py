"""
Multi-Expert Orchestrator
==========================

Orchestratore centrale per il sistema multi-expert.

Il MultiExpertOrchestrator coordina:
1. ExpertRouter: Selezione degli Expert
2. Expert: Esecuzione parallela/sequenziale
3. AdaptiveSynthesizer: Sintesi adattiva con disagreement detection

Pipeline completa:
    Query → Router → [Experts in parallelo] → AdaptiveSynthesizer → SynthesisResult

Il sistema decide automaticamente se:
- CONVERGENT: Integrare le prospettive in risposta unificata
- DIVERGENT: Presentare alternative con spiegazione del conflitto

Esempio:
    >>> from merlt.experts import MultiExpertOrchestrator
    >>>
    >>> orchestrator = MultiExpertOrchestrator(ai_service=openrouter)
    >>> result = await orchestrator.process("Cos'è la legittima difesa?")
    >>> print(result.synthesis)
    >>> print(f"Mode: {result.mode}")
"""

import structlog
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass
from datetime import datetime

from merlt.experts.base import BaseExpert, ExpertContext, ExpertResponse
from merlt.experts.router import ExpertRouter, RoutingDecision
from merlt.experts.synthesizer import AdaptiveSynthesizer, SynthesisConfig, SynthesisResult, SynthesisMode

# Import opzionale per HybridExpertRouter (neural gating)
try:
    from merlt.experts.neural_gating import (
        HybridExpertRouter,
        HybridRoutingDecision,
        ExpertGatingMLP,
    )
    NEURAL_GATING_AVAILABLE = True
except ImportError:
    NEURAL_GATING_AVAILABLE = False
    HybridExpertRouter = None
    HybridRoutingDecision = None
from merlt.experts.literal import LiteralExpert
from merlt.experts.systemic import SystemicExpert
from merlt.experts.principles import PrinciplesExpert
from merlt.experts.precedent import PrecedentExpert
from merlt.experts.query_analyzer import analyze_query, enrich_context
from merlt.tools import BaseTool
from merlt.rlcf.execution_trace import ExecutionTrace, Action

log = structlog.get_logger()


@dataclass
class OrchestratorConfig:
    """
    Configurazione dell'orchestratore.

    Attributes:
        selection_threshold: Soglia minima per selezionare un expert
        max_experts: Numero massimo di expert da invocare
        parallel_execution: Se eseguire in parallelo
        timeout_seconds: Timeout per ogni expert

    Note:
        La modalità di sintesi (convergent/divergent) è gestita
        automaticamente da AdaptiveSynthesizer tramite disagreement detection.
    """
    selection_threshold: float = 0.2
    max_experts: int = 4
    parallel_execution: bool = True
    timeout_seconds: float = 30.0


class MultiExpertOrchestrator:
    """
    Orchestratore per il sistema multi-expert interpretativo.

    Coordina il flusso completo:
    1. Riceve una query
    2. Il Router decide quali Expert invocare
    3. Gli Expert analizzano in parallelo
    4. AdaptiveSynthesizer sintetizza con disagreement detection
    5. Ritorna SynthesisResult (convergent o divergent)

    Esempio:
        >>> # Setup base
        >>> orchestrator = MultiExpertOrchestrator(
        ...     synthesizer=AdaptiveSynthesizer(ai_service=openrouter)
        ... )
        >>> result = await orchestrator.process("Art. 52 c.p.")
        >>> print(result.mode)  # "convergent" o "divergent"
        >>>
        >>> # Con AI service e tools
        >>> tools = [SemanticSearchTool(...), GraphSearchTool(...)]
        >>> synthesizer = AdaptiveSynthesizer(ai_service=openrouter)
        >>> orchestrator = MultiExpertOrchestrator(
        ...     tools=tools,
        ...     ai_service=openrouter_service,
        ...     synthesizer=synthesizer,
        ...     config=OrchestratorConfig(max_experts=3)
        ... )
        >>> result = await orchestrator.process(
        ...     query="Interpretazione della legittima difesa",
        ...     entities={"norm_references": ["urn:norma:cp:art52"]}
        ... )
        >>> if result.mode == SynthesisMode.DIVERGENT:
        ...     for alt in result.alternatives:
        ...         print(f"- {alt['expert']}: {alt['position'][:50]}")
    """

    # Mapping tipo -> classe Expert
    EXPERT_CLASSES: Dict[str, Type[BaseExpert]] = {
        "literal": LiteralExpert,
        "systemic": SystemicExpert,
        "principles": PrinciplesExpert,
        "precedent": PrecedentExpert,
    }

    def __init__(
        self,
        synthesizer: AdaptiveSynthesizer,
        tools: Optional[List[BaseTool]] = None,
        ai_service: Any = None,
        config: Optional[OrchestratorConfig] = None,
        router: Optional[ExpertRouter] = None,
        hybrid_router: Optional["HybridExpertRouter"] = None,
        gating_policy: Optional[Any] = None,
        embedding_service: Optional[Any] = None
    ):
        """
        Inizializza l'orchestratore.

        Args:
            synthesizer: AdaptiveSynthesizer per sintesi (OBBLIGATORIO)
            tools: Tools condivisi da tutti gli Expert
            ai_service: Servizio AI per LLM
            config: Configurazione orchestratore
            router: Router regex tradizionale (opzionale, usato come fallback)
            hybrid_router: HybridExpertRouter per neural+regex routing (preferito se presente)
            gating_policy: GatingPolicy per neural routing (legacy, usa hybrid_router invece)
            embedding_service: Servizio per encoding query (richiesto per neural routing)
        """
        self.synthesizer = synthesizer
        self.tools = tools or []
        self.ai_service = ai_service
        self.config = config or OrchestratorConfig()
        self.gating_policy = gating_policy
        self.embedding_service = embedding_service

        # Router: preferisce HybridExpertRouter se disponibile
        self.hybrid_router = hybrid_router
        self.router = router or ExpertRouter()

        # Determina strategia di routing
        if self.hybrid_router is not None:
            self._routing_strategy = "hybrid"
        elif self.gating_policy is not None and self.embedding_service is not None:
            self._routing_strategy = "neural_policy"
        else:
            self._routing_strategy = "regex"

        # Inizializza Expert
        self._experts: Dict[str, BaseExpert] = {}
        self._init_experts()

        log.info(
            "MultiExpertOrchestrator initialized",
            experts=list(self._experts.keys()),
            tools=len(self.tools),
            has_ai=ai_service is not None,
            routing_strategy=self._routing_strategy,
            has_hybrid_router=hybrid_router is not None,
            synthesis_mode=self.synthesizer.config.mode.value
        )

    def _init_experts(self):
        """Inizializza tutti gli Expert."""
        for expert_type, expert_class in self.EXPERT_CLASSES.items():
            self._experts[expert_type] = expert_class(
                tools=self.tools,
                ai_service=self.ai_service
            )

    async def _apply_gating_policy(
        self,
        context: ExpertContext,
        trace: ExecutionTrace
    ) -> Dict[str, float]:
        """
        Applica GatingPolicy per calcolare pesi expert.

        Args:
            context: Contesto della query
            trace: ExecutionTrace per tracciare azioni

        Returns:
            Dict con pesi expert {expert_type: weight}
        """
        if not self.gating_policy:
            raise ValueError("GatingPolicy non fornita")

        if not self.embedding_service:
            raise ValueError("EmbeddingService richiesto per GatingPolicy")

        # Import torch per device handling
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch richiesto per GatingPolicy")

        # 1. Encode query
        query_embedding = await self.embedding_service.encode_query_async(
            context.query_text
        )

        # 2. Converti in tensor (usa cpu per evitare problemi MPS in test)
        query_tensor = torch.tensor(
            query_embedding,
            dtype=torch.float32,
            device="cpu"
        ).unsqueeze(0)  # [1, input_dim]

        # 3. Forward pass nella policy
        self.gating_policy.eval()  # Eval mode
        with torch.no_grad():
            weights_tensor, log_probs_tensor = self.gating_policy.forward(query_tensor)

        # 4. Converti in dict
        weights_numpy = weights_tensor.squeeze(0).cpu().numpy()
        log_probs_numpy = log_probs_tensor.squeeze(0).cpu().numpy()

        expert_types = list(self.EXPERT_CLASSES.keys())
        weights_dict = {}

        for i, expert_type in enumerate(expert_types):
            weight = float(weights_numpy[i])
            log_prob = float(log_probs_numpy[i])

            weights_dict[expert_type] = weight

            # Traccia azione di selezione expert con query_embedding per backprop
            trace.add_expert_selection(
                expert_type=expert_type,
                weight=weight,
                log_prob=log_prob,
                metadata={
                    "source": "gating_policy",
                    "query_embedding_dim": len(query_embedding),
                    "query_embedding": query_embedding.tolist(),  # Per REINFORCE backprop
                    "action_index": i  # Indice dell'expert per loss calculation
                }
            )

        log.info(
            "GatingPolicy applied",
            trace_id=context.trace_id,
            weights=weights_dict,
            total_log_prob=trace.total_log_prob
        )

        return weights_dict

    async def process(
        self,
        query: str,
        entities: Optional[Dict[str, List[str]]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        return_trace: bool = False
    ) -> SynthesisResult:
        """
        Processa una query attraverso il sistema multi-expert.

        Args:
            query: Query in linguaggio naturale
            entities: Entità estratte (norm_references, legal_concepts)
            retrieved_chunks: Chunks già recuperati
            metadata: Metadati aggiuntivi
            return_trace: Se True, ritorna (result, trace) invece di solo result

        Returns:
            SynthesisResult con sintesi finale (o tuple se return_trace=True)
            - mode: "convergent" o "divergent"
            - synthesis: Testo sintesi
            - disagreement_analysis: Analisi del disagreement
            - alternatives: Liste alternative (solo in divergent mode)
        """
        import time
        start_time = time.time()

        trace_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Inizializza ExecutionTrace
        trace = ExecutionTrace(
            query_id=trace_id,
            metadata={
                "query": query,
                "has_gating_policy": self.gating_policy is not None
            }
        )

        log.info(f"Processing query", query=query[:50], trace_id=trace_id)

        # Step 1: Analizza query per estrarre entità
        query_analysis = analyze_query(query)

        log.info(
            "Query analyzed",
            articles=query_analysis.article_numbers,
            concepts=query_analysis.legal_concepts[:3] if query_analysis.legal_concepts else [],
            query_type=query_analysis.query_type,
            trace_id=trace_id
        )

        # Step 2: Costruisci context con entità estratte
        # Merge provided entities with extracted ones
        merged_entities = entities or {}
        if query_analysis.norm_references:
            merged_entities["norm_references"] = query_analysis.norm_references
        if query_analysis.legal_concepts:
            merged_entities["legal_concepts"] = query_analysis.legal_concepts
        if query_analysis.article_numbers:
            merged_entities["article_numbers"] = query_analysis.article_numbers

        context = ExpertContext(
            query_text=query,
            entities=merged_entities,
            retrieved_chunks=retrieved_chunks or [],
            metadata={
                **(metadata or {}),
                "query_analysis": {
                    "query_type": query_analysis.query_type,
                    "confidence": query_analysis.confidence,
                    "article_numbers": query_analysis.article_numbers,
                    "legal_concepts": query_analysis.legal_concepts,
                }
            },
            trace_id=trace_id
        )

        # Step 3: Routing - strategia basata su configurazione
        routing_used = "unknown"
        neural_confidence = None

        if self._routing_strategy == "hybrid" and self.hybrid_router is not None:
            # Hybrid routing: neural + regex fallback
            routing_decision = await self.hybrid_router.route(context)
            routing_used = "neural" if routing_decision.neural_used else "regex_fallback"
            neural_confidence = routing_decision.neural_confidence

            log.info(
                "Hybrid routing decision",
                neural_used=routing_decision.neural_used,
                neural_confidence=routing_decision.neural_confidence,
                query_type=routing_decision.query_type,
                weights=routing_decision.expert_weights
            )

            # Traccia azione di routing
            trace.add_action(Action(
                action_type="routing",
                parameters={
                    "strategy": "hybrid",
                    "neural_used": routing_decision.neural_used,
                    "neural_confidence": routing_decision.neural_confidence,
                    "query_type": routing_decision.query_type,
                },
                log_prob=-0.1 if routing_decision.neural_used else -0.5,  # Neural routing ha log_prob più alto
            ))

            # Seleziona Expert
            selected_experts = routing_decision.get_selected_experts(
                threshold=self.config.selection_threshold
            )[:self.config.max_experts]

        elif self._routing_strategy == "neural_policy" and self.gating_policy and self.embedding_service:
            # Neural routing con GatingPolicy (legacy)
            weights = await self._apply_gating_policy(context, trace)
            routing_used = "neural_policy"

            log.info(
                "Neural routing via GatingPolicy",
                weights=weights,
                trace_actions=trace.num_actions
            )

            # Converti weights in selected_experts format
            selected_experts = [
                (expert_type, weight)
                for expert_type, weight in weights.items()
                if weight >= self.config.selection_threshold
            ][:self.config.max_experts]

            if not selected_experts:
                # Fallback: seleziona top expert
                top_expert = max(weights.items(), key=lambda x: x[1])
                selected_experts = [top_expert]

        else:
            # Traditional regex-based routing
            routing_decision = await self.router.route(context)
            routing_used = "regex"

            log.info(
                "Traditional routing decision",
                query_type=routing_decision.query_type,
                weights=routing_decision.expert_weights
            )

            # Traccia azione di routing
            trace.add_action(Action(
                action_type="routing",
                parameters={
                    "strategy": "regex",
                    "query_type": routing_decision.query_type,
                },
                log_prob=-0.5,
            ))

            # Step 4: Seleziona Expert
            selected_experts = routing_decision.get_selected_experts(
                threshold=self.config.selection_threshold
            )[:self.config.max_experts]

            if not selected_experts:
                # Fallback: usa tutti gli expert con peso uguale
                selected_experts = [(exp, 1.0 / len(self._experts)) for exp in self._experts.keys()]

        # Aggiorna metadata del trace con info routing
        trace.metadata["routing"] = {
            "strategy_used": routing_used,
            "neural_confidence": neural_confidence,
            "selected_experts": [e[0] for e in selected_experts],
        }

        log.info(f"Selected experts", experts=[e[0] for e in selected_experts])

        # Step 5: Esegui Expert
        if self.config.parallel_execution:
            responses = await self._run_experts_parallel(selected_experts, context)
        else:
            responses = await self._run_experts_sequential(selected_experts, context)

        # Step 6: Sintetizza con AdaptiveSynthesizer (disagreement detection)
        weights = {exp: w for exp, w in selected_experts}
        synthesis_result = await self.synthesizer.synthesize(
            query=query,
            responses=responses,
            weights=weights,
            trace_id=trace_id
        )

        # Calcola tempo di esecuzione
        execution_time_ms = (time.time() - start_time) * 1000

        # Aggiungi trace summary ai metadati
        trace.metadata["synthesis_result"] = {
            "mode": synthesis_result.mode.value,
            "confidence": synthesis_result.confidence,
            "experts_used": [r.expert_type for r in responses],
            "has_disagreement": synthesis_result.disagreement_analysis.has_disagreement if synthesis_result.disagreement_analysis else False,
            "num_alternatives": len(synthesis_result.alternatives),
            "execution_time_ms": execution_time_ms
        }

        log.info(
            f"Query processed",
            trace_id=trace_id,
            synthesis_mode=synthesis_result.mode.value,
            experts_run=len(responses),
            confidence=synthesis_result.confidence,
            has_disagreement=synthesis_result.disagreement_analysis.has_disagreement if synthesis_result.disagreement_analysis else False,
            time_ms=execution_time_ms,
            trace_actions=trace.num_actions
        )

        # Ritorna trace se richiesto (per RLCF loop)
        if return_trace:
            return synthesis_result, trace

        return synthesis_result

    async def _run_experts_parallel(
        self,
        selected_experts: List[tuple],
        context: ExpertContext
    ) -> List[ExpertResponse]:
        """Esegue Expert in parallelo."""
        async def run_with_timeout(expert_type: str) -> Optional[ExpertResponse]:
            expert = self._experts.get(expert_type)
            if not expert:
                return None

            try:
                return await asyncio.wait_for(
                    expert.analyze(context),
                    timeout=self.config.timeout_seconds
                )
            except asyncio.TimeoutError:
                log.warning(f"Expert {expert_type} timed out")
                return ExpertResponse(
                    expert_type=expert_type,
                    interpretation=f"Timeout durante l'analisi",
                    confidence=0.0,
                    limitations="Timeout",
                    trace_id=context.trace_id
                )
            except Exception as e:
                log.error(f"Expert {expert_type} failed: {e}")
                return ExpertResponse(
                    expert_type=expert_type,
                    interpretation=f"Errore: {str(e)}",
                    confidence=0.0,
                    limitations=str(e),
                    trace_id=context.trace_id
                )

        tasks = [run_with_timeout(exp) for exp, _ in selected_experts]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

    async def _run_experts_sequential(
        self,
        selected_experts: List[tuple],
        context: ExpertContext
    ) -> List[ExpertResponse]:
        """Esegue Expert in sequenza."""
        responses = []

        for expert_type, _ in selected_experts:
            expert = self._experts.get(expert_type)
            if not expert:
                continue

            try:
                response = await asyncio.wait_for(
                    expert.analyze(context),
                    timeout=self.config.timeout_seconds
                )
                responses.append(response)
            except asyncio.TimeoutError:
                log.warning(f"Expert {expert_type} timed out")
                responses.append(ExpertResponse(
                    expert_type=expert_type,
                    interpretation="Timeout",
                    confidence=0.0,
                    trace_id=context.trace_id
                ))
            except Exception as e:
                log.error(f"Expert {expert_type} failed: {e}")

        return responses

    async def process_with_routing(
        self,
        query: str,
        **kwargs
    ) -> tuple:
        """
        Processa e ritorna anche la decisione di routing.

        Returns:
            Tuple (SynthesisResult, RoutingDecision) se non usa GatingPolicy
            Tuple (SynthesisResult, ExecutionTrace) se usa GatingPolicy
        """
        # Se ha GatingPolicy, ritorna trace invece di routing_decision
        if self.gating_policy and self.embedding_service:
            result = await self.process(query, return_trace=True, **kwargs)
            return result  # (synthesis_result, trace)

        # Altrimenti traditional routing
        context = ExpertContext(
            query_text=query,
            entities=kwargs.get("entities") or {},
            retrieved_chunks=kwargs.get("retrieved_chunks") or [],
            metadata=kwargs.get("metadata") or {},
            trace_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )

        routing_decision = await self.router.route(context)
        synthesis_result = await self.process(query, **kwargs)

        return synthesis_result, routing_decision

    def get_expert(self, expert_type: str) -> Optional[BaseExpert]:
        """Ottiene un Expert per tipo."""
        return self._experts.get(expert_type)

    def list_experts(self) -> List[str]:
        """Lista gli Expert disponibili."""
        return list(self._experts.keys())

    async def run_single_expert(
        self,
        expert_type: str,
        query: str,
        **kwargs
    ) -> ExpertResponse:
        """
        Esegue un singolo Expert specifico.

        Utile per testing o quando si vuole bypassare il routing.
        """
        expert = self._experts.get(expert_type)
        if not expert:
            return ExpertResponse(
                expert_type=expert_type,
                interpretation=f"Expert '{expert_type}' non trovato",
                confidence=0.0
            )

        context = ExpertContext(
            query_text=query,
            entities=kwargs.get("entities") or {},
            retrieved_chunks=kwargs.get("retrieved_chunks") or [],
            metadata=kwargs.get("metadata") or {},
            trace_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )

        return await expert.analyze(context)
