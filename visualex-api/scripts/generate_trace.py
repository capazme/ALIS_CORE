#!/usr/bin/env python3
"""
Generate Pipeline Trace JSON.

Runs a query through the MERL-T pipeline and outputs the complete trace
as JSON for analysis and debugging.

Usage:
    # With mocked LLM (fast, no API key needed)
    python scripts/generate_trace.py "Cos'è la risoluzione del contratto?"

    # With live LLM (requires OPENROUTER_API_KEY)
    python scripts/generate_trace.py --live "Cos'è la risoluzione del contratto?"

    # Output to file
    python scripts/generate_trace.py -o trace.json "Query text"

    # Different user profile
    python scripts/generate_trace.py --profile analisi "Query text"

    # Pretty print (default) vs compact
    python scripts/generate_trace.py --compact "Query text"
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualex.experts import (
    PipelineOrchestrator,
    PipelineRequest,
    OrchestratorConfig,
    LiteralExpert,
    SystemicExpert,
    PrinciplesExpert,
    PrecedentExpert,
    LiteralConfig,
    SystemicConfig,
    PrinciplesConfig,
    PrecedentConfig,
    ExpertRouter,
    GatingNetwork,
    Synthesizer,
    LLMProviderFactory,
    FailoverLLMService,
)
from visualex.ner import NERService


# =============================================================================
# Mock Dependencies (for testing without LLM)
# =============================================================================


class MockChunkRetriever:
    """Mock retriever returning sample legal chunks."""

    SAMPLE_CHUNKS = [
        {
            "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
            "text": (
                "Art. 1453 - Risolubilità del contratto per inadempimento. "
                "Nei contratti con prestazioni corrispettive, quando uno dei "
                "contraenti non adempie le sue obbligazioni, l'altro può a sua "
                "scelta chiedere l'adempimento o la risoluzione del contratto, "
                "salvo, in ogni caso, il risarcimento del danno."
            ),
            "articolo": "1453",
            "tipo_atto": "regio.decreto",
            "rubrica": "Risolubilità del contratto per inadempimento",
        },
        {
            "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1454",
            "text": (
                "Art. 1454 - Diffida ad adempiere. "
                "Alla parte inadempiente l'altra può intimare per iscritto di "
                "adempiere in un congruo termine, con dichiarazione che, decorso "
                "inutilmente detto termine, il contratto s'intenderà senz'altro risoluto."
            ),
            "articolo": "1454",
            "tipo_atto": "regio.decreto",
            "rubrica": "Diffida ad adempiere",
        },
        {
            "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1455",
            "text": (
                "Art. 1455 - Importanza dell'inadempimento. "
                "Il contratto non si può risolvere se l'inadempimento di una "
                "delle parti ha scarsa importanza, avuto riguardo all'interesse "
                "dell'altra."
            ),
            "articolo": "1455",
            "tipo_atto": "regio.decreto",
            "rubrica": "Importanza dell'inadempimento",
        },
        {
            "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1456",
            "text": (
                "Art. 1456 - Clausola risolutiva espressa. "
                "I contraenti possono convenire espressamente che il contratto "
                "si risolva nel caso che una determinata obbligazione non sia "
                "adempiuta secondo le modalità stabilite."
            ),
            "articolo": "1456",
            "tipo_atto": "regio.decreto",
            "rubrica": "Clausola risolutiva espressa",
        },
    ]

    async def retrieve(
        self,
        query: str = "",
        top_k: int = 5,
        query_embedding: Optional[List[float]] = None,
        **kwargs,
    ) -> List[dict]:
        """Return sample chunks."""
        return self.SAMPLE_CHUNKS[:top_k]


class MockGraphTraverser:
    """Mock graph traverser returning sample relations."""

    async def get_related(
        self,
        urn: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 1,
        limit: int = 10,
    ) -> List[dict]:
        """Return sample graph relations."""
        return [
            {
                "source_urn": urn,
                "target_urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1454",
                "relation_type": "RIFERIMENTO",
                "weight": 0.85,
            },
            {
                "source_urn": urn,
                "target_urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1218",
                "relation_type": "PRESUPPONE",
                "weight": 0.75,
            },
            {
                "source_urn": urn,
                "target_urn": "urn:nir:stato:legge:2005-02-15;15~art3",
                "relation_type": "MODIFICA",
                "weight": 0.60,
            },
        ]

    async def get_history(self, urn: str) -> List[dict]:
        """Return sample historical versions."""
        return [
            {
                "urn": urn,
                "version_date": "1942-03-16",
                "status": "vigente",
            }
        ]


class MockLLMService:
    """Mock LLM service returning structured responses."""

    EXPERT_RESPONSES = {
        "literal": (
            "**Interpretazione Letterale**\n\n"
            "La risoluzione del contratto per inadempimento è un rimedio previsto "
            "dall'art. 1453 c.c. che consente alla parte non inadempiente di sciogliere "
            "il vincolo contrattuale quando la controparte non esegue la prestazione dovuta.\n\n"
            "Il testo normativo prevede due alternative:\n"
            "1. Chiedere l'adempimento forzato\n"
            "2. Chiedere la risoluzione del contratto\n\n"
            "In entrambi i casi è sempre salvo il diritto al risarcimento del danno."
        ),
        "systemic": (
            "**Interpretazione Sistematica**\n\n"
            "La risoluzione per inadempimento si inserisce nel sistema dei rimedi "
            "contrattuali del Codice Civile, collegandosi a:\n\n"
            "- Art. 1454 (diffida ad adempiere)\n"
            "- Art. 1455 (importanza dell'inadempimento)\n"
            "- Art. 1456 (clausola risolutiva espressa)\n"
            "- Art. 1457 (termine essenziale)\n\n"
            "Il sistema configura la risoluzione come rimedio subordinato alla gravità "
            "dell'inadempimento, escludendo i casi di scarsa importanza."
        ),
        "principles": (
            "**Ratio Legis e Principi**\n\n"
            "La disciplina della risoluzione si fonda su principi fondamentali:\n\n"
            "1. **Sinallagmaticità**: equilibrio tra prestazioni corrispettive\n"
            "2. **Buona fede**: art. 1375 c.c., dovere di correttezza\n"
            "3. **Conservazione del contratto**: la risoluzione è extrema ratio\n"
            "4. **Proporzionalità**: l'inadempimento deve essere grave\n\n"
            "La Costituzione (art. 41) tutela l'iniziativa economica privata, "
            "di cui il contratto è strumento essenziale."
        ),
        "precedent": (
            "**Giurisprudenza**\n\n"
            "Orientamenti consolidati della Cassazione:\n\n"
            "- **Cass. SS.UU. 553/2009**: la gravità dell'inadempimento va valutata "
            "in concreto, considerando l'interesse del creditore\n"
            "- **Cass. 18320/2015**: l'inadempimento deve alterare l'equilibrio contrattuale\n"
            "- **Cass. 26574/2019**: la risoluzione giudiziale ha effetto retroattivo\n\n"
            "Il giudice deve valutare comparativamente gli interessi delle parti."
        ),
    }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Return mock response based on context."""
        # Detect which expert is calling based on prompt content
        prompt_lower = prompt.lower()

        if "letterale" in prompt_lower or "literal" in prompt_lower:
            return self.EXPERT_RESPONSES["literal"]
        elif "sistematic" in prompt_lower or "systemic" in prompt_lower:
            return self.EXPERT_RESPONSES["systemic"]
        elif "principi" in prompt_lower or "ratio" in prompt_lower:
            return self.EXPERT_RESPONSES["principles"]
        elif "giurisprudenz" in prompt_lower or "precedent" in prompt_lower:
            return self.EXPERT_RESPONSES["precedent"]
        elif "sintetizz" in prompt_lower or "synthesi" in prompt_lower:
            return (
                "La risoluzione del contratto per inadempimento (art. 1453 c.c.) "
                "è il rimedio che consente alla parte fedele di sciogliere il contratto "
                "quando la controparte non adempie. Richiede un inadempimento grave "
                "(art. 1455) e può essere chiesta in alternativa all'adempimento, "
                "sempre con diritto al risarcimento. La giurisprudenza richiede una "
                "valutazione concreta della gravità in relazione all'interesse creditorio."
            )
        else:
            return "Risposta generica dell'LLM per il contesto fornito."


# =============================================================================
# Orchestrator Factory
# =============================================================================


def create_orchestrator(
    use_live_llm: bool = False,
    timeout_ms: float = 60000.0,
) -> PipelineOrchestrator:
    """
    Create a PipelineOrchestrator with appropriate dependencies.

    Args:
        use_live_llm: If True, use real LLM (requires OPENROUTER_API_KEY)
        timeout_ms: Expert timeout in milliseconds

    Returns:
        Configured PipelineOrchestrator
    """
    # Create mock dependencies
    retriever = MockChunkRetriever()
    graph_traverser = MockGraphTraverser()

    # Create LLM service (mock or live)
    if use_live_llm:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable required for --live mode"
            )
        factory = LLMProviderFactory()
        provider = factory.create("openrouter")
        llm_service = FailoverLLMService(providers=[provider])
    else:
        llm_service = MockLLMService()

    # Create config
    config = OrchestratorConfig(
        expert_timeout_ms=timeout_ms,
        total_timeout_ms=timeout_ms * 4,
        parallel_execution=True,
        enable_tracing=True,
        enable_metrics=True,
        enable_feedback_hooks=True,
    )

    # Create experts
    literal_expert = LiteralExpert(
        retriever=retriever,
        llm_service=llm_service,
        config=LiteralConfig(),
    )

    systemic_expert = SystemicExpert(
        retriever=retriever,
        graph_traverser=graph_traverser,
        llm_service=llm_service,
        config=SystemicConfig(),
    )

    principles_expert = PrinciplesExpert(
        retriever=retriever,
        llm_service=llm_service,
        config=PrinciplesConfig(),
    )

    precedent_expert = PrecedentExpert(
        retriever=retriever,
        llm_service=llm_service,
        config=PrecedentConfig(),
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        ner_service=NERService(),
        router=ExpertRouter(),
        gating=GatingNetwork(llm_service=llm_service),
        synthesizer=Synthesizer(llm_service=llm_service),
    )

    # Register experts
    orchestrator.register_expert("literal", literal_expert)
    orchestrator.register_expert("systemic", systemic_expert)
    orchestrator.register_expert("principles", principles_expert)
    orchestrator.register_expert("precedent", precedent_expert)

    return orchestrator


# =============================================================================
# Main Execution
# =============================================================================


async def generate_trace(
    query: str,
    user_profile: str = "ricerca",
    use_live_llm: bool = False,
    timeout_ms: float = 60000.0,
) -> Dict[str, Any]:
    """
    Generate a complete pipeline trace for the given query.

    Args:
        query: Legal query text
        user_profile: User profile (consulenza|ricerca|analisi|contributore)
        use_live_llm: Use real LLM instead of mocks
        timeout_ms: Expert timeout

    Returns:
        Complete trace as dictionary
    """
    orchestrator = create_orchestrator(
        use_live_llm=use_live_llm,
        timeout_ms=timeout_ms,
    )

    request = PipelineRequest(
        query=query,
        user_profile=user_profile,
    )

    result = await orchestrator.process_query(request)

    # Build comprehensive output
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "query": query,
            "user_profile": user_profile,
            "mode": "live" if use_live_llm else "mock",
            "success": result.success,
            "error": result.error,
        },
        "response": (
            result.response.to_dict()
            if hasattr(result.response, "to_dict")
            else str(result.response)
        ),
        "trace": result.trace.to_dict(),
        "metrics": result.metrics.to_dict(),
        "feedback_hooks": [fh.to_dict() for fh in result.feedback_hooks],
    }

    return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate MERL-T Pipeline Trace JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="Cos'è la risoluzione del contratto per inadempimento?",
        help="Legal query to process (default: sample contract resolution query)",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="ricerca",
        choices=["consulenza", "ricerca", "analisi", "contributore"],
        help="User profile for response formatting (default: ricerca)",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live LLM (requires OPENROUTER_API_KEY)",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (no indentation)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60000.0,
        help="Expert timeout in milliseconds (default: 60000)",
    )

    args = parser.parse_args()

    # Run async generation
    try:
        trace = asyncio.run(
            generate_trace(
                query=args.query,
                user_profile=args.profile,
                use_live_llm=args.live,
                timeout_ms=args.timeout,
            )
        )
    except Exception as e:
        print(f"Error generating trace: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    indent = None if args.compact else 2
    json_output = json.dumps(trace, ensure_ascii=False, indent=indent)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json_output, encoding="utf-8")
        print(f"Trace written to: {output_path}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
