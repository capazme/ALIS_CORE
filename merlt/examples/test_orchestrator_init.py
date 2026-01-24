#!/usr/bin/env python3
"""Test orchestrator initialization"""

import asyncio
import structlog

log = structlog.get_logger()

async def test_init():
    """Test if we can initialize the orchestrator"""
    try:
        from merlt.rlcf.ai_service import OpenRouterService
        from merlt.experts.synthesizer import AdaptiveSynthesizer, SynthesisConfig
        from merlt.experts.orchestrator import MultiExpertOrchestrator, OrchestratorConfig

        print("✅ Imports successful")

        # Create AI service
        ai_service = OpenRouterService()
        print("✅ OpenRouterService created")

        # Create synthesizer
        synthesis_config = SynthesisConfig(
            convergent_threshold=0.5,
            resolvability_weight=0.3,
            include_disagreement_explanation=True,
            max_alternatives=3,
        )
        synthesizer = AdaptiveSynthesizer(
            config=synthesis_config,
            ai_service=ai_service,
        )
        print("✅ AdaptiveSynthesizer created")

        # Create orchestrator
        orchestrator_config = OrchestratorConfig(
            max_experts=4,
            timeout_seconds=60,
            parallel_execution=True,
        )
        orchestrator = MultiExpertOrchestrator(
            synthesizer=synthesizer,
            tools=[],
            ai_service=ai_service,
            config=orchestrator_config,
        )
        print("✅ MultiExpertOrchestrator created")

        # Test simple query
        print("\nTesting query...")
        result = await orchestrator.process("Cos'è la legittima difesa?")
        print(f"✅ Query successful!")
        print(f"   Mode: {result.mode}")
        print(f"   Synthesis: {result.synthesis[:100]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_init())
