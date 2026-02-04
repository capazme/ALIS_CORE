"""
Integration Tests for LLM Providers.

These tests make REAL API calls and require valid API keys.
Run with: pytest tests/integration/test_llm_integration.py -v -s

Configuration via .env file:
- OPENROUTER_API_KEY: OpenRouter API key
- OPENROUTER_DEFAULT_MODEL: Default model to use (e.g., google/gemini-2.5-flash)

Results are documented in: tests/integration/llm_test_results.md
"""

import asyncio
import os
import time
import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def has_api_key():
    """Check if OpenRouter API key is available."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


def get_default_model():
    """Get default model from .env."""
    return os.getenv("OPENROUTER_DEFAULT_MODEL", "google/gemini-2.5-flash")


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not has_api_key(),
    reason="OPENROUTER_API_KEY not set. Create .env file or export the variable."
)


class ResultsCollector:
    """Collector for test results documentation."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self.total_cost = 0.0
        self.total_tokens = 0

    def add(self, test_name: str, model: str, result: Dict[str, Any]):
        self.results.append({
            "test": test_name,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            **result
        })
        if result.get("cost"):
            self.total_cost += result["cost"]
        if result.get("total_tokens"):
            self.total_tokens += result["total_tokens"]

    def save(self, filepath: str):
        """Save results to markdown file."""
        with open(filepath, "w") as f:
            f.write("# LLM Integration Test Results\n\n")
            f.write(f"**Date:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Default Model:** {get_default_model()}\n")
            f.write(f"**Total Tests:** {len(self.results)}\n")
            f.write(f"**Total Cost:** ${self.total_cost:.4f}\n")
            f.write(f"**Total Tokens:** {self.total_tokens:,}\n\n")

            # Summary table
            f.write("## Summary\n\n")
            f.write("| Test | Model | Status | Latency | Tokens | Cost |\n")
            f.write("|------|-------|--------|---------|--------|------|\n")

            for r in self.results:
                status = "✅" if r.get("success") else "❌"
                latency = f"{r.get('latency_ms', 0):.0f}ms"
                tokens = r.get('total_tokens', 'N/A')
                cost = f"${r.get('cost', 0):.4f}" if r.get('cost') else "N/A"
                f.write(f"| {r['test']} | {r['model']} | {status} | {latency} | {tokens} | {cost} |\n")

            # Detailed results
            f.write("\n## Detailed Results\n\n")
            for r in self.results:
                f.write(f"### {r['test']} - {r['model']}\n\n")
                f.write(f"- **Status:** {'Success' if r.get('success') else 'Failed'}\n")
                f.write(f"- **Latency:** {r.get('latency_ms', 0):.0f}ms\n")
                if r.get('prompt_tokens'):
                    f.write(f"- **Prompt Tokens:** {r.get('prompt_tokens')}\n")
                if r.get('completion_tokens'):
                    f.write(f"- **Completion Tokens:** {r.get('completion_tokens')}\n")
                if r.get('total_tokens'):
                    f.write(f"- **Total Tokens:** {r.get('total_tokens')}\n")
                if r.get('cost'):
                    f.write(f"- **Estimated Cost:** ${r.get('cost'):.4f}\n")
                if r.get('error'):
                    f.write(f"- **Error:** {r.get('error')}\n")
                if r.get('response_preview'):
                    f.write(f"\n**Response Preview:**\n```\n{r.get('response_preview')}\n```\n")
                f.write("\n")

            # Cost summary
            f.write("## Cost Summary\n\n")
            f.write(f"- **Total Cost:** ${self.total_cost:.4f}\n")
            f.write(f"- **Total Tokens:** {self.total_tokens:,}\n")
            f.write(f"- **Average Cost per Test:** ${self.total_cost / len(self.results):.4f}\n" if self.results else "")


# Global results collector
test_results = ResultsCollector()


@pytest.fixture(scope="module")
def results_path():
    """Path for results file."""
    return Path(__file__).parent / "llm_test_results.md"


@pytest.fixture(scope="module", autouse=True)
def save_results(results_path):
    """Save results after all tests complete."""
    yield
    test_results.save(str(results_path))
    print(f"\n\n📄 Results saved to: {results_path}")
    print(f"💰 Total cost: ${test_results.total_cost:.4f}")
    print(f"🔢 Total tokens: {test_results.total_tokens:,}")


class TestDefaultModelBasic:
    """Basic tests using the default model from .env."""

    @pytest.fixture
    def provider(self):
        """Create OpenRouter provider with default model."""
        from visualex.experts.llm import OpenRouterProvider
        return OpenRouterProvider()  # Uses OPENROUTER_DEFAULT_MODEL from .env

    @pytest.mark.asyncio
    async def test_simple_query(self, provider):
        """Test simple query with default model."""
        model = get_default_model()
        prompt = "Rispondi solo 'OK' se funzioni correttamente."

        try:
            response = await provider.generate(prompt=prompt, max_tokens=10)

            test_results.add("Simple Query", model, {
                "success": True,
                "latency_ms": response.latency_ms,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": response.usage.estimated_cost,
                "response_preview": response.content,
            })

            assert response.content
        except Exception as e:
            test_results.add("Simple Query", model, {"success": False, "error": str(e)})
            raise

    @pytest.mark.asyncio
    async def test_legal_query_contratto(self, provider):
        """Test legal query about contracts."""
        model = get_default_model()
        prompt = "Cos'è un contratto secondo il codice civile italiano? Rispondi in 2 frasi."

        try:
            response = await provider.generate(prompt=prompt, max_tokens=150)

            test_results.add("Legal Query - Contratto", model, {
                "success": True,
                "latency_ms": response.latency_ms,
                "total_tokens": response.usage.total_tokens,
                "cost": response.usage.estimated_cost,
                "response_preview": response.content[:300],
            })

            assert "contratto" in response.content.lower() or "accordo" in response.content.lower()
        except Exception as e:
            test_results.add("Legal Query - Contratto", model, {"success": False, "error": str(e)})
            raise

    @pytest.mark.asyncio
    async def test_legal_query_risoluzione(self, provider):
        """Test legal query about contract resolution."""
        model = get_default_model()
        prompt = """Art. 1453 c.c. - Risolubilità del contratto per inadempimento.
Domanda: Quando posso risolvere un contratto? Rispondi brevemente."""

        try:
            response = await provider.generate(prompt=prompt, max_tokens=200)

            test_results.add("Legal Query - Risoluzione", model, {
                "success": True,
                "latency_ms": response.latency_ms,
                "total_tokens": response.usage.total_tokens,
                "cost": response.usage.estimated_cost,
                "response_preview": response.content[:300],
            })

            assert len(response.content) > 50
        except Exception as e:
            test_results.add("Legal Query - Risoluzione", model, {"success": False, "error": str(e)})
            raise


class TestExpertQueries:
    """Test queries simulating each expert type."""

    @pytest.fixture
    def provider(self):
        from visualex.experts.llm import OpenRouterProvider
        return OpenRouterProvider()

    @pytest.mark.asyncio
    async def test_literal_expert_query(self, provider):
        """Test query for LiteralExpert."""
        model = get_default_model()
        prompt = """Sei un esperto legale. Analizza LETTERALMENTE questo testo:

Art. 1453 c.c.: "Nei contratti con prestazioni corrispettive, quando uno dei contraenti
non adempie le sue obbligazioni, l'altro può a sua scelta chiedere l'adempimento o
la risoluzione del contratto, salvo, in ogni caso, il risarcimento del danno."

Domanda: Quali sono le due scelte del contraente non inadempiente? Cita il testo."""

        try:
            response = await provider.generate(prompt=prompt, max_tokens=250)

            test_results.add("LiteralExpert Query", model, {
                "success": True,
                "latency_ms": response.latency_ms,
                "total_tokens": response.usage.total_tokens,
                "cost": response.usage.estimated_cost,
                "response_preview": response.content[:400],
            })

            assert "adempimento" in response.content.lower()
            assert "risoluzione" in response.content.lower()
        except Exception as e:
            test_results.add("LiteralExpert Query", model, {"success": False, "error": str(e)})
            raise

    @pytest.mark.asyncio
    async def test_systemic_expert_query(self, provider):
        """Test query for SystemicExpert."""
        model = get_default_model()
        prompt = """Sei un esperto di diritto civile. Analizza le CONNESSIONI SISTEMATICHE:

- Art. 1453 c.c. (Risolubilità per inadempimento)
- Art. 1455 c.c. (Importanza dell'inadempimento)
- Art. 1456 c.c. (Clausola risolutiva espressa)

Come si collegano queste norme? Qual è la logica del sistema?"""

        try:
            response = await provider.generate(prompt=prompt, max_tokens=350)

            test_results.add("SystemicExpert Query", model, {
                "success": True,
                "latency_ms": response.latency_ms,
                "total_tokens": response.usage.total_tokens,
                "cost": response.usage.estimated_cost,
                "response_preview": response.content[:400],
            })

            assert len(response.content) > 100
        except Exception as e:
            test_results.add("SystemicExpert Query", model, {"success": False, "error": str(e)})
            raise

    @pytest.mark.asyncio
    async def test_principles_expert_query(self, provider):
        """Test query for PrinciplesExpert."""
        model = get_default_model()
        prompt = """Sei un costituzionalista. Quali PRINCIPI COSTITUZIONALI tutelano
il contraente debole nei contratti?

Considera: Art. 3 Cost. (uguaglianza), Art. 41 Cost. (iniziativa economica).
Rispondi brevemente."""

        try:
            response = await provider.generate(prompt=prompt, max_tokens=300)

            test_results.add("PrinciplesExpert Query", model, {
                "success": True,
                "latency_ms": response.latency_ms,
                "total_tokens": response.usage.total_tokens,
                "cost": response.usage.estimated_cost,
                "response_preview": response.content[:400],
            })

            assert "costituzion" in response.content.lower() or "art." in response.content.lower()
        except Exception as e:
            test_results.add("PrinciplesExpert Query", model, {"success": False, "error": str(e)})
            raise

    @pytest.mark.asyncio
    async def test_precedent_expert_query(self, provider):
        """Test query for PrecedentExpert."""
        model = get_default_model()
        prompt = """Sei un esperto di giurisprudenza italiana.

Come ha interpretato la Cassazione il requisito della "non scarsa importanza"
dell'inadempimento (art. 1455 c.c.)? Quali criteri usa?"""

        try:
            response = await provider.generate(prompt=prompt, max_tokens=350)

            test_results.add("PrecedentExpert Query", model, {
                "success": True,
                "latency_ms": response.latency_ms,
                "total_tokens": response.usage.total_tokens,
                "cost": response.usage.estimated_cost,
                "response_preview": response.content[:400],
            })

            assert len(response.content) > 100
        except Exception as e:
            test_results.add("PrecedentExpert Query", model, {"success": False, "error": str(e)})
            raise


class TestFullExpertPipeline:
    """End-to-end test of the full expert pipeline."""

    @pytest.fixture
    def provider(self):
        from visualex.experts.llm import OpenRouterProvider
        return OpenRouterProvider()

    @pytest.mark.asyncio
    async def test_moe_pipeline_simulation(self, provider):
        """
        Simulate full Mixture of Experts pipeline.

        Flow: Query → 4 Experts → Synthesis
        """
        model = get_default_model()
        start_time = time.time()
        total_cost = 0.0
        total_tokens = 0

        base_query = "Quando posso risolvere un contratto per inadempimento?"
        norm_context = """Art. 1453 c.c.: "Nei contratti con prestazioni corrispettive,
quando uno dei contraenti non adempie le sue obbligazioni, l'altro può a sua scelta
chiedere l'adempimento o la risoluzione del contratto, salvo il risarcimento del danno."
"""

        expert_responses = {}

        # 1. Literal Expert
        try:
            response = await provider.generate(
                prompt=f"Interpreta LETTERALMENTE:\n{norm_context}\n\nRispondi in 80 parole max.",
                max_tokens=150
            )
            expert_responses["literal"] = response.content
            total_cost += response.usage.estimated_cost or 0
            total_tokens += response.usage.total_tokens or 0
        except Exception as e:
            expert_responses["literal"] = f"FAILED: {e}"

        # 2. Systemic Expert
        try:
            response = await provider.generate(
                prompt=f"Analizza connessioni con art. 1454, 1455, 1456 c.c.:\n{norm_context}\n\n80 parole max.",
                max_tokens=150
            )
            expert_responses["systemic"] = response.content
            total_cost += response.usage.estimated_cost or 0
            total_tokens += response.usage.total_tokens or 0
        except Exception as e:
            expert_responses["systemic"] = f"FAILED: {e}"

        # 3. Principles Expert
        try:
            response = await provider.generate(
                prompt=f"Principi costituzionali applicabili (art. 3, 24 Cost.):\n{norm_context}\n\n80 parole max.",
                max_tokens=150
            )
            expert_responses["principles"] = response.content
            total_cost += response.usage.estimated_cost or 0
            total_tokens += response.usage.total_tokens or 0
        except Exception as e:
            expert_responses["principles"] = f"FAILED: {e}"

        # 4. Precedent Expert
        try:
            response = await provider.generate(
                prompt=f"Orientamenti Cassazione su gravità inadempimento:\n{norm_context}\n\n80 parole max.",
                max_tokens=150
            )
            expert_responses["precedent"] = response.content
            total_cost += response.usage.estimated_cost or 0
            total_tokens += response.usage.total_tokens or 0
        except Exception as e:
            expert_responses["precedent"] = f"FAILED: {e}"

        # 5. Synthesizer
        synthesis_prompt = f"""Sintetizza queste risposte degli esperti alla domanda: "{base_query}"

LETTERALE: {expert_responses.get('literal', 'N/A')[:200]}

SISTEMATICA: {expert_responses.get('systemic', 'N/A')[:200]}

COSTITUZIONALE: {expert_responses.get('principles', 'N/A')[:200]}

GIURISPRUDENZA: {expert_responses.get('precedent', 'N/A')[:200]}

Fornisci una risposta integrata e coerente (max 150 parole)."""

        try:
            response = await provider.generate(prompt=synthesis_prompt, max_tokens=300)
            synthesis = response.content
            total_cost += response.usage.estimated_cost or 0
            total_tokens += response.usage.total_tokens or 0
        except Exception as e:
            synthesis = f"FAILED: {e}"

        elapsed_ms = (time.time() - start_time) * 1000
        all_success = not any("FAILED" in str(v) for v in expert_responses.values()) and "FAILED" not in synthesis

        test_results.add("MoE Pipeline - Full", model, {
            "success": all_success,
            "latency_ms": elapsed_ms,
            "total_tokens": total_tokens,
            "cost": total_cost,
            "response_preview": f"""
Experts: {len([v for v in expert_responses.values() if 'FAILED' not in str(v)])}/4 success

SYNTHESIS:
{synthesis[:500]}...""",
        })

        assert all_success, f"Pipeline failed: {expert_responses}"
        assert len(synthesis) > 100


class TestProviderFeatures:
    """Test provider-specific features."""

    @pytest.mark.asyncio
    async def test_failover_service(self):
        """Test FailoverLLMService."""
        from visualex.experts.llm import OpenRouterProvider, FailoverLLMService, FailoverConfig

        provider = OpenRouterProvider()
        service = FailoverLLMService(
            providers=[provider],
            config=FailoverConfig(cooldown_seconds=60),
        )

        try:
            response = await service.generate("Cos'è un contratto? Una frase.", max_tokens=50)

            test_results.add("Failover Service", get_default_model(), {
                "success": True,
                "latency_ms": 0,
                "response_preview": response[:200],
            })

            assert len(response) > 10
        except Exception as e:
            test_results.add("Failover Service", get_default_model(), {"success": False, "error": str(e)})
            raise

    @pytest.mark.asyncio
    async def test_factory_creates_provider(self):
        """Test LLMProviderFactory."""
        from visualex.experts.llm import LLMProviderFactory

        factory = LLMProviderFactory()
        provider = factory.create("openrouter")

        try:
            response = await provider.generate("Rispondi 'OK'.", max_tokens=10)

            test_results.add("Factory Create", "openrouter", {
                "success": True,
                "latency_ms": response.latency_ms,
                "response_preview": response.content,
            })

            assert response.content
        except Exception as e:
            test_results.add("Factory Create", "openrouter", {"success": False, "error": str(e)})
            raise

    def test_available_providers(self):
        """Test available providers list."""
        from visualex.experts.llm import LLMProviderFactory

        providers = LLMProviderFactory.get_available_providers()

        test_results.add("Available Providers", "API", {
            "success": "openrouter" in providers,
            "latency_ms": 0,
            "response_preview": f"Providers: {providers}",
        })

        assert "openrouter" in providers

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing available models."""
        from visualex.experts.llm import OpenRouterProvider

        provider = OpenRouterProvider()

        try:
            models = await provider.list_models()

            test_results.add("List Models", "API", {
                "success": len(models) > 0,
                "latency_ms": 0,
                "response_preview": f"Found {len(models)} models. Sample: {[m['id'] for m in models[:5]]}",
            })

            assert len(models) > 0
        except Exception as e:
            test_results.add("List Models", "API", {"success": False, "error": str(e)})
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
