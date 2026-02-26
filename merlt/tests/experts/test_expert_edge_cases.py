"""
Edge case tests for Expert implementations.

Tests per situazioni limite degli Expert:
- Confidence con fonti assenti o bassa qualita'
- Context malformato (campi None)
- Query vuota, Unicode, entita' vuote
- Tool execution errors
- ExpertResponse validation
- Clone isolation
- Concurrent execution
"""

import pytest
import asyncio
from typing import List
from copy import deepcopy

from merlt.experts import (
    LiteralExpert,
    SystemicExpert,
    PrinciplesExpert,
    PrecedentExpert,
    ExpertContext,
    ExpertResponse,
    LegalSource,
    ConfidenceFactors,
)
from merlt.tools import BaseTool, ToolResult, ToolParameter, ParameterType


# ============================================================================
# Mock Tools
# ============================================================================

class MockSemanticSearchTool(BaseTool):
    """Mock per SemanticSearchTool."""
    name = "semantic_search"
    description = "Mock semantic search"

    def __init__(self, results=None, should_fail=False):
        super().__init__()
        self._results = results
        self._should_fail = should_fail

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("query", ParameterType.STRING, "Query"),
            ToolParameter("top_k", ParameterType.INTEGER, "Results", required=False, default=5),
            ToolParameter("expert_type", ParameterType.STRING, "Expert", required=False),
            ToolParameter("source_types", ParameterType.ARRAY, "Source types", required=False),
        ]

    async def execute(self, query: str, top_k: int = 5, expert_type: str = None, source_types: list = None) -> ToolResult:
        if self._should_fail:
            raise RuntimeError("Tool execution failed")
        results = self._results if self._results is not None else []
        return ToolResult.ok(
            data={"results": results, "total": len(results), "expert_type": expert_type},
            tool_name=self.name,
        )


class MockGraphSearchTool(BaseTool):
    """Mock per GraphSearchTool."""
    name = "graph_search"
    description = "Mock graph search"

    def __init__(self, results=None, should_fail=False):
        super().__init__()
        self._results = results
        self._should_fail = should_fail

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("start_node", ParameterType.STRING, "Start"),
            ToolParameter("relation_types", ParameterType.ARRAY, "Relations", required=False),
            ToolParameter("max_hops", ParameterType.INTEGER, "Hops", required=False, default=2),
            ToolParameter("direction", ParameterType.STRING, "Direction", required=False),
        ]

    async def execute(self, start_node: str, **kwargs) -> ToolResult:
        if self._should_fail:
            raise RuntimeError("Graph search failed")
        if self._results is not None:
            return ToolResult.ok(data=self._results, tool_name=self.name)
        return ToolResult.ok(
            data={"nodes": [], "edges": [], "total_nodes": 0},
            tool_name=self.name,
        )


# ============================================================================
# Confidence Edge Cases
# ============================================================================

class TestConfidenceEdgeCases:
    """Tests per confidence in condizioni limite."""

    @pytest.mark.asyncio
    async def test_confidence_with_no_sources(self):
        """Expert senza fonti produce confidence bassa."""
        expert = LiteralExpert()
        context = ExpertContext(query_text="Domanda senza fonti")

        response = await expert.analyze(context)

        assert response.expert_type == "literal"
        # Without LLM and no sources, confidence should be low
        assert response.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_confidence_with_all_low_relevance_sources(self):
        """Expert con fonti a bassa rilevanza."""
        low_results = [
            {
                "chunk_id": f"chunk_{i}",
                "text": "Testo poco rilevante",
                "urn": f"urn:norma:test:art{i}",
                "final_score": 0.1,
            }
            for i in range(3)
        ]
        tools = [MockSemanticSearchTool(results=low_results)]
        expert = LiteralExpert(tools=tools)

        context = ExpertContext(query_text="Query con fonti irrilevanti")
        response = await expert.analyze(context)

        assert response.expert_type == "literal"
        assert response.trace_id == context.trace_id


# ============================================================================
# Malformed Context
# ============================================================================

class TestMalformedContext:
    """Tests per ExpertContext con campi malformati."""

    @pytest.mark.asyncio
    async def test_context_with_empty_entities(self):
        """Context con entities vuote."""
        expert = LiteralExpert()
        context = ExpertContext(query_text="Test", entities={})

        response = await expert.analyze(context)
        assert response.expert_type == "literal"

    @pytest.mark.asyncio
    async def test_context_with_none_embedding(self):
        """Context con query_embedding None."""
        expert = SystemicExpert()
        context = ExpertContext(
            query_text="Test embedding None",
            query_embedding=None,
        )

        response = await expert.analyze(context)
        assert response.expert_type == "systemic"
        assert response.trace_id == context.trace_id

    @pytest.mark.asyncio
    async def test_context_with_empty_metadata(self):
        """Context con metadata vuoti."""
        expert = PrinciplesExpert()
        context = ExpertContext(
            query_text="Test metadata vuoti",
            metadata={},
        )

        response = await expert.analyze(context)
        assert response.expert_type == "principles"


# ============================================================================
# Query Edge Cases
# ============================================================================

class TestQueryEdgeCases:
    """Tests per query problematiche."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Query vuota."""
        expert = LiteralExpert()
        context = ExpertContext(query_text="")

        response = await expert.analyze(context)
        assert response.expert_type == "literal"
        assert response.trace_id == context.trace_id

    @pytest.mark.asyncio
    async def test_unicode_query(self):
        """Query con caratteri Unicode complessi."""
        expert = PrecedentExpert()
        context = ExpertContext(
            query_text="Interpretazione dell'art. 2043 c.c. \u2014 danno \u00abingiusto\u00bb e nesso di causalit\u00e0"
        )

        response = await expert.analyze(context)
        assert response.expert_type == "precedent"

    @pytest.mark.asyncio
    async def test_query_rewriting_with_empty_entities(self):
        """Query rewriting con entities vuote non genera errore."""
        expert = LiteralExpert()
        context = ExpertContext(
            query_text="Test rewriting",
            entities={"article_numbers": [], "legal_concepts": []},
        )

        rewritten = expert._rewrite_search_query(context)
        # With empty entities, should return original query
        assert rewritten == "Test rewriting"


# ============================================================================
# Expert Types with No Results
# ============================================================================

class TestExpertsNoResults:
    """Ogni expert con tool che non ritorna risultati."""

    @pytest.fixture
    def empty_tools(self):
        return [MockSemanticSearchTool(results=[]), MockGraphSearchTool()]

    @pytest.mark.asyncio
    async def test_literal_no_results(self, empty_tools):
        expert = LiteralExpert(tools=empty_tools)
        context = ExpertContext(query_text="Query senza risultati")
        response = await expert.analyze(context)
        assert response.expert_type == "literal"
        assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_systemic_no_results(self, empty_tools):
        expert = SystemicExpert(tools=empty_tools)
        context = ExpertContext(query_text="Query senza risultati")
        response = await expert.analyze(context)
        assert response.expert_type == "systemic"

    @pytest.mark.asyncio
    async def test_principles_no_results(self, empty_tools):
        expert = PrinciplesExpert(tools=empty_tools)
        context = ExpertContext(query_text="Query senza risultati")
        response = await expert.analyze(context)
        assert response.expert_type == "principles"

    @pytest.mark.asyncio
    async def test_precedent_no_results(self, empty_tools):
        expert = PrecedentExpert(tools=empty_tools)
        context = ExpertContext(query_text="Query senza risultati")
        response = await expert.analyze(context)
        assert response.expert_type == "precedent"


# ============================================================================
# Tool Execution Error Handling
# ============================================================================

class TestToolExecutionErrors:
    """Tests per errori nei tools."""

    @pytest.mark.asyncio
    async def test_semantic_tool_failure(self):
        """Expert gestisce errore del semantic search tool."""
        tools = [MockSemanticSearchTool(should_fail=True)]
        expert = LiteralExpert(tools=tools)
        context = ExpertContext(query_text="Query che fa fallire il tool")

        # Should not raise, expert handles tool errors gracefully
        response = await expert.analyze(context)
        assert response.expert_type == "literal"

    @pytest.mark.asyncio
    async def test_graph_tool_failure(self):
        """Expert gestisce errore del graph search tool."""
        tools = [
            MockSemanticSearchTool(results=[
                {"chunk_id": "c1", "text": "Test", "urn": "urn:test:1", "final_score": 0.9}
            ]),
            MockGraphSearchTool(should_fail=True),
        ]
        expert = LiteralExpert(tools=tools)
        context = ExpertContext(
            query_text="Query con graph failure",
            entities={"norm_references": ["urn:test:1"]},
        )

        response = await expert.analyze(context)
        assert response.expert_type == "literal"


# ============================================================================
# ExpertResponse Validation
# ============================================================================

class TestExpertResponseValidation:
    """Tests per validazione ExpertResponse."""

    def test_required_fields(self):
        """ExpertResponse richiede expert_type."""
        response = ExpertResponse(expert_type="literal")
        assert response.expert_type == "literal"
        assert response.confidence == 0.5  # default
        assert response.interpretation == ""  # default

    def test_to_dict_completeness(self):
        """to_dict include tutti i campi."""
        response = ExpertResponse(
            expert_type="systemic",
            interpretation="Test interpretation",
            legal_basis=[
                LegalSource(
                    source_type="norm",
                    source_id="urn:test",
                    citation="Art. 1 test",
                )
            ],
            confidence=0.8,
            trace_id="test_trace",
        )
        d = response.to_dict()

        assert d["expert_type"] == "systemic"
        assert d["interpretation"] == "Test interpretation"
        assert d["confidence"] == 0.8
        assert len(d["legal_basis"]) == 1
        assert d["trace_id"] == "test_trace"

    def test_is_low_confidence(self):
        """is_low_confidence con threshold."""
        response = ExpertResponse(expert_type="literal", confidence=0.2)
        assert response.is_low_confidence(threshold=0.3) is True

        response2 = ExpertResponse(expert_type="literal", confidence=0.5)
        assert response2.is_low_confidence(threshold=0.3) is False

    def test_confidence_factors_compute_overall(self):
        """ConfidenceFactors.compute_overall bounds."""
        # All high
        cf = ConfidenceFactors(
            norm_clarity=1.0,
            jurisprudence_alignment=1.0,
            contextual_ambiguity=0.0,
            source_availability=1.0,
            definition_coverage=1.0,
        )
        assert cf.compute_overall() == 1.0

        # All low with high ambiguity
        cf2 = ConfidenceFactors(
            norm_clarity=0.0,
            jurisprudence_alignment=0.0,
            contextual_ambiguity=1.0,
            source_availability=0.0,
            definition_coverage=0.0,
        )
        assert cf2.compute_overall() == 0.0


# ============================================================================
# Clone Isolation
# ============================================================================

class TestCloneIsolation:
    """Tests per isolamento clone tools."""

    def test_tool_clone_traces_isolated(self):
        """Clone di tool non condivide trace list."""
        original = MockSemanticSearchTool()
        cloned = original.clone()

        # Aggiungi trace all'originale
        original._tool_call_traces.append({"test": True})

        # Clone non deve avere la trace
        assert len(cloned._tool_call_traces) == 0
        assert len(original._tool_call_traces) == 1

    def test_expert_tool_registry_isolation(self):
        """Expert con tools clonati mantiene registry separata."""
        tools = [MockSemanticSearchTool(), MockGraphSearchTool()]
        expert1 = LiteralExpert(tools=tools)
        expert2 = LiteralExpert(tools=tools)

        # Registry separate
        assert expert1._tool_registry is not expert2._tool_registry


# ============================================================================
# Concurrent Expert Execution
# ============================================================================

class TestConcurrentExecution:
    """Tests per esecuzione concorrente."""

    @pytest.mark.asyncio
    async def test_concurrent_experts_same_query(self):
        """Esecuzione concorrente di tutti e 4 gli expert."""
        experts = [
            LiteralExpert(),
            SystemicExpert(),
            PrinciplesExpert(),
            PrecedentExpert(),
        ]

        context = ExpertContext(
            query_text="Art. 2043 c.c. responsabilita' civile"
        )

        tasks = [expert.analyze(context) for expert in experts]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 4
        expert_types = {r.expert_type for r in responses}
        assert expert_types == {"literal", "systemic", "principles", "precedent"}

        # All should share the same trace_id
        trace_ids = {r.trace_id for r in responses}
        assert len(trace_ids) == 1

    @pytest.mark.asyncio
    async def test_concurrent_experts_with_tools(self):
        """Esecuzione concorrente con tools condivisi."""
        tools = [MockSemanticSearchTool(), MockGraphSearchTool()]

        experts = [
            LiteralExpert(tools=[t.clone() for t in tools]),
            SystemicExpert(tools=[t.clone() for t in tools]),
        ]

        context = ExpertContext(query_text="Test concorrente con tools")

        tasks = [expert.analyze(context) for expert in experts]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 2
        assert responses[0].expert_type != responses[1].expert_type
