"""
[P2] Unit tests for expert query rewriting.

Each expert's _rewrite_search_query produces a query focused on its
interpretive lens:
- LiteralExpert: text + definitions
- SystemicExpert: connections + system
- PrinciplesExpert: ratio legis
- PrecedentExpert: giurisprudenza
"""

from unittest.mock import patch, MagicMock

import pytest

from merlt.experts.base import ExpertContext


def _make_context(query, articles=None, concepts=None):
    """Build an ExpertContext with given entities."""
    entities = {}
    if articles:
        entities["article_numbers"] = articles
    if concepts:
        entities["legal_concepts"] = concepts
    return ExpertContext(query_text=query, entities=entities)


def _create_expert(expert_cls):
    """Instantiate an expert with mocked dependencies to avoid YAML/tool issues."""
    with patch.object(expert_cls, "_load_config", lambda self: None):
        expert = object.__new__(expert_cls)
        expert.expert_type = expert_cls.expert_type
        expert.description = expert_cls.description
        expert.tools = []
        expert.config = {}
        expert.ai_service = None
        expert.policy_manager = None
        expert._current_trace = None
        expert._llm_call_traces = []
        expert._tool_call_traces = []
        from merlt.tools import ToolRegistry
        expert._tool_registry = ToolRegistry()
        expert.prompt_template = ""
        expert.temperature = 0.3
        expert.model = "test"
    return expert


# --- Lazy imports to avoid heavy module-level imports ---

def _get_literal_expert():
    from merlt.experts.literal import LiteralExpert
    return _create_expert(LiteralExpert)


def _get_systemic_expert():
    from merlt.experts.systemic import SystemicExpert
    return _create_expert(SystemicExpert)


def _get_principles_expert():
    from merlt.experts.principles import PrinciplesExpert
    return _create_expert(PrinciplesExpert)


def _get_precedent_expert():
    from merlt.experts.precedent import PrecedentExpert
    return _create_expert(PrecedentExpert)


class TestLiteralRewrite:
    def test_p2_literal_with_articles_and_concepts(self):
        expert = _get_literal_expert()
        ctx = _make_context("domanda", articles=["1321"], concepts=["contratto"])
        result = expert._rewrite_search_query(ctx)
        assert "testo" in result
        assert "1321" in result
        assert "definizione" in result or "significato" in result

    def test_p2_literal_with_articles_only(self):
        expert = _get_literal_expert()
        ctx = _make_context("domanda", articles=["2043"])
        result = expert._rewrite_search_query(ctx)
        assert "testo" in result
        assert "articolo" in result
        assert "2043" in result

    def test_p2_literal_fallback_uses_original(self):
        expert = _get_literal_expert()
        ctx = _make_context("cosa dice la legge?")
        result = expert._rewrite_search_query(ctx)
        assert result == "cosa dice la legge?"


class TestSystemicRewrite:
    def test_p2_systemic_with_articles_and_concepts(self):
        expert = _get_systemic_expert()
        ctx = _make_context("domanda", articles=["1321"], concepts=["contratto"])
        result = expert._rewrite_search_query(ctx)
        assert "correlate" in result or "sistema" in result
        assert "1321" in result

    def test_p2_systemic_with_concepts_only(self):
        expert = _get_systemic_expert()
        ctx = _make_context("domanda", concepts=["obbligazione"])
        result = expert._rewrite_search_query(ctx)
        assert "obbligazione" in result
        assert "sistema" in result or "correlazione" in result

    def test_p2_systemic_fallback_uses_original(self):
        expert = _get_systemic_expert()
        ctx = _make_context("norma collegata")
        result = expert._rewrite_search_query(ctx)
        assert result == "norma collegata"


class TestPrinciplesRewrite:
    def test_p2_principles_with_concepts(self):
        expert = _get_principles_expert()
        ctx = _make_context("domanda", concepts=["buona fede"])
        result = expert._rewrite_search_query(ctx)
        assert "ratio legis" in result
        assert "buona fede" in result

    def test_p2_principles_with_articles_only(self):
        expert = _get_principles_expert()
        ctx = _make_context("domanda", articles=["1175"])
        result = expert._rewrite_search_query(ctx)
        assert "ratio legis" in result
        assert "1175" in result
        assert "finalità" in result or "scopo" in result

    def test_p2_principles_fallback_uses_original(self):
        expert = _get_principles_expert()
        ctx = _make_context("perche' esiste questa norma?")
        result = expert._rewrite_search_query(ctx)
        assert result == "perche' esiste questa norma?"


class TestPrecedentRewrite:
    def test_p2_precedent_with_concepts_and_articles(self):
        expert = _get_precedent_expert()
        ctx = _make_context("domanda", articles=["2043"], concepts=["danno"])
        result = expert._rewrite_search_query(ctx)
        assert "giurisprudenza" in result
        assert "2043" in result

    def test_p2_precedent_with_concepts_only(self):
        expert = _get_precedent_expert()
        ctx = _make_context("domanda", concepts=["risarcimento"])
        result = expert._rewrite_search_query(ctx)
        assert "giurisprudenza" in result
        assert "risarcimento" in result
        assert "precedente" in result or "massima" in result

    def test_p2_precedent_with_articles_only(self):
        expert = _get_precedent_expert()
        ctx = _make_context("domanda", articles=["2043"])
        result = expert._rewrite_search_query(ctx)
        assert "giurisprudenza" in result
        assert "2043" in result
        assert "sentenza" in result or "cassazione" in result

    def test_p2_precedent_fallback_uses_original(self):
        expert = _get_precedent_expert()
        ctx = _make_context("cosa dice la cassazione?")
        result = expert._rewrite_search_query(ctx)
        assert result == "cosa dice la cassazione?"
