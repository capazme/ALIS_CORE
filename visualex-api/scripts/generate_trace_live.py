#!/usr/bin/env python3
"""
Generate Pipeline Trace JSON with LIVE data.

Uses real FalkorDB graph (27k+ nodes, 43k+ relations) and optionally real LLM.

Usage:
    # With real graph data, mock LLM
    python scripts/generate_trace_live.py "Cos'è la risoluzione del contratto?"

    # With real graph AND real LLM
    python scripts/generate_trace_live.py --live-llm "Cos'è la risoluzione del contratto?"

    # Output to file
    python scripts/generate_trace_live.py -o trace_live.json "Query"

Requirements:
    - FalkorDB running on port 6380 with merl_t_dev graph
    - Qdrant running on port 6333 (optional, for vector search)
    - OPENROUTER_API_KEY for --live-llm mode
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis

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
# Real FalkorDB Retriever
# =============================================================================


class FalkorDBRetriever:
    """
    Retriever that fetches real data from FalkorDB graph.

    Uses a multi-strategy approach:
    1. Concept-based: Find ConcettoGiuridico nodes, traverse DISCIPLINA to Norms
    2. Rubrica-based: Match query terms against article titles (high precision)
    3. Text-based: Fall back to testo_vigente search (high recall, lower precision)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6380,
        graph_name: str = "merl_t_dev",
    ):
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        return self._client

    async def retrieve(
        self,
        query: str = "",
        top_k: int = 5,
        query_embedding: Optional[List[float]] = None,
        **kwargs,
    ) -> List[dict]:
        """
        Retrieve relevant legal chunks from FalkorDB using multi-strategy approach.

        Strategy priority:
        1. Concept-based retrieval via DISCIPLINA relations
        2. Rubrica matching (article titles)
        3. Text content matching (fallback)
        """
        client = self._get_client()
        keywords = self._extract_keywords(query)

        # Build concept search terms (combine adjacent keywords for phrases)
        concept_terms = self._build_concept_terms(keywords)

        results = []
        seen_urns = set()

        # Strategy 1: Concept-based retrieval
        concept_results = await self._retrieve_via_concepts(client, concept_terms, top_k)
        for r in concept_results:
            if r.get("urn") and r["urn"] not in seen_urns:
                r["retrieval_method"] = "concept"
                r["relevance_boost"] = 1.0  # Highest priority
                results.append(r)
                seen_urns.add(r["urn"])

        # Strategy 2: Rubrica-based retrieval (article titles)
        if len(results) < top_k:
            rubrica_results = await self._retrieve_via_rubrica(client, keywords, top_k)
            for r in rubrica_results:
                if r.get("urn") and r["urn"] not in seen_urns:
                    r["retrieval_method"] = "rubrica"
                    r["relevance_boost"] = 0.8
                    results.append(r)
                    seen_urns.add(r["urn"])

        # Strategy 3: Text-based retrieval (fallback)
        if len(results) < top_k:
            text_results = await self._retrieve_via_text(client, keywords, top_k)
            for r in text_results:
                if r.get("urn") and r["urn"] not in seen_urns:
                    r["retrieval_method"] = "text"
                    r["relevance_boost"] = 0.5
                    results.append(r)
                    seen_urns.add(r["urn"])

        # Sort by relevance boost, then return top_k
        results.sort(key=lambda x: x.get("relevance_boost", 0), reverse=True)
        return results[:top_k]

    async def _retrieve_via_concepts(
        self, client, concept_terms: List[str], limit: int
    ) -> List[dict]:
        """
        Find norms via ConcettoGiuridico nodes and DISCIPLINA relations.

        This is the most precise method - if a concept matches, the related
        norms are highly likely to be relevant.

        Strategy: Find norms that are linked to concepts matching MULTIPLE
        search terms (higher relevance score).
        """
        # Track how many concept terms each norm matches
        norm_scores: Dict[str, Dict[str, Any]] = {}

        # Get individual keywords for multi-match scoring
        keywords = [t for t in concept_terms if ' ' not in t][:5]

        for term in concept_terms[:6]:
            try:
                # Find concepts matching the term and get norms that DISCIPLINA them
                cypher = f"""
                MATCH (n:Norma)-[r:DISCIPLINA]->(c:ConcettoGiuridico)
                WHERE toLower(c.nome) CONTAINS '{term.lower()}'
                   OR toLower(c.node_id) CONTAINS '{term.lower().replace(' ', '_')}'
                RETURN DISTINCT n.URN as urn,
                       n.testo_vigente as text,
                       n.numero_articolo as articolo,
                       n.tipo_documento as tipo_atto,
                       n.rubrica as rubrica,
                       c.nome as matched_concept
                LIMIT {limit * 2}
                """
                response = client.execute_command(
                    "GRAPH.QUERY",
                    self.graph_name,
                    cypher,
                )
                chunks = self._parse_concept_response(response)

                for chunk in chunks:
                    urn = chunk.get("urn")
                    if not urn:
                        continue

                    if urn not in norm_scores:
                        norm_scores[urn] = {
                            **chunk,
                            "concept_matches": [],
                            "score": 0,
                        }

                    # Track which concept matched
                    matched_concept = chunk.get("matched_concept", "")
                    if matched_concept and matched_concept not in norm_scores[urn]["concept_matches"]:
                        norm_scores[urn]["concept_matches"].append(matched_concept)

                    # Score: +2 for phrase match, +1 for single keyword match
                    if ' ' in term:
                        norm_scores[urn]["score"] += 2
                    else:
                        norm_scores[urn]["score"] += 1

            except Exception as e:
                print(f"Warning: Concept query failed for '{term}': {e}")

        # Boost norms where rubrica contains key terms
        rubrica_boost_terms = [t for t in concept_terms if len(t) > 5][:3]
        for urn, data in norm_scores.items():
            rubrica = (data.get("rubrica") or "").lower()
            for term in rubrica_boost_terms:
                if term.lower() in rubrica:
                    data["score"] += 3  # Significant boost for rubrica match

        # Sort by score (descending) and return top results
        sorted_norms = sorted(
            norm_scores.values(),
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        return sorted_norms[:limit]

    async def _retrieve_via_rubrica(
        self, client, keywords: List[str], limit: int
    ) -> List[dict]:
        """
        Find norms where rubrica (article title) matches query terms.

        Rubrica matches are high precision - the title indicates what
        the article is about.
        """
        results = []

        for keyword in keywords[:3]:
            if len(keyword) < 4:  # Skip short keywords for rubrica search
                continue
            try:
                cypher = f"""
                MATCH (n:Norma)
                WHERE toLower(n.rubrica) CONTAINS '{keyword.lower()}'
                RETURN n.URN as urn,
                       n.testo_vigente as text,
                       n.numero_articolo as articolo,
                       n.tipo_documento as tipo_atto,
                       n.rubrica as rubrica
                LIMIT {limit}
                """
                response = client.execute_command(
                    "GRAPH.QUERY",
                    self.graph_name,
                    cypher,
                )
                chunks = self._parse_graph_response(response)
                results.extend(chunks)
            except Exception as e:
                print(f"Warning: Rubrica query failed for '{keyword}': {e}")

        return results

    async def _retrieve_via_text(
        self, client, keywords: List[str], limit: int
    ) -> List[dict]:
        """
        Fall back to text content search.

        This has high recall but lower precision - articles may mention
        a term without being primarily about it.
        """
        results = []

        for keyword in keywords[:2]:  # Limit text search to avoid noise
            if len(keyword) < 5:  # Skip short keywords
                continue
            try:
                cypher = f"""
                MATCH (n:Norma)
                WHERE toLower(n.testo_vigente) CONTAINS '{keyword.lower()}'
                RETURN n.URN as urn,
                       n.testo_vigente as text,
                       n.numero_articolo as articolo,
                       n.tipo_documento as tipo_atto,
                       n.rubrica as rubrica
                LIMIT {limit}
                """
                response = client.execute_command(
                    "GRAPH.QUERY",
                    self.graph_name,
                    cypher,
                )
                chunks = self._parse_graph_response(response)
                results.extend(chunks)
            except Exception as e:
                print(f"Warning: Text query failed for '{keyword}': {e}")

        return results

    def _build_concept_terms(self, keywords: List[str]) -> List[str]:
        """
        Build concept search terms from keywords.

        Combines adjacent keywords to form phrases that might match
        concept names like "risoluzione del contratto" or "risoluzione per inadempimento".
        """
        terms = []

        # Add individual keywords
        terms.extend(keywords)

        # Add related legal terms (synonyms/variants)
        legal_variants = {
            "risoluzione": ["risolubilità", "risolvibilità", "scioglimento"],
            "inadempimento": ["inadempiente", "inadempienza"],
            "contratto": ["contrattuale", "negozio"],
        }
        for kw in keywords:
            if kw in legal_variants:
                terms.extend(legal_variants[kw])

        # Add bigrams (2 adjacent keywords)
        for i in range(len(keywords) - 1):
            bigram = f"{keywords[i]} {keywords[i+1]}"
            terms.append(bigram)

        # Add trigrams (3 adjacent keywords)
        for i in range(len(keywords) - 2):
            trigram = f"{keywords[i]} {keywords[i+1]} {keywords[i+2]}"
            terms.append(trigram)

        # Sort by length descending (prefer longer, more specific phrases)
        terms.sort(key=len, reverse=True)

        return terms

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        import re

        # Stopwords to filter (NOT including "per" - it's important in legal phrases)
        stopwords = {
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
            "di", "a", "da", "in", "con", "su", "tra", "fra",
            "che", "e", "è", "sono", "cosa", "cos'è", "quale", "quali",
            "come", "quando", "dove", "perché", "chi", "del", "della",
            "dei", "delle", "dello", "degli", "al", "alla", "ai", "alle",
            "cos",  # from cos'è
        }

        # Clean and tokenize
        # Replace apostrophe with space, remove punctuation except hyphens
        cleaned = query.lower().replace("'", " ")
        cleaned = re.sub(r'[^\w\s-]', '', cleaned)  # Remove punctuation
        words = cleaned.split()

        # Filter stopwords but keep "per" (important in legal phrases)
        keywords = []
        for i, w in enumerate(words):
            if w in stopwords:
                continue
            if len(w) < 3 and w != "per":  # Keep "per" even though it's short
                continue
            keywords.append(w)

        return keywords

    def _parse_graph_response(self, response) -> List[dict]:
        """Parse FalkorDB GRAPH.QUERY response."""
        chunks = []
        try:
            if len(response) >= 2:
                data = response[1]
                for row in data:
                    if len(row) >= 5:
                        chunks.append({
                            "urn": row[0] if row[0] else "",
                            "text": row[1] if row[1] else "",
                            "articolo": row[2] if row[2] else "",
                            "tipo_atto": row[3] if row[3] else "",
                            "rubrica": row[4] if row[4] else "",
                        })
        except Exception as e:
            print(f"Warning: Failed to parse response: {e}")
        return chunks

    def _parse_concept_response(self, response) -> List[dict]:
        """Parse concept query response (includes matched_concept field)."""
        chunks = []
        try:
            if len(response) >= 2:
                data = response[1]
                for row in data:
                    if len(row) >= 6:
                        chunks.append({
                            "urn": row[0] if row[0] else "",
                            "text": row[1] if row[1] else "",
                            "articolo": row[2] if row[2] else "",
                            "tipo_atto": row[3] if row[3] else "",
                            "rubrica": row[4] if row[4] else "",
                            "matched_concept": row[5] if row[5] else "",
                        })
        except Exception as e:
            print(f"Warning: Failed to parse concept response: {e}")
        return chunks


class FalkorDBGraphTraverser:
    """Graph traverser using real FalkorDB data."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6380,
        graph_name: str = "merl_t_dev",
    ):
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        return self._client

    async def get_related(
        self,
        urn: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 1,
        limit: int = 10,
    ) -> List[dict]:
        """Get related nodes from the graph."""
        client = self._get_client()

        # Build Cypher query with inline parameters
        if relation_types:
            rel_filter = "|".join(relation_types)
            cypher = f"""
            MATCH (source)-[r:{rel_filter}]-(target)
            WHERE source.URN = '{urn}' OR source.node_id = '{urn}'
            RETURN source.URN as source_urn,
                   target.URN as target_urn,
                   type(r) as relation_type,
                   1.0 as weight
            LIMIT {limit}
            """
        else:
            cypher = f"""
            MATCH (source)-[r]-(target)
            WHERE source.URN = '{urn}' OR source.node_id = '{urn}'
            RETURN source.URN as source_urn,
                   target.URN as target_urn,
                   type(r) as relation_type,
                   1.0 as weight
            LIMIT {limit}
            """

        try:
            response = client.execute_command(
                "GRAPH.QUERY",
                self.graph_name,
                cypher,
            )
            return self._parse_relations(response)
        except Exception as e:
            print(f"Warning: Graph traversal failed: {e}")
            return []

    async def get_history(self, urn: str) -> List[dict]:
        """Get historical versions (if available)."""
        # For now, return empty - multivigenza not implemented
        return []

    async def get_neighbors(
        self,
        urn: str,
        relation_types: Optional[List[str]] = None,
        depth: int = 1,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring norms in the graph.

        Implements the GraphTraverser protocol expected by SystemicExpert.
        """
        client = self._get_client()

        # Build Cypher query based on depth
        if depth == 1:
            if relation_types:
                rel_filter = "|".join(relation_types)
                cypher = f"""
                MATCH (source)-[r:{rel_filter}]->(target)
                WHERE source.URN = '{urn}' OR source.node_id = '{urn}'
                RETURN target.URN as urn,
                       target.testo_vigente as text,
                       target.rubrica as rubrica,
                       target.numero_articolo as articolo,
                       type(r) as relation_type,
                       labels(target) as labels
                LIMIT {limit}
                """
            else:
                cypher = f"""
                MATCH (source)-[r]->(target)
                WHERE source.URN = '{urn}' OR source.node_id = '{urn}'
                RETURN target.URN as urn,
                       target.testo_vigente as text,
                       target.rubrica as rubrica,
                       target.numero_articolo as articolo,
                       type(r) as relation_type,
                       labels(target) as labels
                LIMIT {limit}
                """
        else:
            # Multi-hop traversal
            cypher = f"""
            MATCH path = (source)-[*1..{depth}]->(target)
            WHERE source.URN = '{urn}' OR source.node_id = '{urn}'
            WITH target, relationships(path) as rels
            RETURN target.URN as urn,
                   target.testo_vigente as text,
                   target.rubrica as rubrica,
                   target.numero_articolo as articolo,
                   [r in rels | type(r)] as relation_types,
                   labels(target) as labels
            LIMIT {limit}
            """

        try:
            response = client.execute_command(
                "GRAPH.QUERY",
                self.graph_name,
                cypher,
            )
            return self._parse_neighbors(response, depth)
        except Exception as e:
            print(f"Warning: get_neighbors failed for '{urn}': {e}")
            return []

    async def get_modifications(
        self,
        urn: str,
        as_of_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get modification history for a norm.

        Implements the GraphTraverser protocol expected by SystemicExpert.
        Looks for MODIFICA, ABROGA, SOSTITUISCE relations.
        """
        client = self._get_client()

        # Query for modification relations
        cypher = f"""
        MATCH (source)-[r:MODIFICA|ABROGA|SOSTITUISCE|modifica|abroga]->(target)
        WHERE target.URN = '{urn}' OR target.node_id = '{urn}'
        RETURN source.URN as modifier_urn,
               source.rubrica as modifier_rubrica,
               source.data_vigenza as data_vigenza,
               type(r) as modification_type,
               r.data_decorrenza as data_decorrenza,
               r.nota as nota
        ORDER BY r.data_decorrenza DESC
        LIMIT 20
        """

        try:
            response = client.execute_command(
                "GRAPH.QUERY",
                self.graph_name,
                cypher,
            )
            modifications = self._parse_modifications(response)

            # Filter by date if specified
            if as_of_date and modifications:
                modifications = [
                    m for m in modifications
                    if not m.get("data_decorrenza") or m["data_decorrenza"] <= as_of_date
                ]

            return modifications
        except Exception as e:
            print(f"Warning: get_modifications failed for '{urn}': {e}")
            return []

    def _parse_neighbors(self, response, depth: int = 1) -> List[Dict[str, Any]]:
        """Parse neighbors response from FalkorDB."""
        neighbors = []
        try:
            if len(response) >= 2:
                data = response[1]
                for row in data:
                    if depth == 1 and len(row) >= 6:
                        neighbors.append({
                            "urn": row[0] if row[0] else "",
                            "text": row[1] if row[1] else "",
                            "rubrica": row[2] if row[2] else "",
                            "articolo": row[3] if row[3] else "",
                            "relation_type": row[4] if row[4] else "",
                            "labels": row[5] if row[5] else [],
                        })
                    elif depth > 1 and len(row) >= 6:
                        neighbors.append({
                            "urn": row[0] if row[0] else "",
                            "text": row[1] if row[1] else "",
                            "rubrica": row[2] if row[2] else "",
                            "articolo": row[3] if row[3] else "",
                            "relation_types": row[4] if row[4] else [],
                            "labels": row[5] if row[5] else [],
                        })
        except Exception as e:
            print(f"Warning: Failed to parse neighbors: {e}")
        return neighbors

    def _parse_modifications(self, response) -> List[Dict[str, Any]]:
        """Parse modifications response from FalkorDB."""
        modifications = []
        try:
            if len(response) >= 2:
                data = response[1]
                for row in data:
                    if len(row) >= 6:
                        modifications.append({
                            "modifier_urn": row[0] if row[0] else "",
                            "modifier_rubrica": row[1] if row[1] else "",
                            "data_vigenza": row[2] if row[2] else "",
                            "modification_type": row[3] if row[3] else "",
                            "data_decorrenza": row[4] if row[4] else "",
                            "nota": row[5] if row[5] else "",
                        })
        except Exception as e:
            print(f"Warning: Failed to parse modifications: {e}")
        return modifications

    def _parse_relations(self, response) -> List[dict]:
        """Parse graph relations response."""
        relations = []
        try:
            if len(response) >= 2:
                data = response[1]
                for row in data:
                    if len(row) >= 4:
                        relations.append({
                            "source_urn": row[0] if row[0] else "",
                            "target_urn": row[1] if row[1] else "",
                            "relation_type": row[2] if row[2] else "",
                            "weight": float(row[3]) if row[3] else 1.0,
                        })
        except Exception as e:
            print(f"Warning: Failed to parse relations: {e}")

        return relations


# =============================================================================
# Mock LLM (for testing without API key)
# =============================================================================


class MockLLMService:
    """Mock LLM that generates realistic responses based on retrieved data."""

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate mock response."""
        prompt_lower = prompt.lower()

        if "letterale" in prompt_lower or "literal" in prompt_lower:
            return self._literal_response(prompt)
        elif "sistematic" in prompt_lower:
            return self._systemic_response(prompt)
        elif "principi" in prompt_lower or "ratio" in prompt_lower:
            return self._principles_response(prompt)
        elif "giurisprudenz" in prompt_lower or "precedent" in prompt_lower:
            return self._precedent_response(prompt)
        elif "sintetizz" in prompt_lower or "synthesi" in prompt_lower:
            return self._synthesis_response(prompt)
        else:
            return "Risposta basata sui dati recuperati dal knowledge graph."

    def _literal_response(self, prompt: str) -> str:
        return (
            "**Interpretazione Letterale**\n\n"
            "Analizzando il testo normativo secondo il criterio letterale previsto "
            "dall'art. 12 delle Preleggi, emerge che la disposizione in esame deve "
            "essere interpretata nel suo significato proprio.\n\n"
            "Le fonti recuperate dal knowledge graph confermano questa interpretazione."
        )

    def _systemic_response(self, prompt: str) -> str:
        return (
            "**Interpretazione Sistematica**\n\n"
            "Dal punto di vista sistematico, la norma si inserisce nel quadro "
            "complessivo dell'ordinamento giuridico italiano.\n\n"
            "L'analisi delle relazioni nel knowledge graph evidenzia collegamenti "
            "con altre disposizioni normative che confermano questa lettura."
        )

    def _principles_response(self, prompt: str) -> str:
        return (
            "**Ratio Legis e Principi**\n\n"
            "La disposizione trova fondamento nei principi costituzionali e nella "
            "ratio legis sottostante.\n\n"
            "I principi giuridici estratti dal knowledge graph supportano questa "
            "interpretazione teleologica."
        )

    def _precedent_response(self, prompt: str) -> str:
        return (
            "**Giurisprudenza**\n\n"
            "La giurisprudenza consolidata conferma l'interpretazione proposta.\n\n"
            "Gli atti giudiziari presenti nel knowledge graph forniscono orientamenti "
            "applicativi rilevanti."
        )

    def _synthesis_response(self, prompt: str) -> str:
        return (
            "Sulla base dell'analisi multi-expert condotta, emerge una interpretazione "
            "convergente che combina il criterio letterale, sistematico, teleologico "
            "e giurisprudenziale. Le fonti recuperate dal knowledge graph MERL-T "
            "supportano questa conclusione."
        )


# =============================================================================
# Orchestrator Factory
# =============================================================================


def create_live_orchestrator(
    use_live_llm: bool = False,
    falkordb_host: str = "localhost",
    falkordb_port: int = 6380,
    graph_name: str = "merl_t_dev",
    timeout_ms: float = 60000.0,
) -> PipelineOrchestrator:
    """Create orchestrator with real FalkorDB data."""

    # Create real retrievers
    retriever = FalkorDBRetriever(
        host=falkordb_host,
        port=falkordb_port,
        graph_name=graph_name,
    )
    graph_traverser = FalkorDBGraphTraverser(
        host=falkordb_host,
        port=falkordb_port,
        graph_name=graph_name,
    )

    # Create LLM service
    if use_live_llm:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required for --live-llm mode")
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
    )

    # Create experts with real data
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
# Main
# =============================================================================


async def generate_trace(
    query: str,
    user_profile: str = "ricerca",
    use_live_llm: bool = False,
    timeout_ms: float = 60000.0,
) -> Dict[str, Any]:
    """Generate trace with live FalkorDB data."""

    orchestrator = create_live_orchestrator(
        use_live_llm=use_live_llm,
        timeout_ms=timeout_ms,
    )

    request = PipelineRequest(
        query=query,
        user_profile=user_profile,
    )

    result = await orchestrator.process_query(request)

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "query": query,
            "user_profile": user_profile,
            "mode": "live_llm" if use_live_llm else "live_graph_mock_llm",
            "data_source": "FalkorDB merl_t_dev (27k+ nodes, 43k+ relations)",
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
    parser = argparse.ArgumentParser(
        description="Generate Pipeline Trace with LIVE FalkorDB data",
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="Cos'è la risoluzione del contratto per inadempimento?",
        help="Legal query to process",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="ricerca",
        choices=["consulenza", "ricerca", "analisi", "contributore"],
        help="User profile",
    )

    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Use live LLM (requires OPENROUTER_API_KEY)",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output",
    )

    args = parser.parse_args()

    # Check FalkorDB connection
    try:
        client = redis.Redis(host="localhost", port=6380)
        client.ping()
    except Exception as e:
        print(f"Error: Cannot connect to FalkorDB on port 6380: {e}")
        print("Make sure merl-t-falkordb-dev container is running.")
        sys.exit(1)

    try:
        trace = asyncio.run(
            generate_trace(
                query=args.query,
                user_profile=args.profile,
                use_live_llm=args.live_llm,
            )
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    indent = None if args.compact else 2
    json_output = json.dumps(trace, ensure_ascii=False, indent=indent)

    if args.output:
        Path(args.output).write_text(json_output, encoding="utf-8")
        print(f"Trace written to: {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
