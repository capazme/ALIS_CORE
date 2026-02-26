"""Factory for generating QATrace test data."""

import random
import uuid
from datetime import datetime, timezone, timedelta


EXPERT_TYPES = ["literal", "systemic", "principles", "precedent"]

_SAMPLE_QUERIES = [
    "Cos'e' la responsabilita' contrattuale secondo l'art. 1218 c.c.?",
    "Quali sono i presupposti della legittima difesa ex art. 52 c.p.?",
    "Come si applica il principio di proporzionalita' nel diritto amministrativo?",
    "Qual e' la differenza tra dolo eventuale e colpa cosciente?",
    "In cosa consiste il danno biologico ex art. 2059 c.c.?",
]

_SAMPLE_SOURCES = [
    {"article_urn": "urn:nir:stato:codice.civile:1942-03-16;262~art1218", "expert": "literal", "relevance": 0.95},
    {"article_urn": "urn:nir:stato:codice.penale:1930-10-19;1398~art52", "expert": "systemic", "relevance": 0.88},
    {"article_urn": "urn:nir:stato:costituzione:1947-12-27~art3", "expert": "principles", "relevance": 0.92},
]


def create_expert_response(expert_type: str = "literal", **overrides) -> dict:
    """Create a single expert response dict.

    Args:
        expert_type: One of literal, systemic, principles, precedent.
        **overrides: Override any field.
    """
    defaults = {
        "expert_name": expert_type,
        "answer": f"Risposta dell'esperto {expert_type} alla query.",
        "confidence": round(random.uniform(0.5, 1.0), 3),
        "sources": random.sample(_SAMPLE_SOURCES, k=min(2, len(_SAMPLE_SOURCES))),
        "reasoning": f"L'esperto {expert_type} ha analizzato le fonti pertinenti.",
        "execution_time_ms": random.randint(200, 3000),
    }
    defaults.update(overrides)
    return defaults


def create_trace(**overrides) -> dict:
    """Create a QATrace dict.

    Returns dict with: trace_id, query, expert_responses (list of 4),
    sources, synthesis, confidence, created_at, and additional metadata.
    """
    expert_responses = [create_expert_response(et) for et in EXPERT_TYPES]
    all_sources = []
    for resp in expert_responses:
        all_sources.extend(resp["sources"])

    defaults = {
        "trace_id": f"trace_{uuid.uuid4().hex[:12]}",
        "user_id": str(uuid.uuid4()),
        "query": random.choice(_SAMPLE_QUERIES),
        "selected_experts": list(EXPERT_TYPES),
        "expert_responses": expert_responses,
        "synthesis_mode": random.choice(["convergent", "divergent"]),
        "synthesis_text": "Sintesi delle risposte degli esperti sulla questione giuridica.",
        "sources": all_sources,
        "confidence": round(random.uniform(0.5, 1.0), 3),
        "execution_time_ms": random.randint(1000, 8000),
        "consent_level": "basic",
        "query_type": random.choice(["definitional", "interpretive", "comparative"]),
        "routing_method": random.choice(["neural", "llm_fallback", "regex"]),
        "created_at": (
            datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 720))
        ).isoformat(),
    }
    defaults.update(overrides)
    return defaults
