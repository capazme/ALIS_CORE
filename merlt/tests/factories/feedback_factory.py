"""Factory for generating test feedback across all 8 types (F1-F8)."""

import random
import uuid
from datetime import datetime, timezone


# Feedback type definitions: type_code -> (description, default fields builder)
FEEDBACK_TYPES = {
    "F1": "NER correction",
    "F2": "Router classification",
    "F3": "Literal expert output",
    "F4": "Systemic expert output",
    "F5": "Principles expert output",
    "F6": "Precedent expert output",
    "F7": "Synthesizer output",
    "F8": "Bridge quality",
}


def _base_feedback(feedback_type: str, **overrides) -> dict:
    """Common fields for all feedback types."""
    defaults = {
        "feedback_id": str(uuid.uuid4()),
        "feedback_type": feedback_type,
        "trace_id": f"trace_{uuid.uuid4().hex[:12]}",
        "user_id": str(uuid.uuid4()),
        "user_authority": round(random.uniform(0.1, 1.0), 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(overrides)
    return defaults


def _ner_feedback(**overrides) -> dict:
    """F1: NER correction feedback."""
    data = _base_feedback("F1", **overrides)
    data.setdefault("selected_text", "art. 2043 c.c.")
    data.setdefault("correction_type", random.choice(["confirmation", "correction", "annotation"]))
    data.setdefault("original_parsed", {"tipo_atto": "codice civile", "articolo": "2043"})
    data.setdefault(
        "correct_reference",
        {"tipo_atto": "codice civile", "numero": "262", "anno": "1942", "articoli": ["2043"]},
    )
    return data


def _router_feedback(**overrides) -> dict:
    """F2: Router classification feedback."""
    data = _base_feedback("F2", **overrides)
    data.setdefault("predicted_experts", ["literal", "systemic"])
    data.setdefault("correct_experts", ["literal", "precedent"])
    data.setdefault("rating", random.randint(1, 5))
    data.setdefault("reason", "Il router ha selezionato systemic ma serviva precedent")
    return data


def _expert_feedback(feedback_type: str, expert_name: str, **overrides) -> dict:
    """F3-F6: Expert output feedback."""
    data = _base_feedback(feedback_type, **overrides)
    data.setdefault("expert_name", expert_name)
    data.setdefault("rating", random.randint(1, 5))
    data.setdefault("thumbs", random.choice(["up", "down"]))
    data.setdefault("reason", f"Risposta dell'esperto {expert_name} accurata")
    data.setdefault("retrieval_score", round(random.uniform(0.0, 1.0), 2))
    data.setdefault("reasoning_score", round(random.uniform(0.0, 1.0), 2))
    return data


def _synthesizer_feedback(**overrides) -> dict:
    """F7: Synthesizer output feedback."""
    data = _base_feedback("F7", **overrides)
    data.setdefault("synthesis_score", round(random.uniform(0.0, 1.0), 2))
    data.setdefault("rating", random.randint(1, 5))
    data.setdefault("thumbs", random.choice(["up", "down"]))
    data.setdefault("preferred_expert", random.choice(["literal", "systemic", "principles", "precedent"]))
    data.setdefault("detailed_comment", "La sintesi integra bene le diverse prospettive")
    return data


def _bridge_feedback(**overrides) -> dict:
    """F8: Bridge quality feedback."""
    data = _base_feedback("F8", **overrides)
    data.setdefault("source_id", f"urn:nir:stato:legge:2024-01-15;12~art{random.randint(1, 200)}")
    data.setdefault("source_relevance", random.randint(1, 5))
    data.setdefault("reason", "Il collegamento tra chunk e nodo grafo e' corretto")
    return data


_BUILDERS = {
    "F1": _ner_feedback,
    "F2": _router_feedback,
    "F3": lambda **kw: _expert_feedback("F3", "literal", **kw),
    "F4": lambda **kw: _expert_feedback("F4", "systemic", **kw),
    "F5": lambda **kw: _expert_feedback("F5", "principles", **kw),
    "F6": lambda **kw: _expert_feedback("F6", "precedent", **kw),
    "F7": _synthesizer_feedback,
    "F8": _bridge_feedback,
}


def create_feedback(feedback_type: str = "F3", **overrides) -> dict:
    """Create a single feedback dict for the given type (F1-F8).

    Args:
        feedback_type: One of F1-F8 (default: F3 literal expert).
        **overrides: Override any field in the returned dict.

    Returns:
        dict matching the feedback model for the given type.
    """
    builder = _BUILDERS.get(feedback_type)
    if builder is None:
        raise ValueError(f"Unknown feedback_type {feedback_type!r}. Use F1-F8.")
    return builder(**overrides)
