"""
Performance Baseline Benchmarks
================================

pytest-benchmark tests for key CPU-bound operations.
Establishes performance baselines for regression detection.

Run: pytest tests/benchmark/test_performance_baselines.py --benchmark-only
"""

import importlib
import importlib.util
import json
import os
import sys
import pytest
from uuid import uuid4


def _load_urn_parser():
    """Load urn_parser module directly from file to avoid circular import."""
    mod_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "merlt", "citation", "urn_parser.py"
    )
    mod_path = os.path.abspath(mod_path)
    spec = importlib.util.spec_from_file_location("urn_parser_direct", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_urn_parser = _load_urn_parser()


# =============================================================================
# URN PARSING BENCHMARK
# =============================================================================

@pytest.mark.benchmark(group="urn-parsing")
def test_parse_urn_codice_civile(benchmark):
    """Benchmark URN parsing for codice civile article."""
    result = benchmark(_urn_parser.parse_urn, "urn:nir:stato:codice.civile:1942;art1453")

    assert result.parsed_successfully
    assert result.article == "1453"
    assert result.is_codice


@pytest.mark.benchmark(group="urn-parsing")
def test_parse_urn_decreto_legislativo(benchmark):
    """Benchmark URN parsing for decreto legislativo."""
    result = benchmark(
        _urn_parser.parse_urn,
        "urn:nir:stato:decreto.legislativo:2003-06-30;196~art1",
    )

    assert result.parsed_successfully
    assert result.act_type == "decreto.legislativo"


# =============================================================================
# PII MASKING BENCHMARK
# =============================================================================

@pytest.mark.benchmark(group="pii-masking")
def test_mask_pii_clean_text(benchmark):
    """Benchmark PII masking on text without PII."""
    from merlt.rlcf.pii_service import PIIMaskingService

    svc = PIIMaskingService()
    text = "L'articolo 1453 del codice civile disciplina la risoluzione del contratto per inadempimento."

    result = benchmark(svc.mask_text, text)

    assert result == text  # No PII to mask


@pytest.mark.benchmark(group="pii-masking")
def test_mask_pii_with_patterns(benchmark):
    """Benchmark PII masking on text with all PII types."""
    from merlt.rlcf.pii_service import PIIMaskingService

    svc = PIIMaskingService()
    text = (
        "Contattare RSSMRA85A01H501Z al numero +39 333 1234567 "
        "o via email mario.rossi@example.it entro il 15/03/2024."
    )

    result = benchmark(svc.mask_text, text)

    assert "[CF]" in result
    assert "[EMAIL]" in result
    assert "[TELEFONO]" in result
    assert "[DATA]" in result


# =============================================================================
# CITATION FORMATTING BENCHMARK
# =============================================================================

@pytest.mark.benchmark(group="citation-format")
def test_format_italian_date(benchmark):
    """Benchmark Italian date formatting."""
    result = benchmark(_urn_parser.format_italian_date, "1990-08-07")

    assert result == "7 agosto 1990"


@pytest.mark.benchmark(group="citation-format")
def test_get_codice_abbreviation(benchmark):
    """Benchmark codice abbreviation lookup."""
    result = benchmark(_urn_parser.get_codice_abbreviation, "codice.civile")

    assert result == "c.c."


# =============================================================================
# BRIDGE TABLE BATCH PREPARE BENCHMARK
# =============================================================================

@pytest.mark.benchmark(group="bridge-table")
def test_prepare_batch_mappings(benchmark):
    """Benchmark preparing batch mapping parameters (no DB required)."""

    def prepare_batch(mappings):
        params = []
        for m in mappings:
            metadata_json = json.dumps(m.get("metadata")) if m.get("metadata") else None
            params.append({
                "chunk_id": str(m["chunk_id"]),
                "graph_node_urn": m["graph_node_urn"],
                "node_type": m["node_type"],
                "relation_type": m.get("relation_type"),
                "confidence": m.get("confidence"),
                "chunk_text": m.get("chunk_text"),
                "source": m.get("source"),
                "metadata": metadata_json,
            })
        return params

    mappings = [
        {
            "chunk_id": uuid4(),
            "graph_node_urn": f"urn:nir:stato:codice.civile:1942;art{i}",
            "node_type": "Norma",
            "relation_type": "contained_in",
            "confidence": 0.95,
            "source": "visualex",
            "metadata": {"batch": True, "idx": i},
        }
        for i in range(100)
    ]

    result = benchmark(prepare_batch, mappings)

    assert len(result) == 100
    assert result[0]["node_type"] == "Norma"
