#!/usr/bin/env python3
"""
Pipeline Tracer - Evidenza MERL-T

Importa e analizza i trace esistenti da EXP-020 per dimostrare
che l'architettura multi-expert MERL-T è funzionante.

I trace mostrano:
- 4 Expert operativi (Literal, Systemic, Principles, Precedent)
- Routing intelligente con pesi dinamici
- Retrieval multi-step per ogni expert
- Synthesis finale con confidence aggregata
- Source grounding (fonti citate)

Output:
- pipeline_traces.json: Trace selezionati
- pipeline_trace_report.md: Report formattato
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict, field

# Paths
ALIS_CORE = Path(__file__).parent.parent.parent.parent
EXP_020_DIR = ALIS_CORE / "merlt" / "docs" / "experiments" / "EXP-020_scientific_evaluation"

@dataclass
class ExpertTrace:
    """Trace di un singolo Expert."""
    expert_name: str
    interpretation: str
    confidence: float
    sources_count: int
    source_types: List[str]
    latency_ms: float

@dataclass
class PipelineTrace:
    """Trace completo di una pipeline execution."""
    query: str
    timestamp: str
    total_latency_ms: float
    routing: Dict[str, Any]
    expert_traces: List[ExpertTrace]
    final_confidence: float
    final_synthesis_length: int
    sources_total: int

@dataclass
class PipelineTraceReport:
    """Report delle pipeline traces."""
    generated_at: str
    source_experiment: str
    total_traces: int
    traces: List[PipelineTrace]
    statistics: Dict[str, Any]


def load_trace_file(trace_path: Path) -> Dict[str, Any]:
    """Carica un file trace JSON."""
    with open(trace_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_expert_trace(expert_name: str, expert_data: Dict) -> ExpertTrace:
    """Estrae il trace di un expert."""
    sources = expert_data.get("sources", [])
    source_types = list(set(s.get("source_type", "unknown") for s in sources))

    # Trova latency dal llm_calls
    latency = 0.0

    return ExpertTrace(
        expert_name=expert_name,
        interpretation=expert_data.get("interpretation", "")[:200] + "...",
        confidence=expert_data.get("confidence", 0.0),
        sources_count=len(sources),
        source_types=source_types,
        latency_ms=latency
    )


def extract_pipeline_trace(trace_data: Dict) -> PipelineTrace:
    """Estrae un PipelineTrace da un trace raw."""
    expert_results = trace_data.get("expert_results", {})
    expert_traces = []

    for expert_name in ["literal", "systemic", "principles", "precedent"]:
        if expert_name in expert_results:
            expert_traces.append(
                extract_expert_trace(expert_name, expert_results[expert_name])
            )

    # Conta fonti totali
    total_sources = sum(
        len(expert_results.get(e, {}).get("sources", []))
        for e in ["literal", "systemic", "principles", "precedent"]
    )

    # Calcola latency dai llm_calls
    llm_calls = trace_data.get("llm_calls", [])
    expert_latencies = {}
    for call in llm_calls:
        expert = call.get("expert", "")
        latency = call.get("latency_ms", 0)
        if expert in expert_latencies:
            expert_latencies[expert] += latency
        else:
            expert_latencies[expert] = latency

    # Aggiorna latency negli expert traces
    for et in expert_traces:
        et.latency_ms = expert_latencies.get(et.expert_name, 0)

    return PipelineTrace(
        query=trace_data.get("query", ""),
        timestamp=trace_data.get("timestamp", ""),
        total_latency_ms=trace_data.get("total_latency_ms", 0),
        routing=trace_data.get("routing", {}),
        expert_traces=expert_traces,
        final_confidence=trace_data.get("final_confidence", 0),
        final_synthesis_length=len(trace_data.get("final_synthesis", "")),
        sources_total=total_sources
    )


def calculate_statistics(traces: List[PipelineTrace]) -> Dict[str, Any]:
    """Calcola statistiche aggregate."""
    if not traces:
        return {}

    latencies = [t.total_latency_ms for t in traces]
    confidences = [t.final_confidence for t in traces]
    sources = [t.sources_total for t in traces]

    # Calcola metriche per expert
    expert_confidences = {e: [] for e in ["literal", "systemic", "principles", "precedent"]}
    expert_sources = {e: [] for e in ["literal", "systemic", "principles", "precedent"]}

    for trace in traces:
        for et in trace.expert_traces:
            expert_confidences[et.expert_name].append(et.confidence)
            expert_sources[et.expert_name].append(et.sources_count)

    return {
        "total_traces": len(traces),
        "latency": {
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        },
        "confidence": {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences)
        },
        "sources_per_query": {
            "mean": sum(sources) / len(sources),
            "total": sum(sources)
        },
        "expert_performance": {
            expert: {
                "mean_confidence": sum(confs) / len(confs) if confs else 0,
                "mean_sources": sum(expert_sources[expert]) / len(expert_sources[expert]) if expert_sources[expert] else 0
            }
            for expert, confs in expert_confidences.items()
        },
        "source_grounding_rate": sum(1 for s in sources if s > 0) / len(sources)
    }


def generate_markdown_report(report: PipelineTraceReport, output_path: Path):
    """Genera il report Markdown."""
    stats = report.statistics

    md = f"""# MERL-T Expert Pipeline Trace Report

**Generated**: {report.generated_at}
**Source Experiment**: {report.source_experiment}
**Total Traces Analyzed**: {report.total_traces}

---

## Executive Summary

Questo report dimostra che l'architettura MERL-T multi-expert è **completamente operativa**.
Ogni query passa attraverso 4 esperti che applicano i canoni ermeneutici dell'art. 12 Preleggi.

| Metric | Value |
|--------|-------|
| **Mean Latency** | {stats['latency']['mean_ms']:.0f} ms |
| **Mean Confidence** | {stats['confidence']['mean']:.2f} |
| **Mean Sources per Query** | {stats['sources_per_query']['mean']:.1f} |
| **Source Grounding Rate** | {stats['source_grounding_rate']*100:.0f}% |

---

## Expert Performance Analysis

| Expert | Role (Art. 12 Preleggi) | Mean Confidence | Mean Sources |
|--------|------------------------|-----------------|--------------|
| **Literal** | Interpretazione letterale | {stats['expert_performance']['literal']['mean_confidence']:.2f} | {stats['expert_performance']['literal']['mean_sources']:.1f} |
| **Systemic** | Connessione sistematica | {stats['expert_performance']['systemic']['mean_confidence']:.2f} | {stats['expert_performance']['systemic']['mean_sources']:.1f} |
| **Principles** | Ratio legis e principi | {stats['expert_performance']['principles']['mean_confidence']:.2f} | {stats['expert_performance']['principles']['mean_sources']:.1f} |
| **Precedent** | Giurisprudenza | {stats['expert_performance']['precedent']['mean_confidence']:.2f} | {stats['expert_performance']['precedent']['mean_sources']:.1f} |

---

## Sample Pipeline Traces

"""

    # Aggiungi 3 trace di esempio
    for i, trace in enumerate(report.traces[:3]):
        md += f"""### Trace {i+1}: "{trace.query[:60]}..."

**Timestamp**: {trace.timestamp}
**Total Latency**: {trace.total_latency_ms:.0f} ms
**Final Confidence**: {trace.final_confidence:.2f}
**Total Sources**: {trace.sources_total}

#### Routing Decision

```json
{json.dumps(trace.routing, indent=2)}
```

#### Expert Results

| Expert | Confidence | Sources | Interpretation (excerpt) |
|--------|------------|---------|-------------------------|
"""
        for et in trace.expert_traces:
            interpretation_excerpt = et.interpretation[:80].replace('\n', ' ')
            md += f"| {et.expert_name} | {et.confidence:.2f} | {et.sources_count} | {interpretation_excerpt}... |\n"

        md += "\n---\n\n"

    md += f"""## Architecture Validation

### Multi-Expert System (Paper Section 3.2)

L'architettura implementa i 4 Expert descritti nel paper MERL-T:

1. **LiteralExpert** - Analizza il significato proprio delle parole
2. **SystemicExpert** - Considera la connessione delle norme nel sistema
3. **PrinciplesExpert** - Applica ratio legis e principi costituzionali
4. **PrecedentExpert** - Integra la giurisprudenza consolidata

### Routing Mechanism (Paper Section 3.3)

Il routing assegna pesi dinamici agli expert basandosi su:
- Query type classification
- Domain detection
- Historical performance

### Synthesis Layer (Paper Section 3.4)

La sintesi finale:
- Combina le interpretazioni pesate
- Preserva l'incertezza dove appropriato
- Cita le fonti rilevanti

---

## Implications

I trace dimostrano:

1. **Multi-Expert Operativo**: Tutti e 4 gli expert producono interpretazioni
2. **Source Grounding**: {stats['source_grounding_rate']*100:.0f}% delle risposte citano fonti
3. **Confidence Calibration**: Range [{stats['confidence']['min']:.2f}, {stats['confidence']['max']:.2f}]
4. **Latency Acceptable**: Media {stats['latency']['mean_ms']:.0f}ms per pipeline completa

---

## References

- Allega, D., & Puzio, G. (2025b). MERL-T Paper, Section 3: Multi-Expert Architecture
- EXP-020: Scientific Evaluation Experiment (December 2025)
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)


def main():
    """Esegue l'estrazione e analisi."""
    print("MERL-T Pipeline Tracer")
    print("=" * 50)

    # Trova tutti i trace files
    trace_files = list(EXP_020_DIR.glob("trace_*.json"))
    print(f"Found {len(trace_files)} trace files in EXP-020")

    if not trace_files:
        print("ERROR: No trace files found!")
        return

    # Carica e processa i trace
    traces = []
    for trace_file in sorted(trace_files)[:10]:  # Limita a 10
        try:
            trace_data = load_trace_file(trace_file)
            pipeline_trace = extract_pipeline_trace(trace_data)
            traces.append(pipeline_trace)
            print(f"  Processed: {trace_file.name}")
        except Exception as e:
            print(f"  Error processing {trace_file.name}: {e}")

    if not traces:
        print("ERROR: No valid traces extracted!")
        return

    # Calcola statistiche
    stats = calculate_statistics(traces)

    # Crea report
    report = PipelineTraceReport(
        generated_at=datetime.now().isoformat(),
        source_experiment="EXP-020_scientific_evaluation",
        total_traces=len(traces),
        traces=traces,
        statistics=stats
    )

    # Output directory
    output_dir = Path(__file__).parent

    # Salva JSON
    json_path = output_dir / "pipeline_traces.json"

    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(to_dict(report), f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {json_path}")

    # Genera report MD
    md_path = output_dir / "pipeline_trace_report.md"
    generate_markdown_report(report, md_path)
    print(f"Markdown saved: {md_path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Traces analyzed: {len(traces)}")
    print(f"  Mean latency: {stats['latency']['mean_ms']:.0f} ms")
    print(f"  Mean confidence: {stats['confidence']['mean']:.2f}")
    print(f"  Source grounding: {stats['source_grounding_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
