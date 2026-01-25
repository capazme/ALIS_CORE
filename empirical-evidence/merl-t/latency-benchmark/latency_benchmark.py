#!/usr/bin/env python3
"""
MERL-T Latency Benchmark Script

Analyzes latency metrics from existing pipeline traces.
Calculates percentile statistics (p50, p95, p99) and breakdowns by expert.

Usage:
    python latency_benchmark.py

Inputs:
    - ../expert-pipeline-trace/pipeline_traces.json

Outputs:
    - latency_results.json: Structured latency analysis data
    - latency_report.md: Human-readable latency report
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def calculate_percentile(data: list[float], percentile: int) -> float:
    """Calculate the specified percentile of a dataset."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = (percentile / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    weight = index - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def calculate_statistics(latencies: list[float]) -> dict[str, float]:
    """Calculate comprehensive statistics for a list of latency values."""
    if not latencies:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_dev_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

    return {
        "count": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2),
        "std_dev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0,
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "p50_ms": round(calculate_percentile(latencies, 50), 2),
        "p95_ms": round(calculate_percentile(latencies, 95), 2),
        "p99_ms": round(calculate_percentile(latencies, 99), 2),
    }


def extract_latencies(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract all latency data from traces."""
    total_latencies = []
    expert_latencies = {
        "literal": [],
        "systemic": [],
        "principles": [],
        "precedent": [],
    }

    # Track per-trace expert latencies for breakdown analysis
    trace_details = []

    for trace in traces:
        total_latency = trace.get("total_latency_ms", 0)
        if total_latency > 0:
            total_latencies.append(total_latency)

        trace_info = {
            "query": trace.get("query", "")[:50] + "...",
            "timestamp": trace.get("timestamp", ""),
            "total_latency_ms": round(total_latency, 2),
            "experts": {},
        }

        for expert_trace in trace.get("expert_traces", []):
            expert_name = expert_trace.get("expert_name", "")
            latency = expert_trace.get("latency_ms", 0)

            if expert_name in expert_latencies and latency > 0:
                expert_latencies[expert_name].append(latency)
                trace_info["experts"][expert_name] = round(latency, 2)

        trace_details.append(trace_info)

    return {
        "total_latencies": total_latencies,
        "expert_latencies": expert_latencies,
        "trace_details": trace_details,
    }


def analyze_latencies(extracted: dict[str, Any]) -> dict[str, Any]:
    """Perform full latency analysis."""
    total_stats = calculate_statistics(extracted["total_latencies"])

    expert_stats = {}
    for expert_name, latencies in extracted["expert_latencies"].items():
        expert_stats[expert_name] = calculate_statistics(latencies)

    # Calculate expert contribution to total latency
    expert_contributions = {}
    if total_stats["mean_ms"] > 0:
        for expert_name, stats in expert_stats.items():
            contribution_pct = (stats["mean_ms"] / total_stats["mean_ms"]) * 100
            expert_contributions[expert_name] = round(contribution_pct, 2)

    # Identify slowest and fastest experts
    expert_means = {name: stats["mean_ms"] for name, stats in expert_stats.items() if stats["count"] > 0}
    slowest_expert = max(expert_means, key=expert_means.get) if expert_means else None
    fastest_expert = min(expert_means, key=expert_means.get) if expert_means else None

    return {
        "pipeline_total": total_stats,
        "by_expert": expert_stats,
        "expert_contributions_pct": expert_contributions,
        "insights": {
            "slowest_expert": slowest_expert,
            "slowest_expert_mean_ms": expert_means.get(slowest_expert, 0),
            "fastest_expert": fastest_expert,
            "fastest_expert_mean_ms": expert_means.get(fastest_expert, 0),
            "latency_spread_ms": round(total_stats["max_ms"] - total_stats["min_ms"], 2),
        },
    }


def generate_report(analysis: dict[str, Any], trace_count: int) -> str:
    """Generate a markdown report from the analysis."""
    report_lines = [
        "# MERL-T Latency Benchmark Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Traces Analyzed:** {trace_count}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Median Pipeline Latency (p50):** {analysis['pipeline_total']['p50_ms']:,.2f} ms ({analysis['pipeline_total']['p50_ms']/1000:.2f} s)",
        f"- **95th Percentile (p95):** {analysis['pipeline_total']['p95_ms']:,.2f} ms ({analysis['pipeline_total']['p95_ms']/1000:.2f} s)",
        f"- **99th Percentile (p99):** {analysis['pipeline_total']['p99_ms']:,.2f} ms ({analysis['pipeline_total']['p99_ms']/1000:.2f} s)",
        f"- **Slowest Expert:** `{analysis['insights']['slowest_expert']}` (avg: {analysis['insights']['slowest_expert_mean_ms']:,.2f} ms)",
        f"- **Fastest Expert:** `{analysis['insights']['fastest_expert']}` (avg: {analysis['insights']['fastest_expert_mean_ms']:,.2f} ms)",
        "",
        "---",
        "",
        "## Pipeline Total Latency",
        "",
        "| Metric | Value (ms) | Value (s) |",
        "|--------|-----------|-----------|",
        f"| Mean | {analysis['pipeline_total']['mean_ms']:,.2f} | {analysis['pipeline_total']['mean_ms']/1000:.2f} |",
        f"| Std Dev | {analysis['pipeline_total']['std_dev_ms']:,.2f} | {analysis['pipeline_total']['std_dev_ms']/1000:.2f} |",
        f"| Min | {analysis['pipeline_total']['min_ms']:,.2f} | {analysis['pipeline_total']['min_ms']/1000:.2f} |",
        f"| Max | {analysis['pipeline_total']['max_ms']:,.2f} | {analysis['pipeline_total']['max_ms']/1000:.2f} |",
        f"| **p50 (Median)** | **{analysis['pipeline_total']['p50_ms']:,.2f}** | **{analysis['pipeline_total']['p50_ms']/1000:.2f}** |",
        f"| **p95** | **{analysis['pipeline_total']['p95_ms']:,.2f}** | **{analysis['pipeline_total']['p95_ms']/1000:.2f}** |",
        f"| **p99** | **{analysis['pipeline_total']['p99_ms']:,.2f}** | **{analysis['pipeline_total']['p99_ms']/1000:.2f}** |",
        "",
        "---",
        "",
        "## Latency by Expert",
        "",
    ]

    experts = ["literal", "systemic", "principles", "precedent"]

    for expert in experts:
        stats = analysis["by_expert"].get(expert, {})
        contribution = analysis["expert_contributions_pct"].get(expert, 0)

        report_lines.extend([
            f"### {expert.capitalize()} Expert",
            "",
            f"**Contribution to Pipeline:** {contribution:.1f}%",
            "",
            "| Metric | Value (ms) |",
            "|--------|-----------|",
            f"| Count | {stats.get('count', 0)} |",
            f"| Mean | {stats.get('mean_ms', 0):,.2f} |",
            f"| Std Dev | {stats.get('std_dev_ms', 0):,.2f} |",
            f"| Min | {stats.get('min_ms', 0):,.2f} |",
            f"| Max | {stats.get('max_ms', 0):,.2f} |",
            f"| **p50** | **{stats.get('p50_ms', 0):,.2f}** |",
            f"| **p95** | **{stats.get('p95_ms', 0):,.2f}** |",
            f"| **p99** | **{stats.get('p99_ms', 0):,.2f}** |",
            "",
        ])

    # Expert comparison table
    report_lines.extend([
        "---",
        "",
        "## Expert Comparison",
        "",
        "| Expert | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Contribution |",
        "|--------|-----------|----------|----------|----------|--------------|",
    ])

    for expert in experts:
        stats = analysis["by_expert"].get(expert, {})
        contribution = analysis["expert_contributions_pct"].get(expert, 0)
        report_lines.append(
            f"| {expert.capitalize()} | {stats.get('mean_ms', 0):,.2f} | "
            f"{stats.get('p50_ms', 0):,.2f} | {stats.get('p95_ms', 0):,.2f} | "
            f"{stats.get('p99_ms', 0):,.2f} | {contribution:.1f}% |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## Observations",
        "",
        f"1. **Latency Spread:** The difference between fastest and slowest pipeline runs is "
        f"{analysis['insights']['latency_spread_ms']:,.2f} ms ({analysis['insights']['latency_spread_ms']/1000:.2f} s)",
        "",
        f"2. **Bottleneck Analysis:** The `{analysis['insights']['slowest_expert']}` expert is the slowest "
        f"on average, contributing most to overall pipeline latency.",
        "",
        f"3. **Optimization Target:** Reducing latency in the `{analysis['insights']['slowest_expert']}` expert "
        f"would have the highest impact on overall performance.",
        "",
        f"4. **Consistency:** Standard deviation of {analysis['pipeline_total']['std_dev_ms']:,.2f} ms indicates "
        f"{'high' if analysis['pipeline_total']['std_dev_ms'] > 5000 else 'moderate' if analysis['pipeline_total']['std_dev_ms'] > 2000 else 'low'} "
        f"variability in response times.",
        "",
        "---",
        "",
        "*Report generated by MERL-T Latency Benchmark*",
    ])

    return "\n".join(report_lines)


def main():
    """Main entry point for the latency benchmark."""
    # Paths
    script_dir = Path(__file__).parent
    traces_path = script_dir.parent / "expert-pipeline-trace" / "pipeline_traces.json"
    results_path = script_dir / "latency_results.json"
    report_path = script_dir / "latency_report.md"

    print(f"MERL-T Latency Benchmark")
    print(f"========================")
    print(f"Reading traces from: {traces_path}")

    # Load traces
    if not traces_path.exists():
        print(f"ERROR: Traces file not found at {traces_path}")
        return 1

    with open(traces_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    traces = data.get("traces", [])
    print(f"Found {len(traces)} traces to analyze")

    if not traces:
        print("ERROR: No traces found in the file")
        return 1

    # Extract and analyze latencies
    print("Extracting latency data...")
    extracted = extract_latencies(traces)

    print("Calculating statistics...")
    analysis = analyze_latencies(extracted)

    # Build results JSON
    results = {
        "generated_at": datetime.now().isoformat(),
        "source_file": str(traces_path),
        "traces_analyzed": len(traces),
        "analysis": analysis,
        "trace_details": extracted["trace_details"],
    }

    # Write results JSON
    print(f"Writing results to: {results_path}")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate and write report
    print(f"Generating report: {report_path}")
    report = generate_report(analysis, len(traces))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Print summary
    print()
    print("Summary:")
    print(f"  - Pipeline p50: {analysis['pipeline_total']['p50_ms']:,.2f} ms")
    print(f"  - Pipeline p95: {analysis['pipeline_total']['p95_ms']:,.2f} ms")
    print(f"  - Pipeline p99: {analysis['pipeline_total']['p99_ms']:,.2f} ms")
    print(f"  - Slowest expert: {analysis['insights']['slowest_expert']}")
    print(f"  - Fastest expert: {analysis['insights']['fastest_expert']}")
    print()
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
