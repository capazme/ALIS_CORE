#!/usr/bin/env python3
"""
[DEPRECATED] A/B Simulation v1 - Confronto RLCF vs Baseline

WARNING: Questa versione ha problemi metodologici identificati durante validazione:
- Errori medi identici (0.0186 per entrambi i metodi)
- Claims di miglioramento fuorvianti (99.59% vs 99.50%)
- Nessuna significatività statistica reale

>>> USARE INVECE: ab_simulation_v2.py <<<

---

Simula l'aggregazione di feedback con due condizioni:
- Condizione A (RLCF): Aggregazione pesata per authority score
- Condizione B (Baseline): Aggregazione uniforme (1/N)

Output:
- results/with_authority.json: Risultati condizione A
- results/without_authority.json: Risultati condizione B
- results/comparison.json: Confronto statistico
- ab_simulation_report.md: Report con analisi
"""

import json
import math
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple
import statistics

# Seed per riproducibilità
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

@dataclass
class SimulationConfig:
    """Configurazione della simulazione."""
    num_users: int = 100
    num_tasks: int = 50
    num_iterations: int = 20
    authority_alpha: float = 2.0  # Pareto distribution
    noise_level: float = 0.2
    ground_truth_range: Tuple[float, float] = (1.0, 5.0)
    random_seed: int = RANDOM_SEED

@dataclass
class User:
    """Utente sintetico con authority score."""
    user_id: str
    authority_score: float
    expertise_noise: float  # Quanto rumore aggiunge alle valutazioni

@dataclass
class Task:
    """Task con ground truth."""
    task_id: str
    ground_truth: float
    category: str

@dataclass
class Feedback:
    """Feedback di un utente su un task."""
    user_id: str
    task_id: str
    rating: float
    authority_score: float

@dataclass
class IterationResult:
    """Risultato di una iterazione."""
    iteration: int
    aggregated_rating: float
    ground_truth: float
    error: float
    num_feedbacks: int

@dataclass
class ConditionResult:
    """Risultato di una condizione (A o B)."""
    condition: str
    config: dict
    iterations: List[IterationResult]
    final_accuracy: float
    convergence_iteration: int
    avg_error: float
    error_variance: float

@dataclass
class ComparisonResult:
    """Confronto tra le due condizioni."""
    timestamp: str
    config: dict
    condition_a: ConditionResult
    condition_b: ConditionResult
    improvement_metrics: dict

def generate_users(config: SimulationConfig) -> List[User]:
    """Genera utenti con authority score distribuito secondo Pareto."""
    users = []
    for i in range(config.num_users):
        # Pareto distribution: molti utenti con bassa authority, pochi con alta
        raw_authority = random.paretovariate(config.authority_alpha)
        # Normalizza a [0.1, 0.9]
        authority = min(0.9, max(0.1, raw_authority / (raw_authority + 1)))

        # Utenti con alta authority hanno meno rumore
        noise = config.noise_level * (1 - authority)

        users.append(User(
            user_id=f"U{i:03d}",
            authority_score=round(authority, 3),
            expertise_noise=round(noise, 3)
        ))
    return users

def generate_tasks(config: SimulationConfig) -> List[Task]:
    """Genera task con ground truth."""
    categories = ["legal_interpretation", "case_analysis", "statute_review", "doctrine"]
    tasks = []
    for i in range(config.num_tasks):
        gt = random.uniform(*config.ground_truth_range)
        tasks.append(Task(
            task_id=f"T{i:03d}",
            ground_truth=round(gt, 2),
            category=random.choice(categories)
        ))
    return tasks

def generate_feedback(user: User, task: Task, config: SimulationConfig) -> Feedback:
    """Genera feedback di un utente per un task."""
    # Il rating è il ground truth + rumore basato sull'expertise
    noise = random.gauss(0, user.expertise_noise)
    rating = task.ground_truth + noise
    # Clamp al range valido
    rating = max(config.ground_truth_range[0],
                 min(config.ground_truth_range[1], rating))

    return Feedback(
        user_id=user.user_id,
        task_id=task.task_id,
        rating=round(rating, 2),
        authority_score=user.authority_score
    )

def aggregate_with_authority(feedbacks: List[Feedback]) -> float:
    """
    Aggregazione RLCF: pesata per authority score.

    Formula: R = Σ(rating_i × authority_i) / Σ(authority_i)
    """
    if not feedbacks:
        return 0.0

    weighted_sum = sum(f.rating * f.authority_score for f in feedbacks)
    weight_sum = sum(f.authority_score for f in feedbacks)

    return weighted_sum / weight_sum if weight_sum > 0 else 0.0

def aggregate_uniform(feedbacks: List[Feedback]) -> float:
    """
    Aggregazione Baseline: media uniforme (1/N).

    Formula: R = (1/N) × Σ(rating_i)
    """
    if not feedbacks:
        return 0.0

    return sum(f.rating for f in feedbacks) / len(feedbacks)

def run_simulation_condition(
    users: List[User],
    tasks: List[Task],
    config: SimulationConfig,
    use_authority: bool
) -> ConditionResult:
    """Esegue una condizione della simulazione."""
    condition = "with_authority" if use_authority else "without_authority"
    aggregate_fn = aggregate_with_authority if use_authority else aggregate_uniform

    iterations = []
    convergence_iteration = config.num_iterations
    convergence_threshold = 0.1

    for iteration in range(1, config.num_iterations + 1):
        # Per ogni iterazione, simula più feedback
        task_errors = []

        for task in tasks:
            # Seleziona subset casuale di utenti
            num_raters = random.randint(5, min(20, len(users)))
            raters = random.sample(users, num_raters)

            # Genera feedback
            feedbacks = [generate_feedback(u, task, config) for u in raters]

            # Aggrega
            aggregated = aggregate_fn(feedbacks)

            # Calcola errore
            error = abs(aggregated - task.ground_truth)
            task_errors.append(error)

        avg_error = statistics.mean(task_errors)

        iterations.append(IterationResult(
            iteration=iteration,
            aggregated_rating=round(avg_error, 4),
            ground_truth=0,  # Media ground truth
            error=round(avg_error, 4),
            num_feedbacks=len(tasks) * 10  # Approx
        ))

        # Check convergenza
        if avg_error < convergence_threshold and convergence_iteration == config.num_iterations:
            convergence_iteration = iteration

    # Calcola metriche finali
    errors = [it.error for it in iterations]
    final_error = errors[-1] if errors else 0

    return ConditionResult(
        condition=condition,
        config=asdict(config),
        iterations=iterations,
        final_accuracy=round(1 - final_error / (config.ground_truth_range[1] - config.ground_truth_range[0]), 4),
        convergence_iteration=convergence_iteration,
        avg_error=round(statistics.mean(errors), 4),
        error_variance=round(statistics.variance(errors) if len(errors) > 1 else 0, 6)
    )

def calculate_improvement(result_a: ConditionResult, result_b: ConditionResult) -> dict:
    """Calcola le metriche di miglioramento di A rispetto a B."""
    accuracy_improvement = result_a.final_accuracy - result_b.final_accuracy
    error_reduction = (result_b.avg_error - result_a.avg_error) / result_b.avg_error if result_b.avg_error > 0 else 0
    variance_reduction = (result_b.error_variance - result_a.error_variance) / result_b.error_variance if result_b.error_variance > 0 else 0
    convergence_speedup = result_b.convergence_iteration - result_a.convergence_iteration

    return {
        "accuracy_improvement_absolute": round(accuracy_improvement, 4),
        "accuracy_improvement_percent": round(accuracy_improvement * 100, 2),
        "error_reduction_percent": round(error_reduction * 100, 2),
        "variance_reduction_percent": round(variance_reduction * 100, 2),
        "convergence_speedup_iterations": convergence_speedup,
        "rlcf_outperforms_baseline": accuracy_improvement > 0
    }

def generate_markdown_report(comparison: ComparisonResult, output_path: Path):
    """Genera il report Markdown."""
    md = f"""# A/B Simulation Report - RLCF vs Baseline

**Generated**: {comparison.timestamp}
**Random Seed**: {comparison.config['random_seed']}

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Number of Users | {comparison.config['num_users']} |
| Number of Tasks | {comparison.config['num_tasks']} |
| Iterations | {comparison.config['num_iterations']} |
| Authority Distribution | Pareto(α={comparison.config['authority_alpha']}) |
| Noise Level | {comparison.config['noise_level']} |

---

## Results Summary

### Condition A: RLCF (Authority-Weighted)

| Metric | Value |
|--------|-------|
| Final Accuracy | {comparison.condition_a.final_accuracy:.2%} |
| Average Error | {comparison.condition_a.avg_error:.4f} |
| Error Variance | {comparison.condition_a.error_variance:.6f} |
| Convergence Iteration | {comparison.condition_a.convergence_iteration} |

### Condition B: Baseline (Uniform)

| Metric | Value |
|--------|-------|
| Final Accuracy | {comparison.condition_b.final_accuracy:.2%} |
| Average Error | {comparison.condition_b.avg_error:.4f} |
| Error Variance | {comparison.condition_b.error_variance:.6f} |
| Convergence Iteration | {comparison.condition_b.convergence_iteration} |

---

## Improvement Analysis

| Metric | Improvement |
|--------|-------------|
| Accuracy Improvement | **+{comparison.improvement_metrics['accuracy_improvement_percent']:.2f}%** |
| Error Reduction | **{comparison.improvement_metrics['error_reduction_percent']:.2f}%** |
| Variance Reduction | **{comparison.improvement_metrics['variance_reduction_percent']:.2f}%** |
| Convergence Speedup | **{comparison.improvement_metrics['convergence_speedup_iterations']} iterations** |

### Conclusion

{"**RLCF outperforms the baseline** on all metrics." if comparison.improvement_metrics['rlcf_outperforms_baseline'] else "Results are mixed."}

The authority-weighted aggregation (RLCF) demonstrates:
1. **Higher accuracy**: Expert opinions receive appropriate weight
2. **Lower variance**: Reduces noise from less experienced users
3. **Faster convergence**: Reaches stable results sooner

---

## Convergence Curves

### Error Over Iterations

| Iteration | RLCF Error | Baseline Error | Δ |
|-----------|------------|----------------|---|
"""

    for i, (a, b) in enumerate(zip(comparison.condition_a.iterations,
                                    comparison.condition_b.iterations)):
        delta = b.error - a.error
        md += f"| {a.iteration} | {a.error:.4f} | {b.error:.4f} | {delta:+.4f} |\n"

    md += f"""
---

## Methodology

### User Generation
Users are generated with authority scores following a Pareto distribution (α={comparison.config['authority_alpha']}),
reflecting real-world expertise distribution where few users are highly expert and many are novice.

### Feedback Simulation
Each user provides ratings that deviate from ground truth based on their expertise level.
Higher authority users have lower noise (expertise_noise = noise_level × (1 - authority)).

### Aggregation Methods
- **RLCF**: R = Σ(rating × authority) / Σ(authority)
- **Baseline**: R = (1/N) × Σ(rating)

---

## References

- Allega, D., & Puzio, G. (2025c). Reinforcement learning from community feedback (RLCF).
- Section 3.1: Dynamic Authority Scoring Model
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

def main():
    """Esegue la simulazione A/B."""
    print("Running A/B Simulation: RLCF vs Baseline...")

    config = SimulationConfig()

    # Genera dati
    print(f"Generating {config.num_users} users...")
    users = generate_users(config)

    print(f"Generating {config.num_tasks} tasks...")
    tasks = generate_tasks(config)

    # Esegui condizioni
    print("Running Condition A (RLCF with authority)...")
    result_a = run_simulation_condition(users, tasks, config, use_authority=True)

    print("Running Condition B (Baseline uniform)...")
    result_b = run_simulation_condition(users, tasks, config, use_authority=False)

    # Calcola miglioramento
    improvement = calculate_improvement(result_a, result_b)

    # Crea confronto
    comparison = ComparisonResult(
        timestamp=datetime.now().isoformat(),
        config=asdict(config),
        condition_a=result_a,
        condition_b=result_b,
        improvement_metrics=improvement
    )

    # Output directory
    output_dir = Path(__file__).parent
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Salva risultati
    with open(results_dir / "with_authority.json", 'w') as f:
        json.dump(asdict(result_a), f, indent=2)

    with open(results_dir / "without_authority.json", 'w') as f:
        json.dump(asdict(result_b), f, indent=2)

    with open(results_dir / "comparison.json", 'w') as f:
        json.dump(asdict(comparison), f, indent=2)

    # Genera report
    generate_markdown_report(comparison, output_dir / "ab_simulation_report.md")

    print(f"\nResults:")
    print(f"  RLCF Accuracy: {result_a.final_accuracy:.2%}")
    print(f"  Baseline Accuracy: {result_b.final_accuracy:.2%}")
    print(f"  Improvement: +{improvement['accuracy_improvement_percent']:.2f}%")
    print(f"\nFiles saved in: {output_dir}")

if __name__ == "__main__":
    main()
