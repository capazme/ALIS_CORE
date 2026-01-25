#!/usr/bin/env python3
"""
A/B Simulation v2 - RLCF vs Baseline (Metodologia Corretta)

Questa simulazione dimostra il vantaggio dell'aggregazione pesata per autorità (RLCF)
rispetto all'aggregazione uniforme (baseline) in modo metodologicamente corretto.

PRINCIPIO CHIAVE:
- Gli utenti con alta autorità forniscono feedback PIU' ACCURATI (bassa varianza)
- Gli utenti con bassa autorità forniscono feedback MENO ACCURATI (alta varianza)
- RLCF pesa i feedback per autorità, quindi dovrebbe avvicinarsi di più al ground truth

METODOLOGIA:
1. Genera utenti con authority score seguendo distribuzione Pareto
2. La varianza del feedback è INVERSAMENTE proporzionale all'autorità
3. Esegue N trial indipendenti per significatività statistica
4. Calcola intervalli di confidenza (95%)
5. Riporta risultati onestamente

Output:
- ab_results_v2.json: Risultati dettagliati
- ab_simulation_report_v2.md: Report formattato
"""

import json
import math
import random
import statistics
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configurazione
@dataclass
class SimulationConfig:
    """Configurazione della simulazione.

    Parametri calibrati su letteratura scientifica:
    - MIT "Surprisingly Popular" Algorithm: 21-35% error reduction
    - Contribution Weighted Model (Management Science): 28-39% improvement
    - Pareto α=1.5 riflette distribuzione 80-20 reale dell'expertise

    References:
    - https://news.mit.edu/2017/algorithm-better-wisdom-crowds-0125
    - https://pubsonline.informs.org/doi/10.1287/mnsc.2014.1909
    """
    num_users: int = 100
    num_tasks: int = 100
    num_trials: int = 30  # Per significatività statistica
    raters_per_task: Tuple[int, int] = (5, 20)  # Range valutatori per task (realistico)
    ground_truth_range: Tuple[float, float] = (1.0, 5.0)  # Scala rating

    # CRITICO: Relazione autorità-accuratezza
    # Calibrato su studi che mostrano esperti 15-20x più accurati in domini specializzati
    # (medicina, diritto, finanza) - vedi PMC7865038, Management Science 2014
    base_noise_std: float = 1.5  # Deviazione standard base (37% della scala - alto rumore)
    authority_noise_factor: float = 0.95  # Expert 20x più accurato (domini specializzati)
    # noise_std = base_noise_std * (1 - authority * authority_noise_factor)
    # authority=1.0 → noise_std = 0.075 (altissima accuratezza, senior expert)
    # authority=0.0 → noise_std = 1.5 (alta varianza, novice/random)

    pareto_alpha: float = 1.3  # Heavy-tail: ~15% produce ~85% del valore (realistico)
    random_seed: int = 42


@dataclass
class User:
    """Utente con authority score."""
    user_id: int
    authority_score: float  # [0, 1]
    noise_std: float  # Deviazione standard del suo feedback


@dataclass
class TrialResult:
    """Risultato di un singolo trial."""
    trial_id: int
    rlcf_mae: float  # Mean Absolute Error
    baseline_mae: float
    rlcf_rmse: float  # Root Mean Squared Error
    baseline_rmse: float
    improvement_mae: float  # (baseline - rlcf) / baseline * 100


@dataclass
class SimulationResults:
    """Risultati aggregati della simulazione."""
    timestamp: str
    config: Dict[str, Any]
    num_trials: int

    # Metriche aggregate
    rlcf_mae_mean: float
    rlcf_mae_std: float
    rlcf_mae_ci95: Tuple[float, float]

    baseline_mae_mean: float
    baseline_mae_std: float
    baseline_mae_ci95: Tuple[float, float]

    improvement_mean: float
    improvement_std: float
    improvement_ci95: Tuple[float, float]

    # Statistiche
    rlcf_wins: int
    baseline_wins: int
    ties: int

    # Trial details
    trials: List[TrialResult]


def generate_users(config: SimulationConfig) -> List[User]:
    """Genera utenti con authority score e noise correlato."""
    users = []
    for i in range(config.num_users):
        # Authority segue Pareto normalizzata [0, 1]
        raw_authority = random.paretovariate(config.pareto_alpha)
        authority = min(1.0, raw_authority / (raw_authority + 1))

        # CRITICO: Noise inversamente proporzionale all'autorità
        noise_std = config.base_noise_std * (1 - authority * config.authority_noise_factor)

        users.append(User(
            user_id=i,
            authority_score=authority,
            noise_std=noise_std
        ))
    return users


def generate_feedback(user: User, ground_truth: float, scale_range: Tuple[float, float]) -> float:
    """Genera feedback di un utente con noise proporzionale alla sua (in)competenza."""
    noise = random.gauss(0, user.noise_std)
    rating = ground_truth + noise
    # Clamp to scale
    return max(scale_range[0], min(scale_range[1], rating))


def aggregate_rlcf(feedbacks: List[Tuple[float, float]]) -> float:
    """
    Aggregazione RLCF: media pesata per autorità.

    Formula: R = Σ(rating_i × authority_i) / Σ(authority_i)
    """
    weighted_sum = sum(rating * authority for rating, authority in feedbacks)
    weight_sum = sum(authority for _, authority in feedbacks)
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def aggregate_baseline(feedbacks: List[Tuple[float, float]]) -> float:
    """
    Aggregazione Baseline: media semplice (ignora autorità).

    Formula: R = (1/N) × Σ(rating_i)
    """
    ratings = [rating for rating, _ in feedbacks]
    return statistics.mean(ratings) if ratings else 0.0


def run_single_trial(trial_id: int, users: List[User], config: SimulationConfig) -> TrialResult:
    """Esegue un singolo trial della simulazione."""
    rlcf_errors = []
    baseline_errors = []

    for task_id in range(config.num_tasks):
        # Ground truth casuale
        ground_truth = random.uniform(*config.ground_truth_range)

        # Seleziona raters casuali
        num_raters = random.randint(*config.raters_per_task)
        raters = random.sample(users, min(num_raters, len(users)))

        # Genera feedback
        feedbacks = []
        for user in raters:
            rating = generate_feedback(user, ground_truth, config.ground_truth_range)
            feedbacks.append((rating, user.authority_score))

        # Aggrega con entrambi i metodi
        rlcf_result = aggregate_rlcf(feedbacks)
        baseline_result = aggregate_baseline(feedbacks)

        # Calcola errori
        rlcf_errors.append(abs(rlcf_result - ground_truth))
        baseline_errors.append(abs(baseline_result - ground_truth))

    # Metriche
    rlcf_mae = statistics.mean(rlcf_errors)
    baseline_mae = statistics.mean(baseline_errors)
    rlcf_rmse = math.sqrt(statistics.mean([e**2 for e in rlcf_errors]))
    baseline_rmse = math.sqrt(statistics.mean([e**2 for e in baseline_errors]))

    improvement = (baseline_mae - rlcf_mae) / baseline_mae * 100 if baseline_mae > 0 else 0

    return TrialResult(
        trial_id=trial_id,
        rlcf_mae=rlcf_mae,
        baseline_mae=baseline_mae,
        rlcf_rmse=rlcf_rmse,
        baseline_rmse=baseline_rmse,
        improvement_mae=improvement
    )


def confidence_interval_95(data: List[float]) -> Tuple[float, float]:
    """Calcola intervallo di confidenza al 95% (t-distribution)."""
    n = len(data)
    if n < 2:
        return (data[0], data[0]) if data else (0, 0)

    mean = statistics.mean(data)
    std = statistics.stdev(data)
    se = std / math.sqrt(n)

    # t-value per 95% CI con df = n-1
    # Per n=30 (df=29), t_0.975 = 2.045
    # Per n>30, converge verso z=1.96
    # Usiamo lookup table per precisione
    t_table = {
        10: 2.262, 15: 2.131, 20: 2.086, 25: 2.060,
        30: 2.045, 40: 2.021, 50: 2.009, 60: 2.000,
        100: 1.984, 120: 1.980
    }
    # Trova il valore più vicino
    closest_n = min(t_table.keys(), key=lambda x: abs(x - n))
    t_value = t_table.get(closest_n, 1.96)

    margin = t_value * se
    return (mean - margin, mean + margin)


def run_simulation(config: SimulationConfig) -> SimulationResults:
    """Esegue la simulazione completa."""
    random.seed(config.random_seed)

    # Genera utenti (fissi per tutti i trial)
    users = generate_users(config)

    # Esegui trial
    trials = []
    for trial_id in range(config.num_trials):
        trial = run_single_trial(trial_id, users, config)
        trials.append(trial)

    # Aggrega risultati
    rlcf_maes = [t.rlcf_mae for t in trials]
    baseline_maes = [t.baseline_mae for t in trials]
    improvements = [t.improvement_mae for t in trials]

    # Conta vittorie
    rlcf_wins = sum(1 for t in trials if t.rlcf_mae < t.baseline_mae)
    baseline_wins = sum(1 for t in trials if t.baseline_mae < t.rlcf_mae)
    ties = sum(1 for t in trials if abs(t.rlcf_mae - t.baseline_mae) < 0.0001)

    return SimulationResults(
        timestamp=datetime.now().isoformat(),
        config=asdict(config),
        num_trials=config.num_trials,
        rlcf_mae_mean=statistics.mean(rlcf_maes),
        rlcf_mae_std=statistics.stdev(rlcf_maes),
        rlcf_mae_ci95=confidence_interval_95(rlcf_maes),
        baseline_mae_mean=statistics.mean(baseline_maes),
        baseline_mae_std=statistics.stdev(baseline_maes),
        baseline_mae_ci95=confidence_interval_95(baseline_maes),
        improvement_mean=statistics.mean(improvements),
        improvement_std=statistics.stdev(improvements),
        improvement_ci95=confidence_interval_95(improvements),
        rlcf_wins=rlcf_wins,
        baseline_wins=baseline_wins,
        ties=ties,
        trials=trials
    )


def generate_report(results: SimulationResults, output_path: Path):
    """Genera report Markdown."""
    cfg = results.config

    md = f"""# A/B Simulation Report v2 - RLCF vs Baseline

**Generated**: {results.timestamp}
**Methodology**: Statistically rigorous with {results.num_trials} independent trials

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Users | {cfg['num_users']} |
| Tasks per Trial | {cfg['num_tasks']} |
| Trials | {cfg['num_trials']} |
| Raters per Task | {cfg['raters_per_task'][0]}-{cfg['raters_per_task'][1]} |
| Base Noise σ | {cfg['base_noise_std']} |
| Authority Noise Factor | {cfg['authority_noise_factor']} |
| Random Seed | {cfg['random_seed']} |

### Noise Model

The key insight of RLCF is that expert users provide more accurate feedback:

```
noise_std = base_noise_std × (1 - authority × authority_noise_factor)
```

| Authority | Noise σ | Interpretation |
|-----------|---------|----------------|
| 0.0 (novice) | {cfg['base_noise_std']:.2f} | High variance feedback |
| 0.5 (intermediate) | {cfg['base_noise_std'] * (1 - 0.5 * cfg['authority_noise_factor']):.2f} | Medium variance |
| 1.0 (expert) | {cfg['base_noise_std'] * (1 - 1.0 * cfg['authority_noise_factor']):.2f} | Low variance feedback |

---

## Results Summary

### Mean Absolute Error (MAE)

| Method | MAE Mean | MAE Std | 95% CI |
|--------|----------|---------|--------|
| **RLCF** | **{results.rlcf_mae_mean:.4f}** | {results.rlcf_mae_std:.4f} | [{results.rlcf_mae_ci95[0]:.4f}, {results.rlcf_mae_ci95[1]:.4f}] |
| Baseline | {results.baseline_mae_mean:.4f} | {results.baseline_mae_std:.4f} | [{results.baseline_mae_ci95[0]:.4f}, {results.baseline_mae_ci95[1]:.4f}] |

### Improvement

| Metric | Value |
|--------|-------|
| Mean Improvement | **{results.improvement_mean:.2f}%** |
| Std | {results.improvement_std:.2f}% |
| 95% CI | [{results.improvement_ci95[0]:.2f}%, {results.improvement_ci95[1]:.2f}%] |

### Win Rate

| Outcome | Count | Percentage |
|---------|-------|------------|
| RLCF Wins | {results.rlcf_wins} | {results.rlcf_wins/results.num_trials*100:.1f}% |
| Baseline Wins | {results.baseline_wins} | {results.baseline_wins/results.num_trials*100:.1f}% |
| Ties | {results.ties} | {results.ties/results.num_trials*100:.1f}% |

---

## Statistical Significance

"""
    # Check if improvement CI excludes 0
    if results.improvement_ci95[0] > 0:
        md += f"""**Result**: RLCF significantly outperforms baseline (p < 0.05)

The 95% confidence interval for improvement [{results.improvement_ci95[0]:.2f}%, {results.improvement_ci95[1]:.2f}%]
does not include zero, indicating **statistically significant improvement**.
"""
    elif results.improvement_ci95[1] < 0:
        md += """**Result**: Baseline outperforms RLCF (unexpected)

This would indicate a problem with the methodology or implementation.
"""
    else:
        md += f"""**Result**: No statistically significant difference

The 95% confidence interval [{results.improvement_ci95[0]:.2f}%, {results.improvement_ci95[1]:.2f}%]
includes zero, meaning the difference could be due to random chance.
"""

    md += f"""

---

## Methodology Notes

### Why RLCF Should Outperform Baseline

1. **Authority-Accuracy Correlation**: In this simulation, users with higher authority scores
   have lower noise variance in their feedback. This models real-world expertise.

2. **Weighted Aggregation**: RLCF weights feedback by authority:
   `R = Σ(rating × authority) / Σ(authority)`

   This gives more weight to accurate expert feedback.

3. **Baseline Ignores Expertise**: Simple averaging treats all feedback equally,
   diluting expert signal with novice noise.

### Limitations

- This is a **simulation** with synthetic data
- Real-world authority-accuracy correlation may differ
- Results depend on the noise model parameters

---

## Trial Data (First 10)

| Trial | RLCF MAE | Baseline MAE | Improvement |
|-------|----------|--------------|-------------|
"""
    for t in results.trials[:10]:
        md += f"| {t.trial_id} | {t.rlcf_mae:.4f} | {t.baseline_mae:.4f} | {t.improvement_mae:.2f}% |\n"

    md += f"""

---

## References

- Allega, D., & Puzio, G. (2025c). RLCF Paper, Section 3.1: Dynamic Authority Scoring
- Formula: A_u(t) = α·B_u + β·T_u(t-1) + γ·P_u(t)
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)


def main():
    """Esegue la simulazione."""
    print("A/B Simulation v2 - RLCF vs Baseline")
    print("=" * 50)

    config = SimulationConfig()
    print(f"Running {config.num_trials} trials with {config.num_users} users, {config.num_tasks} tasks each...")

    results = run_simulation(config)

    # Output directory
    output_dir = Path(__file__).parent

    # Salva JSON
    json_path = output_dir / "ab_results_v2.json"

    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(to_dict(results), f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")

    # Genera report
    md_path = output_dir / "ab_simulation_report_v2.md"
    generate_report(results, md_path)
    print(f"Markdown saved: {md_path}")

    # Summary
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"RLCF MAE:     {results.rlcf_mae_mean:.4f} ± {results.rlcf_mae_std:.4f}")
    print(f"Baseline MAE: {results.baseline_mae_mean:.4f} ± {results.baseline_mae_std:.4f}")
    print(f"Improvement:  {results.improvement_mean:.2f}% (95% CI: [{results.improvement_ci95[0]:.2f}%, {results.improvement_ci95[1]:.2f}%])")
    print(f"RLCF Wins:    {results.rlcf_wins}/{results.num_trials} ({results.rlcf_wins/results.num_trials*100:.1f}%)")

    if results.improvement_ci95[0] > 0:
        print("\n✓ STATISTICALLY SIGNIFICANT: RLCF outperforms baseline (p < 0.05)")
    else:
        print("\n⚠ NOT SIGNIFICANT: Difference may be due to random chance")


if __name__ == "__main__":
    main()
