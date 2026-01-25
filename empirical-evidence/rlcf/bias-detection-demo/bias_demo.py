#!/usr/bin/env python3
"""
Bias Detection Demo - Evidenza RLCF

Dimostra il funzionamento del sistema di rilevamento bias a 6 dimensioni.
Genera feedback sintetici con bias intenzionali e mostra che il sistema li rileva.

Le 6 dimensioni di bias:
1. b1: Demographic bias (concentrazione per gruppo demografico)
2. b2: Professional clustering (concentrazione per professione)
3. b3: Temporal drift (cambiamento opinioni nel tempo)
4. b4: Geographic concentration (concentrazione regionale)
5. b5: Confirmation bias (tendenza a confermare opinioni precedenti)
6. b6: Anchoring bias (influenza delle prime risposte)

Formula: B_total = sqrt(sum(b_i^2))

Output:
- synthetic_feedbacks.json: Dataset sintetico con bias
- bias_report.json: Output 6-dimensional del detector
- bias_detection_report.md: Report con analisi
"""

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from collections import defaultdict

# Seed per riproducibilità
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

@dataclass
class UserProfile:
    """Profilo utente per analisi bias."""
    user_id: str
    profession: str
    region: str
    experience_years: int
    specialization: str

@dataclass
class SyntheticFeedback:
    """Feedback sintetico con metadata per bias analysis."""
    feedback_id: str
    user_id: str
    task_id: str
    position: str  # "correct", "partially_correct", "incorrect"
    rating: float
    timestamp: str
    user_profile: UserProfile

@dataclass
class BiasScore:
    """Score per una singola dimensione di bias."""
    dimension: str
    score: float
    details: str
    contributing_factors: List[str]

@dataclass
class BiasReport:
    """Report completo del bias detection."""
    task_id: str
    timestamp: str
    num_feedbacks_analyzed: int
    bias_scores: Dict[str, float]
    total_bias_score: float
    bias_level: str  # "low", "medium", "high"
    formula_used: str
    dimension_details: List[BiasScore]
    mitigation_recommendations: List[str]

def generate_user_profiles(num_users: int = 30) -> List[UserProfile]:
    """Genera profili utente con distribuzioni realistiche ma sbilanciate (per demo bias)."""
    professions = ["avvocato", "magistrato", "notaio", "accademico", "praticante"]
    regions = ["lombardia", "lazio", "campania", "sicilia", "piemonte", "veneto"]
    specializations = ["civile", "penale", "amministrativo", "commerciale", "lavoro"]

    users = []
    for i in range(num_users):
        # Bias intenzionale: più avvocati lombardi
        if i < num_users * 0.4:  # 40% avvocati
            profession = "avvocato"
        elif i < num_users * 0.6:  # 20% magistrati
            profession = "magistrato"
        else:
            profession = random.choice(professions)

        if i < num_users * 0.5:  # 50% lombardi (bias geografico)
            region = "lombardia"
        else:
            region = random.choice(regions)

        users.append(UserProfile(
            user_id=f"U{i:03d}",
            profession=profession,
            region=region,
            experience_years=random.randint(1, 30),
            specialization=random.choice(specializations)
        ))

    return users

def generate_synthetic_feedbacks(
    users: List[UserProfile],
    task_id: str = "DEMO_TASK",
    num_feedbacks: int = 50
) -> List[SyntheticFeedback]:
    """Genera feedback sintetici con bias intenzionali."""
    feedbacks = []
    positions = ["correct", "partially_correct", "incorrect"]
    base_time = datetime.now() - timedelta(days=7)

    for i in range(num_feedbacks):
        user = random.choice(users)

        # Bias intenzionale nelle posizioni:
        # - Avvocati tendono a "correct"
        # - Magistrati tendono a "partially_correct"
        # - Altri più variabili
        if user.profession == "avvocato":
            position = random.choices(positions, weights=[0.6, 0.3, 0.1])[0]
        elif user.profession == "magistrato":
            position = random.choices(positions, weights=[0.3, 0.5, 0.2])[0]
        else:
            position = random.choice(positions)

        # Bias temporale: le prime risposte influenzano le successive
        if i < 10:  # Prime 10 risposte
            # Anchor verso "correct"
            if random.random() < 0.3:
                position = "correct"

        # Timestamp con drift temporale
        timestamp = base_time + timedelta(hours=i * 3 + random.randint(0, 2))

        rating = {"correct": 5.0, "partially_correct": 3.0, "incorrect": 1.0}[position]
        rating += random.uniform(-0.5, 0.5)  # Noise

        feedbacks.append(SyntheticFeedback(
            feedback_id=f"FB{i:03d}",
            user_id=user.user_id,
            task_id=task_id,
            position=position,
            rating=round(rating, 2),
            timestamp=timestamp.isoformat(),
            user_profile=user
        ))

    return feedbacks

def calculate_demographic_bias(feedbacks: List[SyntheticFeedback]) -> BiasScore:
    """Calcola b1: demographic bias."""
    # Raggruppa per professione
    by_profession = defaultdict(list)
    for f in feedbacks:
        by_profession[f.user_profile.profession].append(f.position)

    # Calcola concentrazione posizioni per gruppo
    position_concentrations = []
    for profession, positions in by_profession.items():
        if len(positions) > 1:
            # Quanto è concentrata la distribuzione delle posizioni
            position_counts = defaultdict(int)
            for p in positions:
                position_counts[p] += 1
            max_count = max(position_counts.values())
            concentration = max_count / len(positions)
            position_concentrations.append(concentration)

    if not position_concentrations:
        score = 0.0
    else:
        # Bias alto se gruppi diversi hanno posizioni molto diverse
        score = 1 - (sum(position_concentrations) / len(position_concentrations))

    return BiasScore(
        dimension="demographic",
        score=round(score, 3),
        details=f"Analyzed {len(by_profession)} professional groups",
        contributing_factors=[f"{p}: {len(pos)} feedbacks" for p, pos in by_profession.items()]
    )

def calculate_professional_clustering(feedbacks: List[SyntheticFeedback]) -> BiasScore:
    """Calcola b2: professional clustering bias."""
    # Conta feedback per professione
    profession_counts = defaultdict(int)
    for f in feedbacks:
        profession_counts[f.user_profile.profession] += 1

    total = len(feedbacks)
    if total == 0:
        return BiasScore("professional", 0.0, "No feedbacks", [])

    # Calcola concentrazione (Herfindahl index)
    hhi = sum((count / total) ** 2 for count in profession_counts.values())
    # Normalizza: HHI va da 1/n (uniforme) a 1 (monopolio)
    n_professions = len(profession_counts)
    if n_professions <= 1:
        score = 1.0
    else:
        min_hhi = 1 / n_professions
        score = (hhi - min_hhi) / (1 - min_hhi)

    return BiasScore(
        dimension="professional",
        score=round(score, 3),
        details=f"HHI concentration index: {hhi:.3f}",
        contributing_factors=[f"{p}: {c} ({c/total*100:.1f}%)" for p, c in profession_counts.items()]
    )

def calculate_temporal_bias(feedbacks: List[SyntheticFeedback]) -> BiasScore:
    """Calcola b3: temporal drift bias."""
    # Ordina per timestamp
    sorted_fb = sorted(feedbacks, key=lambda f: f.timestamp)

    if len(sorted_fb) < 10:
        return BiasScore("temporal", 0.0, "Not enough data", [])

    # Dividi in prima e seconda metà
    mid = len(sorted_fb) // 2
    first_half = sorted_fb[:mid]
    second_half = sorted_fb[mid:]

    # Calcola distribuzione posizioni per metà
    def position_distribution(fbs):
        counts = defaultdict(int)
        for f in fbs:
            counts[f.position] += 1
        total = len(fbs)
        return {p: c/total for p, c in counts.items()}

    dist1 = position_distribution(first_half)
    dist2 = position_distribution(second_half)

    # Calcola divergenza
    all_positions = set(dist1.keys()) | set(dist2.keys())
    divergence = sum(abs(dist1.get(p, 0) - dist2.get(p, 0)) for p in all_positions) / 2

    return BiasScore(
        dimension="temporal",
        score=round(divergence, 3),
        details=f"Distribution shift between first and second half",
        contributing_factors=[
            f"First half: {dict(dist1)}",
            f"Second half: {dict(dist2)}"
        ]
    )

def calculate_geographic_bias(feedbacks: List[SyntheticFeedback]) -> BiasScore:
    """Calcola b4: geographic concentration bias."""
    region_counts = defaultdict(int)
    for f in feedbacks:
        region_counts[f.user_profile.region] += 1

    total = len(feedbacks)
    if total == 0:
        return BiasScore("geographic", 0.0, "No feedbacks", [])

    # Usa stesso metodo di professional clustering
    hhi = sum((count / total) ** 2 for count in region_counts.values())
    n_regions = len(region_counts)
    if n_regions <= 1:
        score = 1.0
    else:
        min_hhi = 1 / n_regions
        score = (hhi - min_hhi) / (1 - min_hhi)

    return BiasScore(
        dimension="geographic",
        score=round(score, 3),
        details=f"Geographic HHI: {hhi:.3f}",
        contributing_factors=[f"{r}: {c} ({c/total*100:.1f}%)" for r, c in region_counts.items()]
    )

def calculate_confirmation_bias(feedbacks: List[SyntheticFeedback]) -> BiasScore:
    """Calcola b5: confirmation bias."""
    # Traccia posizioni precedenti per utente
    user_history = defaultdict(list)
    confirmations = 0
    total_with_history = 0

    sorted_fb = sorted(feedbacks, key=lambda f: f.timestamp)

    for f in sorted_fb:
        if user_history[f.user_id]:
            total_with_history += 1
            # L'utente conferma la sua posizione precedente?
            if f.position == user_history[f.user_id][-1]:
                confirmations += 1
        user_history[f.user_id].append(f.position)

    if total_with_history == 0:
        score = 0.0
    else:
        # Confirmation rate - expected random rate (1/3)
        confirmation_rate = confirmations / total_with_history
        score = max(0, confirmation_rate - 1/3) / (1 - 1/3)

    return BiasScore(
        dimension="confirmation",
        score=round(score, 3),
        details=f"Confirmation rate: {confirmations}/{total_with_history}",
        contributing_factors=[f"Users with multiple feedbacks: {len([u for u, h in user_history.items() if len(h) > 1])}"]
    )

def calculate_anchoring_bias(feedbacks: List[SyntheticFeedback]) -> BiasScore:
    """Calcola b6: anchoring bias."""
    sorted_fb = sorted(feedbacks, key=lambda f: f.timestamp)

    if len(sorted_fb) < 5:
        return BiasScore("anchoring", 0.0, "Not enough data", [])

    # Le prime 5 risposte sono l'ancora
    anchor_positions = [f.position for f in sorted_fb[:5]]
    anchor_majority = max(set(anchor_positions), key=anchor_positions.count)

    # Quanto le risposte successive seguono l'ancora?
    subsequent = sorted_fb[5:]
    if not subsequent:
        return BiasScore("anchoring", 0.0, "No subsequent feedbacks", [])

    followers = sum(1 for f in subsequent if f.position == anchor_majority)
    follow_rate = followers / len(subsequent)

    # Score: quanto la follow rate supera il random (1/3)
    score = max(0, follow_rate - 1/3) / (1 - 1/3)

    return BiasScore(
        dimension="anchoring",
        score=round(score, 3),
        details=f"Anchor position: {anchor_majority}, follow rate: {follow_rate:.2%}",
        contributing_factors=[f"Followers: {followers}/{len(subsequent)}"]
    )

def calculate_total_bias(dimension_scores: List[BiasScore]) -> tuple:
    """
    Calcola il bias totale usando la formula del paper:
    B_total = sqrt(sum(b_i^2))
    """
    scores = [d.score for d in dimension_scores]
    total = math.sqrt(sum(s**2 for s in scores))

    # Classifica livello
    if total <= 0.5:
        level = "low"
    elif total <= 1.0:
        level = "medium"
    else:
        level = "high"

    return round(total, 3), level

def generate_mitigation_recommendations(report: BiasReport) -> List[str]:
    """Genera raccomandazioni per mitigare i bias rilevati."""
    recommendations = []

    if report.bias_scores["professional"] > 0.3:
        recommendations.append("Diversificare il panel di valutatori per professione")

    if report.bias_scores["geographic"] > 0.3:
        recommendations.append("Includere valutatori da più regioni geografiche")

    if report.bias_scores["temporal"] > 0.2:
        recommendations.append("Investigare cause del drift temporale nelle opinioni")

    if report.bias_scores["confirmation"] > 0.2:
        recommendations.append("Implementare randomizzazione nell'ordine di presentazione")

    if report.bias_scores["anchoring"] > 0.2:
        recommendations.append("Nascondere le risposte iniziali ai valutatori successivi")

    if not recommendations:
        recommendations.append("Nessuna mitigazione urgente necessaria - bias entro limiti accettabili")

    return recommendations

def generate_markdown_report(report: BiasReport, output_path: Path):
    """Genera il report Markdown."""
    md = f"""# Bias Detection Report

**Generated**: {report.timestamp}
**Task ID**: {report.task_id}
**Feedbacks Analyzed**: {report.num_feedbacks_analyzed}

---

## Executive Summary

**Total Bias Score**: {report.total_bias_score} ({report.bias_level.upper()})

**Formula Used**: `{report.formula_used}`

---

## 6-Dimensional Bias Analysis

| Dimension | Score | Level | Description |
|-----------|-------|-------|-------------|
"""

    for dim in report.dimension_details:
        level = "Low" if dim.score < 0.3 else "Medium" if dim.score < 0.6 else "High"
        md += f"| {dim.dimension.capitalize()} | {dim.score:.3f} | {level} | {dim.details} |\n"

    md += f"""
---

## Detailed Analysis

"""

    for dim in report.dimension_details:
        md += f"""### {dim.dimension.capitalize()} Bias (b{report.dimension_details.index(dim)+1})

**Score**: {dim.score:.3f}

**Details**: {dim.details}

**Contributing Factors**:
"""
        for factor in dim.contributing_factors:
            md += f"- {factor}\n"
        md += "\n"

    md += f"""---

## Mitigation Recommendations

"""
    for i, rec in enumerate(report.mitigation_recommendations, 1):
        md += f"{i}. {rec}\n"

    md += f"""
---

## Formula Reference

The total bias score is calculated using the Euclidean norm of all dimension scores:

$$
B_{{total}} = \\sqrt{{\\sum_{{i=1}}^{{6}} b_i^2}} = \\sqrt{{{' + '.join([f'{d.score:.3f}^2' for d in report.dimension_details])}}} = {report.total_bias_score}
$$

**Thresholds**:
- Low: B_total ≤ 0.5
- Medium: 0.5 < B_total ≤ 1.0
- High: B_total > 1.0

---

## References

- Allega, D., & Puzio, G. (2025c). RLCF Paper, Section 3.3: Extended Bias Detection Framework
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

def main():
    """Esegue la demo del bias detection."""
    print("Running Bias Detection Demo...")

    # Genera dati sintetici
    print("Generating synthetic user profiles...")
    users = generate_user_profiles(30)

    print("Generating synthetic feedbacks with intentional biases...")
    feedbacks = generate_synthetic_feedbacks(users, num_feedbacks=50)

    # Calcola bias per dimensione
    print("Calculating 6-dimensional bias scores...")
    dimension_scores = [
        calculate_demographic_bias(feedbacks),
        calculate_professional_clustering(feedbacks),
        calculate_temporal_bias(feedbacks),
        calculate_geographic_bias(feedbacks),
        calculate_confirmation_bias(feedbacks),
        calculate_anchoring_bias(feedbacks)
    ]

    # Calcola totale
    total_score, level = calculate_total_bias(dimension_scores)

    # Crea report
    report = BiasReport(
        task_id="DEMO_TASK",
        timestamp=datetime.now().isoformat(),
        num_feedbacks_analyzed=len(feedbacks),
        bias_scores={d.dimension: d.score for d in dimension_scores},
        total_bias_score=total_score,
        bias_level=level,
        formula_used="B_total = sqrt(sum(b_i^2))",
        dimension_details=dimension_scores,
        mitigation_recommendations=[]
    )
    report.mitigation_recommendations = generate_mitigation_recommendations(report)

    # Output directory
    output_dir = Path(__file__).parent

    # Salva feedbacks sintetici
    feedbacks_data = [asdict(f) for f in feedbacks]
    with open(output_dir / "synthetic_feedbacks.json", 'w') as f:
        json.dump(feedbacks_data, f, indent=2, default=str)

    # Salva report JSON
    with open(output_dir / "bias_report.json", 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)

    # Genera report MD
    generate_markdown_report(report, output_dir / "bias_detection_report.md")

    print(f"\nResults:")
    print(f"  Total Bias Score: {total_score} ({level.upper()})")
    for d in dimension_scores:
        print(f"  - {d.dimension}: {d.score:.3f}")
    print(f"\nFiles saved in: {output_dir}")

if __name__ == "__main__":
    main()
