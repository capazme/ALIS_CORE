# Bias Detection Report

**Generated**: 2026-01-25T14:42:45.201341
**Task ID**: DEMO_TASK
**Feedbacks Analyzed**: 50

---

## Executive Summary

**Total Bias Score**: 0.559 (MEDIUM)

**Formula Used**: `B_total = sqrt(sum(b_i^2))`

---

## 6-Dimensional Bias Analysis

| Dimension | Score | Level | Description |
|-----------|-------|-------|-------------|
| Demographic | 0.489 | Medium | Analyzed 5 professional groups |
| Professional | 0.220 | Low | HHI concentration index: 0.376 |
| Temporal | 0.080 | Low | Distribution shift between first and second half |
| Geographic | 0.133 | Low | Geographic HHI: 0.350 |
| Confirmation | 0.000 | Low | Confirmation rate: 7/25 |
| Anchoring | 0.033 | Low | Anchor position: correct, follow rate: 35.56% |

---

## Detailed Analysis

### Demographic Bias (b1)

**Score**: 0.489

**Details**: Analyzed 5 professional groups

**Contributing Factors**:
- praticante: 5 feedbacks
- accademico: 1 feedbacks
- magistrato: 13 feedbacks
- avvocato: 27 feedbacks
- notaio: 4 feedbacks

### Professional Bias (b2)

**Score**: 0.220

**Details**: HHI concentration index: 0.376

**Contributing Factors**:
- praticante: 5 (10.0%)
- accademico: 1 (2.0%)
- magistrato: 13 (26.0%)
- avvocato: 27 (54.0%)
- notaio: 4 (8.0%)

### Temporal Bias (b3)

**Score**: 0.080

**Details**: Distribution shift between first and second half

**Contributing Factors**:
- First half: {'correct': 0.36, 'incorrect': 0.24, 'partially_correct': 0.4}
- Second half: {'partially_correct': 0.32, 'correct': 0.44, 'incorrect': 0.24}

### Geographic Bias (b4)

**Score**: 0.133

**Details**: Geographic HHI: 0.350

**Contributing Factors**:
- lazio: 9 (18.0%)
- veneto: 9 (18.0%)
- campania: 6 (12.0%)
- lombardia: 26 (52.0%)

### Confirmation Bias (b5)

**Score**: 0.000

**Details**: Confirmation rate: 7/25

**Contributing Factors**:
- Users with multiple feedbacks: 16

### Anchoring Bias (b6)

**Score**: 0.033

**Details**: Anchor position: correct, follow rate: 35.56%

**Contributing Factors**:
- Followers: 16/45

---

## Mitigation Recommendations

1. Nessuna mitigazione urgente necessaria - bias entro limiti accettabili

---

## Formula Reference

The total bias score is calculated using the Euclidean norm of all dimension scores:

$$
B_{total} = \sqrt{\sum_{i=1}^{6} b_i^2} = \sqrt{0.489^2 + 0.220^2 + 0.080^2 + 0.133^2 + 0.000^2 + 0.033^2} = 0.559
$$

**Thresholds**:
- Low: B_total ≤ 0.5
- Medium: 0.5 < B_total ≤ 1.0
- High: B_total > 1.0

---

## References

- Allega, D., & Puzio, G. (2025c). RLCF Paper, Section 3.3: Extended Bias Detection Framework
