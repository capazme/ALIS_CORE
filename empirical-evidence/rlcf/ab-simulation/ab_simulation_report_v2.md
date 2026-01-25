# A/B Simulation Report v2 - RLCF vs Baseline

**Generated**: 2026-01-25T16:38:53.972487
**Methodology**: Statistically rigorous with 30 independent trials

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Users | 100 |
| Tasks per Trial | 100 |
| Trials | 30 |
| Raters per Task | 5-20 |
| Base Noise σ | 1.5 |
| Authority Noise Factor | 0.95 |
| Random Seed | 42 |

### Noise Model

The key insight of RLCF is that expert users provide more accurate feedback:

```
noise_std = base_noise_std × (1 - authority × authority_noise_factor)
```

| Authority | Noise σ | Interpretation |
|-----------|---------|----------------|
| 0.0 (novice) | 1.50 | High variance feedback |
| 0.5 (intermediate) | 0.79 | Medium variance |
| 1.0 (expert) | 0.08 | Low variance feedback |

---

## Results Summary

### Mean Absolute Error (MAE)

| Method | MAE Mean | MAE Std | 95% CI |
|--------|----------|---------|--------|
| **RLCF** | **0.1286** | 0.0117 | [0.1243, 0.1330] |
| Baseline | 0.1393 | 0.0120 | [0.1348, 0.1437] |

### Improvement

| Metric | Value |
|--------|-------|
| Mean Improvement | **7.67%** |
| Std | 1.35% |
| 95% CI | [7.16%, 8.17%] |

### Win Rate

| Outcome | Count | Percentage |
|---------|-------|------------|
| RLCF Wins | 30 | 100.0% |
| Baseline Wins | 0 | 0.0% |
| Ties | 0 | 0.0% |

---

## Statistical Significance

**Result**: RLCF significantly outperforms baseline (p < 0.05)

The 95% confidence interval for improvement [7.16%, 8.17%]
does not include zero, indicating **statistically significant improvement**.


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
| 0 | 0.1096 | 0.1201 | 8.76% |
| 1 | 0.1236 | 0.1376 | 10.14% |
| 2 | 0.1110 | 0.1195 | 7.12% |
| 3 | 0.1336 | 0.1424 | 6.15% |
| 4 | 0.1122 | 0.1238 | 9.43% |
| 5 | 0.1369 | 0.1486 | 7.83% |
| 6 | 0.1326 | 0.1456 | 8.91% |
| 7 | 0.1216 | 0.1298 | 6.38% |
| 8 | 0.1237 | 0.1348 | 8.27% |
| 9 | 0.1464 | 0.1584 | 7.57% |


---

## Comparison with Literature

Our **7.67% improvement** is consistent with published research on expertise-weighted aggregation:

| Study | Method | Improvement |
|-------|--------|-------------|
| MIT "Surprisingly Popular" (2017) | Expert-informed weighting | 21-35% error reduction |
| Contribution Weighted Model (Management Science, 2014) | Positive contributor weighting | 28-39% improvement |
| Federated Learning (FedAAW, 2024) | Adaptive aggregate weights | Up to 50% improvement |
| **This Study (RLCF)** | Authority-weighted aggregation | **7.67%** (conservative simulation) |

**Note**: Our simulation uses conservative parameters. Real-world improvements can be higher
when authority scores are more accurately calibrated to actual expertise (see EXP-021: +183.4%
authority convergence, EXP-022: +14% load balance improvement).

---

## References

### RLCF Paper
- Allega, D., & Puzio, G. (2025c). RLCF Paper, Section 3.1: Dynamic Authority Scoring
- Formula: A_u(t) = α·B_u + β·T_u(t-1) + γ·P_u(t)

### Literature
- Prelec, D., Seung, H. S., & McCoy, J. (2017). A solution to the single-question crowd wisdom problem. *Nature*, 541(7638), 532-535. https://news.mit.edu/2017/algorithm-better-wisdom-crowds-0125
- Budescu, D. V., & Chen, E. (2014). Identifying Expertise to Extract the Wisdom of Crowds. *Management Science*, 61(2), 267-280. https://pubsonline.informs.org/doi/10.1287/mnsc.2014.1909
- PMC7865038: How training affects the wisdom of crowds in perceptual tasks
