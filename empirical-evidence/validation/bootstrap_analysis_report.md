# Statistical Analysis Report - Bootstrap & Effect Sizes

**Generated**: 2026-01-25T22:58:20.079194
**Bootstrap Resamples**: 10,000
**Confidence Level**: 95.0%
**Random Seed**: 42

---

## Executive Summary

This report provides rigorous statistical analysis of empirical evidence for MERL-T and RLCF papers, including:
- Bootstrap confidence intervals (BCa method approximation)
- Cohen's d effect sizes with interpretation
- Statistical power estimates

---

## 1. MERL-T Pipeline Analysis

### 1.1 Sample Characteristics

| Metric | Value |
|--------|-------|
| Total Traces | 9 |
| Success Rate | 88.9% |

### 1.2 Latency (ms) - Bootstrap Analysis

| Statistic | Value |
|-----------|-------|
| Original Mean | 57,781.58 |
| Bootstrap Mean | 57,770.05 |
| Bootstrap Std | 2,005.58 |
| **95% CI** | **[53,782.29, 61,564.80]** |

### 1.3 Confidence Score - Bootstrap Analysis

| Statistic | Value |
|-----------|-------|
| Original Mean | 0.7883 |
| Bootstrap Mean | 0.7884 |
| Bootstrap Std | 0.0938 |
| **95% CI** | **[0.5839, 0.9089]** |

### 1.4 Sources per Query - Bootstrap Analysis

| Statistic | Value |
|-----------|-------|
| Original Mean | 16.67 |
| **95% CI** | **[11.78, 20.56]** |

### 1.5 Expert-Level Analysis

| Expert | Confidence Mean | Confidence 95% CI | Latency Mean (ms) | Latency 95% CI |
|--------|-----------------|-------------------|-------------------|----------------|
| Literal | 0.822 | [0.611, 0.944] | 8,682 | [7,155, 9,922] |
| Systemic | 0.811 | [0.600, 0.933] | 11,864 | [9,922, 13,535] |
| Principles | 0.700 | [0.400, 0.900] | 10,228 | [7,673, 12,115] |
| Precedent | 0.789 | [0.589, 0.900] | 11,133 | [10,166, 12,192] |

### 1.6 Limitations Note

With N=9 samples, bootstrap CIs are wide. For robust claims, N≥30 recommended.

---

## 2. RLCF A/B Simulation Analysis

### 2.1 Sample Characteristics

| Parameter | Value |
|-----------|-------|
| Number of Trials | 30 |
| Users per Trial | 100 |
| Tasks per Trial | 100 |

### 2.2 MAE Comparison - Bootstrap Analysis

| Method | Mean MAE | 95% CI | Bootstrap Std |
|--------|----------|--------|---------------|
| **RLCF** | 0.1286 | [0.1245, 0.1326] | 0.0021 |
| **Baseline** | 0.1393 | [0.1350, 0.1434] | 0.0021 |

### 2.3 Improvement Analysis

| Metric | Value |
|--------|-------|
| Mean Improvement | **7.67%** |
| 95% CI | **[7.17%, 8.12%]** |
| Bootstrap Std | 0.24% |

**Note**: CI does not include zero, indicating statistically significant improvement.

### 2.4 Effect Size (Cohen's d)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Cohen's d | **0.900** | **LARGE** |
| 95% CI | [0.399, 1.498] | |

**Interpretation Guide** (Cohen, 1988):
- |d| < 0.2: negligible
- 0.2 ≤ |d| < 0.5: small
- 0.5 ≤ |d| < 0.8: medium
- |d| ≥ 0.8: large

### 2.5 Statistical Power

| Metric | Value |
|--------|-------|
| Estimated Power | **93.6%** |
| Target (convention) | 80% |
| Status | ✅ Adequate |

### 2.6 Win Rate Analysis

| Outcome | Count | Rate |
|---------|-------|------|
| RLCF Wins | 30 | 100.0% |
| Baseline Wins | 0 | 0.0% |
| Win Rate 95% CI | | [100.0%, 100.0%] |

### 2.7 Non-Parametric Tests (Distribution-Free)

These additional tests make NO distributional assumptions, strengthening the evidence.

#### Permutation Test

| Metric | Value |
|--------|-------|
| Observed Difference | 0.0106 |
| p-value (two-tailed) | **0.0006** |
| N Permutations | 10,000 |
| Interpretation | **Significant (p < 0.05)** |

#### Cliff's Delta (Non-Parametric Effect Size)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Cliff's Delta | **0.487** | **LARGE** |
| Dominance (Baseline > RLCF) | 669 | |
| Dominance (RLCF > Baseline) | 231 | |

**Interpretation Guide** (Romano et al., 2006):
- |δ| < 0.147: negligible
- 0.147 ≤ |δ| < 0.33: small
- 0.33 ≤ |δ| < 0.474: medium
- |δ| ≥ 0.474: large

#### Wilcoxon Signed-Rank Test (Paired)

| Metric | Value |
|--------|-------|
| W Statistic | 0.0 |
| Z Score | -4.782 |
| p-value (approx.) | **0.0000** |
| N Pairs | 30 |
| Interpretation | **Significant (p < 0.05)** |

---

## 3. EXP-021 RLCF Learning Analysis

### 3.1 Confidence Improvement

| Phase | Value | Change |
|-------|-------|--------|
| Baseline | 86.83% | - |
| Post-Training | 90.08% | +3.25pp |
| Relative Improvement | | **+3.74%** |

**Effect Size**: Cohen's d = 1.495 (large)

### 3.2 Source Grounding Improvement

| Phase | Value | Change |
|-------|-------|--------|
| Baseline | 52.62% | - |
| Post-Training | 61.55% | +8.93pp |
| Relative Improvement | | **+16.97%** |

**Effect Size**: Cohen's d = 0.379 (small)

---

## 4. Summary of Effect Sizes

| Comparison | Cohen's d | Interpretation | 95% CI |
|------------|-----------|----------------|--------|
| A/B: RLCF vs Baseline (MAE) | 0.900 | large | [0.399, 1.498] |
| EXP-021: Confidence Improvement | 1.495 | large | [0.762, 2.857] |
| EXP-021: Source Grounding | 0.379 | small | [-0.515, 1.418] |

---

## 5. Methodological Notes

### 5.1 Bootstrap Method

- **Resampling**: 10,000 bootstrap samples with replacement
- **CI Method**: Percentile method (2.5th and 97.5th percentiles)
- **Reproducibility**: Random seed = 42

### 5.2 Effect Size Calculation

Cohen's d calculated as:
```
d = (M1 - M2) / pooled_std
pooled_std = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1 + n2 - 2))
```

### 5.3 Statistical Power

Power estimated using normal approximation:
```
power ≈ Φ(|d|*sqrt(n/2) - z_α/2)
```

### 5.4 Limitations

1. **Small sample size** for pipeline traces (N=9) limits precision
2. **EXP-021 effect sizes** estimated from simulated individual queries
3. **A/B simulation** assumes authority-accuracy correlation (documented circular reasoning)

---

## 6. Recommendations for Publication

1. **Report effect sizes** alongside p-values (done in this analysis)
2. **Use confidence intervals** instead of point estimates where possible
3. **Acknowledge power limitations** for small-sample analyses
4. **Increase sample size** if feasible (target: N≥30 for pipeline traces)

---

*Report generated by bootstrap_analysis.py*
*All calculations are reproducible with seed=42*
