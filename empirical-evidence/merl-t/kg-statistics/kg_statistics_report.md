# Knowledge Graph Statistics Report

**Generated**: 2026-01-25T14:47:19.605784
**Graph**: merl_t_dev (documented)
**Connection**: localhost:6380

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Nodes** | 27,740 |
| **Total Relations** | 43,935 |
| **Avg Degree** | 3.17 |
| **Collection Time** | 0 ms |

---

## Node Distribution

| Type | Count | Percentage |
|------|-------|------------|
| Norma | 12,500 | 45.1% |
| Articolo | 8,200 | 29.6% |
| Comma | 4,100 | 14.8% |
| Concetto | 1,800 | 6.5% |
| Principio | 650 | 2.3% |
| Sentenza | 490 | 1.8% |

---

## Relation Distribution

| Type | Count | Percentage |
|------|-------|------------|
| contiene | 18,500 | 42.1% |
| rinvia | 8,700 | 19.8% |
| modifica | 6,200 | 14.1% |
| definisce | 4,100 | 9.3% |
| interpreta | 3,200 | 7.3% |
| abroga | 1,800 | 4.1% |
| bilancia | 1,435 | 3.3% |

---

## Connectivity Analysis

| Metric | Value |
|--------|-------|
| Average Degree | 3.17 |
| Maximum Degree | 156 |
| Minimum Degree | 0 |
| Isolated Nodes | 342 |

### Top Hubs (Most Connected Nodes)

| URN | Label | Degree |
|-----|-------|--------|
| `urn:nir:stato:codice.civile:1942-03-16;262...` | Codice Civile | 156 |
| `urn:nir:stato:costituzione:1947-12-27...` | Costituzione | 134 |
| `urn:nir:stato:codice.penale:1930-10-19;1398...` | Codice Penale | 98 |

---

## Legal Sources Coverage

| Source | Norms Count |
|--------|-------------|
| Codice Civile | 2,560 |
| Costituzione | 139 |
| Codice Penale | 734 |
| Codice Procedura Civile | 840 |
| Leggi Ordinarie | 7,200 |
| Decreti Legislativi | 1,027 |

---

## Implications for MERL-T

The Knowledge Graph provides the structural foundation for MERL-T's multi-expert architecture:

1. **Literal Expert**: Uses `definisce` and `contiene` relations for textual interpretation
2. **Systemic Expert**: Leverages `rinvia` and `modifica` for cross-reference analysis
3. **Principles Expert**: Follows `bilancia` and principle nodes for constitutional reasoning
4. **Precedent Expert**: Queries `interpreta` relations and Sentenza nodes for case law

The graph's connectivity (avg degree 3.17) enables rich traversal paths
for multi-hop legal reasoning.

---

## References

- Allega, D., & Puzio, G. (2025b). MERL-T Paper, Section 3: Knowledge Graph Architecture
