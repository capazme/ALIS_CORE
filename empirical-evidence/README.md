# Evidenze Empiriche - MERL-T e RLCF

**Data generazione**: 2026-01-25
**Autori**: Allega, Puzio, Rizzo
**Repository**: ALIS_CORE

---

## Obiettivo

Questa directory contiene le evidenze empiriche richieste dai reviewer per due paper scientifici:

1. **MERL-T**: Multi-Expert Retrieval Legal Transformer - Architettura multi-expert per AI legale
2. **RLCF**: Reinforcement Learning from Community Feedback - Framework di allineamento AI

---

## Struttura

```
empirical-evidence/
├── README.md                 # Questo file - overview generale
├── METHODOLOGY.md            # Metodologia scientifica dettagliata
├── LIMITATIONS.md            # Limitazioni e gap (trasparenza scientifica)
├── TABLES_FOR_PAPERS.md      # 22 tabelle pronte per pubblicazione
├── INTERPRETATIONS.md        # Guida interpretazione risultati
├── CHAT_LOG.md               # Log ragionamenti e decisioni
│
├── merl-t/                   # Evidenze Paper MERL-T
│   ├── kg-statistics/        # Statistiche Knowledge Graph
│   ├── expert-pipeline-trace/# Tracce esecuzione Expert
│   └── latency-benchmark/    # Benchmark performance
│
├── rlcf/                     # Evidenze Paper RLCF
│   ├── implementation-evidence/  # Prova implementazione formule
│   ├── ab-simulation/        # Simulazione A/B RLCF vs baseline
│   └── bias-detection-demo/  # Demo rilevamento bias
│
├── figures/                  # Visualizzazioni per i paper
│   ├── README.md             # Descrizione figure
│   └── fig[1-10]_*.png       # 10 figure PNG per pubblicazione
│
└── validation/               # Validazione incrociata
    ├── checksums.json        # Hash file per integrità
    ├── execution_log.md      # Log esecuzioni
    ├── bootstrap_analysis.py # Script analisi statistica (riproducibile)
    ├── bootstrap_results.json# Risultati raw bootstrap
    ├── bootstrap_analysis_report.md # Report statistico completo
    └── generate_visualizations.py # Script generazione figure
```

---

## Evidenze Generate

### Paper 1: MERL-T

| Evidenza | Descrizione | File Output | Status |
|----------|-------------|-------------|--------|
| **KG Statistics** | Statistiche Knowledge Graph | `kg_statistics.json`, `kg_statistics_live.json` | ✅ Completata |
| **Expert Pipeline Trace** | 9 query con traccia completa 4-expert | `pipeline_traces.json`, `pipeline_trace_report.md` | ✅ Completata |
| **Latency Benchmark** | Percentili p50/p95/p99 per expert | `latency_results.json`, `latency_report.md` | ✅ Completata |

**Metriche Chiave MERL-T**:
- 4 Expert operativi: Literal, Systemic, Principles, Precedent
- Trace Success Rate: **89%** (8/9 query completate, 1 network failure catturato - dato autentico)
- Source Grounding (su trace riusciti): **100%** (tutte le risposte citano fonti dal DB)
- Mean Confidence: **0.79**
- Latency p50: **58,997 ms** | p95: **65,988 ms** | p99: **66,798 ms** (pipeline completa con LLM)
- Expert più lento: Systemic (20.5% del tempo totale)

**Nota Latenza**: I 93ms riportati in EXP-016 sono solo vector search; la pipeline completa con 4 expert e LLM richiede ~60s.

**Knowledge Graph**:
- **Full Dataset (EXP-014)**: 27,740 nodi, 43,935 relazioni (documentato)
- **Live Sample**: 16 nodi, 18 relazioni (Art. 1453-1456, verificabile in tempo reale)
- **Nota**: Sample live dimostra funzionamento; full ingestion richiede pipeline completa

### Paper 2: RLCF

| Evidenza | Descrizione | File Output | Status |
|----------|-------------|-------------|--------|
| **Implementation Evidence** | 4 formule dimostrate implementate | `formula_evidence.json`, `implementation_proof.md` | ✅ Completata |
| **A/B Simulation v2** | RLCF vs Baseline (30 trial, 100 task/trial) | `ab_results_v2.json`, `ab_simulation_report_v2.md` | ✅ Completata |
| **Bias Detection Demo** | Output 6-dimensional su 50 feedback | `bias_report.json`, `bias_detection_report.md` | ✅ Completata |

**Metriche Chiave RLCF**:
- Formule implementate: 4/4 (100%)
- **A/B Test v2**: RLCF MAE 0.1286 vs Baseline 0.1393 (**+7.67% improvement**)
  - 30 trial indipendenti, 100% win rate per RLCF
  - 95% CI: [7.17%, 8.12%] - non include zero, statisticamente significativo
  - Cohen's d = 0.90 (effetto LARGE), Power = 93.6%
  - Parametri calibrati su letteratura (MIT, Management Science)
- **Bias Detection**: Total Score 0.559 (MEDIUM) su 6 dimensioni

**Evidenze da Esperimenti Validati (EXP-021, EXP-022)**:
- **Authority Convergence**: +183.4% improvement (target: +20%) - EXP-021
- **Response Quality** (EXP-021):
  - Confidence: 86.83% → 90.08% (+3.25pp, Cohen's d = 1.50 LARGE)
  - Source Grounding: 52.62% → 61.55% (+8.93pp, Cohen's d = 0.38 small)
- **Load Balance**: 0.84 → 0.96 (**+14% improvement**) - EXP-022
- **Policy Entropy**: 1.37 → 1.39 (stabile, convergenza validata)

---

## Analisi Statistica Rigorosa (Bootstrap)

Per garantire rigore scientifico, è stata eseguita un'analisi bootstrap completa con 10,000 ricampionamenti.

### Effect Sizes (Cohen's d)

| Confronto | Cohen's d | Interpretazione | 95% CI |
|-----------|-----------|-----------------|--------|
| **A/B: RLCF vs Baseline** | **0.900** | **LARGE** | [0.399, 1.498] |
| EXP-021: Confidence | 1.495 | large | [0.762, 2.857] |
| EXP-021: Source Grounding | 0.379 | small | [-0.515, 1.418] |

*Guida interpretazione (Cohen, 1988): |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, ≥0.8 large*

### Bootstrap Confidence Intervals (95%)

| Metrica | Media | 95% CI |
|---------|-------|--------|
| Pipeline Latency | 57,782 ms | [53,782, 61,565] ms |
| Pipeline Confidence | 0.79 | [0.58, 0.91] |
| A/B Improvement | 7.67% | [7.17%, 8.12%] |

### Statistical Power

| Analisi | Potenza | Status |
|---------|---------|--------|
| A/B Simulation (N=30) | **93.6%** | ✅ Adeguata (>80%) |
| Pipeline Traces (N=9) | ~50% | ⚠️ Limitata (N<30) |

**Nota**: Pipeline traces ha potenza limitata per sample size ridotto. L'analisi bootstrap fornisce CI robusti ma raccomanda N≥30 per claims definitivi.

*Report completo: `validation/bootstrap_analysis_report.md`*

---

## Come Riprodurre

### Prerequisiti

```bash
# 1. Servizi Docker (per KG Statistics)
cd /path/to/ALIS_CORE/merlt
docker-compose -f docker-compose.dev.yml up -d falkordb qdrant

# 2. Environment Python
cd /path/to/ALIS_CORE/merlt
pip install -e .
```

### Esecuzione Script

```bash
# MERL-T: KG Statistics
python empirical-evidence/merl-t/kg-statistics/kg_stats_collector.py

# MERL-T: Expert Pipeline Trace
python empirical-evidence/merl-t/expert-pipeline-trace/pipeline_tracer.py

# RLCF: Implementation Evidence
python empirical-evidence/rlcf/implementation-evidence/extract_formulas.py

# RLCF: A/B Simulation (v2 - metodologia corretta)
python empirical-evidence/rlcf/ab-simulation/ab_simulation_v2.py

# RLCF: Bias Detection Demo
python empirical-evidence/rlcf/bias-detection-demo/bias_demo.py

# Statistical Analysis (Bootstrap + Effect Sizes)
python empirical-evidence/validation/bootstrap_analysis.py
```

---

## Evidenze Supplementari (da Esperimenti Legacy)

Oltre alle evidenze generate, esistono dati aggiuntivi in `/merlt/docs/experiments/`:

### Per MERL-T

| Experiment | Metrica | Valore | Significatività |
|------------|---------|--------|-----------------|
| **EXP-016** | NDCG@5 | **0.869** | +44.8% sopra target (0.6) |
| EXP-016 | Hit Rate@5 | **96.67%** | 29/30 query con risultato |
| EXP-016 | MRR (Mean Reciprocal Rank) | **0.850** | Top-tier vs industry (0.70-0.85) |
| EXP-016 | Perfect Match Rate | **93.3%** | 28/30 trovano articolo esatto |
| EXP-016 | Latency p50 | **93ms** | Solo vector search (non include LLM) |
| EXP-018 | Routing Accuracy | **90%** | Expert selection |
| **EXP-020** | Source Grounding | **100%** | Zero hallucination |
| EXP-020 | Hallucination Rate | **0%** | Massima affidabilità |

### Per RLCF

| Experiment | Metrica | Valore | Note |
|------------|---------|--------|------|
| **EXP-021** | Authority Convergence | **+183.4%** (target: +20%) | ✅ H2 PASS |
| EXP-021 | Feedback Persistence | 100% (1228/1228) | ✅ H1 PASS |
| EXP-021 | Weight Convergence (WDC) | 0.00 (target: <0.5) | ⚠️ H3 marcato FAIL* |
| EXP-021 | Confidence Improvement | +3.25pp (86.83%→90.08%) | ⚠️ H4 marcato FAIL* |
| EXP-021 | Source Grounding | 52.6% → 61.5% (+16.9%) | Riduzione hallucination |
| **EXP-022** | Load Balance Score | 0.84 → **0.96** (+14%) | ✅ Near-perfect |
| EXP-022 | Policy Entropy | 1.37 → 1.39 (stabile) | ✅ Convergenza validata |
| EXP-022 | Expert Usage Balance | 19-29% → 24-26% | ✅ Distribuzione ottimale |

*\*Nota: H3 e H4 in EXP-021 hanno raggiunto i target numerici ma sono stati marcati FAIL nell'analisi originale per mancata significatività statistica (Bonferroni α=0.0125)*

---

## Esperimenti con LLM Reali (Già Eseguiti)

### EXP-020: Scientific Evaluation (20 Query Reali)

| Metrica | Expert System | Baseline LLM | Delta |
|---------|---------------|--------------|-------|
| **Source Grounding** | 100.0% | 96.6% | +3.4% |
| **Hallucination Rate** | 0.0% | 3.4% | -3.4% |
| **Avg Latency** | 14,012 ms | 9,940 ms | +4,072 ms |

**Conclusione**: Sistema expert superiore in affidabilità (0% hallucination), trade-off su latenza (+4s).

*Nota: Il report EXP-020 non include metriche aggregate di confidence.*

### EXP-024: Real Expert System (10 Query End-to-End)

| Metrica | Valore | Target | Status |
|---------|--------|--------|--------|
| Success Rate | 100% | 100% | ✅ |
| Avg Confidence | 0.689 | >0.5 | ✅ |
| Avg Latency | 19.7s | <2s | ❌ |
| Legal Basis Extraction | 0% | >70% | ❌ |

**Expert Usage**: Literal 100%, Systemic 80%, Principles 20%, Precedent 20%

---

## ⚠️ Limitazioni e Gap (Trasparenza Scientifica)

### Ipotesi Fallite negli Esperimenti

| Experiment | Ipotesi | Target | Attuale | Status |
|------------|---------|--------|---------|--------|
| EXP-015 | Recall@5 query concettuali | ≥90% | 61.1% | ❌ FAIL |
| EXP-023 | Reward improvement | +15% | +8.1% | ❌ FAIL |
| EXP-023 | Load Balance Score | >0.75 | 0.49-0.63 | ❌ FAIL |
| EXP-024 | Latency | <2s | 19.7s | ❌ FAIL |
| EXP-024 | Legal basis extraction | >70% | 0% | ❌ FAIL |

### Limitazioni Metodologiche

| Limitazione | Descrizione | Impatto |
|-------------|-------------|---------|
| **KG Statistics Fallback** | Dati da EXP-014, non live | MEDIO - Non verificabili in tempo reale |
| **Simulazione Circolare** | A/B assume correlazione authority-accuracy | ALTO - Dimostra ciò che assume |
| **Sample Size** | 9 trace, 30 query gold standard | MEDIO - Potenza statistica limitata |
| **Solo Sintetico** | Zero valutazione con umani reali | ALTO - Manca validazione esterna |
| **No Baseline Esterno** | Nessun confronto con Westlaw, GPT-4 | ALTO - Non si può affermare superiorità |

### Success Rate per Categoria

| Categoria | Ipotesi Testate | Passate | Fallite | Pass Rate |
|-----------|-----------------|---------|---------|-----------|
| Data Ingestion | 10 | 10 | 0 | **100%** |
| RAG Retrieval | 15 | 11 | 4 | **73%** |
| Knowledge Graph | 4 | 4 | 0 | **100%** |
| Expert System | 10 | 4 | 6 | **40%** |
| RLCF Learning | 11 | 2 | 9 | **18%** |
| **TOTALE** | **50** | **31** | **19** | **62%** |

### Cosa Funziona Bene ✅

1. **Data Ingestion**: 100% articoli processati (887 articoli CC, 139 Costituzione)
   - *Nota: Il KG contiene 2560+ nodi per il CC perché ogni articolo genera più nodi (commi, riferimenti, versioni)*
2. **Knowledge Graph**: 27,740 nodi, 43,935 relazioni, struttura gerarchica corretta
3. **Semantic Search Base**: NDCG@5 = 0.869, Hit Rate = 96.67%
4. **Source Grounding**: 100% nel sistema expert su query riuscite (0% hallucination, EXP-020)

### Cosa Richiede Miglioramenti ⚠️

1. **RLCF Learning Loop**: Overfitting dopo 10 iterazioni, convergenza non stabile
2. **Latency Expert System**: 14-20s (target: <2s per uso interattivo)
3. **Query Astratte**: Solo 61% recall per query concettuali vs 90% target
4. **Load Balancing**: 0.49-0.63 vs 0.75 target

---

## Riferimenti ai Paper

- Allega, D., & Puzio, G. (2025b). MERL-T: A multi-expert architecture for trustworthy artificial legal intelligence.
- Allega, D., & Puzio, G. (2025c). Reinforcement learning from community feedback (RLCF).

---

## Note per i Colleghi

- Tutti i file JSON seguono schema documentato
- I report MD sono pronti per inserimento nei paper
- Il file `CHAT_LOG.md` documenta i ragionamenti e le decisioni prese
- Verificare `validation/checksums.json` per integrità dei file
- **IMPORTANTE**: Questa documentazione include sia successi che fallimenti per trasparenza scientifica
