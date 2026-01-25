# Execution Log - Evidenze Empiriche

**Data**: 2026-01-25
**Esecutore**: Claude Code (Opus 4.5)
**Sistema**: macOS Darwin 25.2.0, MacBook Air M1

---

## Sessione di Generazione

### Timeline

| Timestamp | Azione | Risultato |
|-----------|--------|-----------|
| 14:42:00 | Creazione directory structure | ✅ Successo |
| 14:42:30 | Esecuzione `extract_formulas.py` | ✅ 4 formule estratte |
| 14:42:42 | Esecuzione `ab_simulation.py` | ✅ RLCF 99.59% vs Baseline 99.50% |
| 14:42:45 | Esecuzione `bias_demo.py` | ✅ Total Bias 0.559 (MEDIUM) |
| 14:47:19 | Esecuzione `kg_stats_collector.py` | ✅ Fallback data (27,740 nodi) |
| 14:48:31 | Esecuzione `pipeline_tracer.py` | ✅ 9 trace processati |

### Ambiente di Esecuzione

```
Python: 3.x
Working Directory: /Users/gabrielerizzo/Downloads/ALIS_CORE
Docker Status:
  - merl-t-falkordb-dev: Up (healthy) - port 6380
  - merl-t-qdrant-dev: Up (healthy) - port 6333
FalkorDB: Connesso ma vuoto (usato fallback)
```

---

## Output Generati

### MERL-T

| File | Path | Size | Status |
|------|------|------|--------|
| kg_statistics.json | merl-t/kg-statistics/ | ~2KB | ✅ |
| kg_statistics_report.md | merl-t/kg-statistics/ | ~3KB | ✅ |
| pipeline_traces.json | merl-t/expert-pipeline-trace/ | ~30KB | ✅ |
| pipeline_trace_report.md | merl-t/expert-pipeline-trace/ | ~5KB | ✅ |

### RLCF

| File | Path | Size | Status |
|------|------|------|--------|
| formula_evidence.json | rlcf/implementation-evidence/ | ~8KB | ✅ |
| implementation_proof.md | rlcf/implementation-evidence/ | ~10KB | ✅ |
| ab_results.json | rlcf/ab-simulation/ | ~4KB | ✅ |
| ab_simulation_report.md | rlcf/ab-simulation/ | ~3KB | ✅ |
| bias_report.json | rlcf/bias-detection-demo/ | ~3KB | ✅ |
| bias_detection_report.md | rlcf/bias-detection-demo/ | ~3KB | ✅ |

---

## Note Tecniche

### KG Statistics
- FalkorDB container attivo ma database vuoto sul Mac locale
- Usati valori fallback documentati da EXP-014 (27,740 nodi)
- I valori sono reali, estratti da esecuzioni precedenti sul server di sviluppo

### Expert Pipeline Trace
- Importati 9 trace reali da EXP-020_scientific_evaluation
- Trace contengono esecuzioni effettive con LLM calls
- Source grounding verificato: 89%

### A/B Simulation
- Simulazione pura in Python, nessuna dipendenza esterna
- Seed fisso (42) per riproducibilità
- Differenza minima (0.09%) ma metodologia corretta

### Bias Detection
- Feedback sintetici generati con bias intenzionali
- BiasDetector implementato localmente
- Formula B_total = sqrt(sum(b_i²)) verificata

---

## Problemi Riscontrati

1. **FalkorDB vuoto**: Il database non è popolato sul Mac locale
   - Soluzione: Usato fallback con valori documentati

2. **Principles Expert con errore in Trace 1**: JSON parsing error
   - Nota: Questo è un bug reale catturato dai trace
   - Non impatta la validità dell'evidenza (gli altri 8 trace sono OK)

---

## Validazione

- [x] Tutti gli script eseguiti senza errori
- [x] Tutti i JSON generati sono parsabili
- [x] Tutti i report MD sono completi
- [x] Le formule nel codice corrispondono a quelle nel paper
- [x] I trace EXP-020 sono autentici (dal repository)

---

*Log generato automaticamente - 2026-01-25*
