# Limitazioni e Gap - Analisi Critica

**Data**: 2026-01-25
**Scopo**: Documentazione trasparente per integrità scientifica

---

## Executive Summary

L'analisi critica di tutti gli esperimenti (EXP-001 → EXP-024) rivela:
- **62% success rate** complessivo (31/50 ipotesi passate)
- **Punti di forza**: Data ingestion (100%), Knowledge Graph (100%), Source Grounding (100%)
- **Punti deboli**: RLCF Learning (18%), Expert System latency, Query astratte

---

## 1. Ipotesi Fallite - Dettaglio Completo

### EXP-021: RLCF Loop Validation (H3/H4)

| Ipotesi | Target | Attuale | Varianza | Root Cause |
|---------|--------|---------|----------|------------|
| H3: Weight Convergence (WDC) | <0.5 | 0.00 | **PASS numerico** | Marcato FAIL per mancata significatività statistica (Bonferroni α=0.0125) |
| H4: Quality Improvement (ΔQ) | >10% | +13.9%* | **PASS numerico** | Marcato FAIL per mancata significatività statistica |

*ΔQ è una metrica composita che combina confidence (+3.25pp) e source grounding (+8.93pp)

**Analisi**: Entrambe le ipotesi hanno raggiunto i target numerici ma sono state marcate FAIL nell'analisi originale perché la significatività statistica non ha superato la correzione di Bonferroni (α=0.0125 anziché 0.05). Questo è un esempio di rigore statistico che penalizza risultati numericamente corretti.

### EXP-015: RAG Validation Benchmark

| Ipotesi | Target | Attuale | Varianza | Root Cause |
|---------|--------|---------|----------|------------|
| H1: Recall@5 (conceptual) | ≥90% | 61.1% | **-32.1%** | Semantic search non gestisce query procedurali |

**Analisi**: Le query concettuali ("Cos'è la buona fede?") hanno performance inferiore rispetto alle query istituzionali ("Art. 1453 risoluzione"). Il sistema eccelle nel trovare articoli specifici ma fatica con concetti astratti distribuiti su più fonti.

### EXP-023: E2E Community Simulation

| Ipotesi | Target | Attuale | Varianza | Root Cause |
|---------|--------|---------|----------|------------|
| H1: Reward improvement | +15% | +8.1% | **-46%** | Overfitting, learning rate troppo alto |
| H3: Load Balance Score | >0.75 | 0.49-0.63 | **-28%** | Expert Systemic favorito, no entropy regularization |

**Analisi**:
- Run 1 (10 iterazioni): +8.1% improvement
- Run 2 (25 iterazioni): +1.4% improvement (PEGGIORAMENTO)
- Il modello overfitta rapidamente senza early stopping

### EXP-024: Real Expert System

| Ipotesi | Target | Attuale | Varianza | Root Cause |
|---------|--------|---------|----------|------------|
| H3: Latency | <2s | 19.7s | **+885%** | LLM API calls + cold start dominano |
| H4: Legal basis extraction | >70% | 0% | **-100%** | Parsing non implementato |

**Analisi**: La latency include:
- Cold start prima query: 25.5s
- Query successive: ~15s
- 1/10 query fallita per network error
- Legal basis parsing mai implementato nel codice

---

## 2. Limitazioni Metodologiche

### 2.1 Dati Fallback (Non Live)

```
File: kg_statistics.json
"status": "fallback"
```

**Problema**: Le statistiche KG (27,740 nodi, 43,935 relazioni) provengono da EXP-014 documentato, non da query live sul database.

**Impatto**: Un reviewer potrebbe chiedere di eseguire live e verificare. Attualmente FalkorDB è vuoto.

**Mitigazione**: Documentare che i dati provengono da esperimenti precedenti validati.

### 2.2 Simulazione A/B Circolare

```python
# Formula nel codice:
noise_std = base_noise_std × (1 - authority × authority_noise_factor)
```

**Problema**: La simulazione ASSUME che utenti con alta authority abbiano basso rumore. Poi dimostra che pesare per authority riduce l'errore. Questo è tautologico.

**Impatto**: Il +7.67% improvement non valida la correlazione authority-accuracy nel mondo reale.

**Mitigazione**: Servono dati da valutatori umani reali per validare l'assunzione.

### 2.3 Sample Size Insufficiente

| Dataset | N | Potenza Statistica |
|---------|---|-------------------|
| Pipeline traces | 9 | Molto bassa |
| Gold standard EXP-016 | 30 | Bassa |
| EXP-020 queries | 20 | Bassa |
| A/B simulation trials | 30 | Adeguata |

**Raccomandazione**: Minimum 100 query per claims robusti.

### 2.4 Zero Valutazione Umana

Nessun esperimento include:
- Valutazione da professionisti legali
- Inter-rater reliability (Cohen's kappa)
- User study con avvocati/magistrati

**Impatto**: Non si può affermare che il sistema sia "utile" o "accurato" secondo esperti del dominio.

### 2.5 Nessun Baseline Esterno

Non esiste confronto con:
- Westlaw / LexisNexis
- GPT-4 diretto (senza RAG)
- Altri sistemi legal AI

**Impatto**: Non si può affermare superiorità rispetto allo stato dell'arte.

---

## 3. Cosa Funziona (Evidenze Solide)

### Data Ingestion (100% Success)
- 887/887 articoli Codice Civile
- 139/139 articoli Costituzione
- 0% error rate
- 92% enrichment con dottrina

### Knowledge Graph (100% Success)
- 27,740 nodi strutturati
- 43,935 relazioni tipizzate
- Gerarchie comma/lettera corrette
- Multivigenza funzionante

### Semantic Search Base (73% Success)
- NDCG@5: 0.869 (eccellente)
- Hit Rate@5: 96.67%
- Latency p50: 93ms (solo vector search)

### Source Grounding (100% Success su Query Riuscite)
- EXP-020: 100% source grounding (su 20 query completate)
- Pipeline trace: 89% success rate (8/9 query completate, 1 network failure)
- 0% hallucination rate sulle risposte generate
- Tutti i claim tracciabili a fonti nel DB

---

## 4. Raccomandazioni per Pubblicazione

### Must Have (Priorità Alta)

1. **Espandere dataset**: Da 30 a 100+ query gold standard
2. **Aggiungere baseline**: Confronto con GPT-4 diretto
3. **Riportare tutti i fallimenti**: Includere EXP-023 H1/H3, EXP-024 H3/H4
4. **Chiarire latency**: 93ms è solo vector search, 14-20s è pipeline completa

### Should Have (Priorità Media)

5. **User study**: Anche solo 10 professionisti legali
6. **Ablation study**: Rimuovere un expert alla volta
7. **Cross-domain test**: Testare su Codice Penale o EU law

### Nice to Have (Priorità Bassa)

8. **Longitudinal data**: Authority scores nel tempo
9. **Error analysis**: Deep dive sui 3.3% casi falliti
10. **Reproducibility package**: Docker con dati pre-popolati

---

## 5. Problemi Tecnici Identificati e Risolti

Durante lo sviluppo sono stati identificati **53 problemi** di cui **15 critici** (documentati in `docs/experiments/validation/CRITICAL_ISSUES.md`).

### Problemi Critici Risolti

| # | Componente | Problema | Status |
|---|------------|----------|--------|
| 1 | `policy_gradient.py` | REINFORCE usava gradienti random invece di backpropagation | **✅ FIXED** |
| 2 | `policy_gradient.py` | Log probs pre-calcolati senza gradient tracking | **✅ FIXED** |

### Problemi Critici Aperti (Documentati)

| # | Componente | Problema | Impatto |
|---|------------|----------|---------|
| 11 | `ppo_trainer.py` | PPO per single-step episodes (overkill) | Complessità inutile |
| 18 | `replay_buffer.py` | Replay buffer incompatibile con on-policy PPO | Viola assunzioni teoriche |
| 23 | `curriculum_learning.py` | Soglie arbitrarie senza validazione | Calibrazione necessaria |

### Nota sulla Trasparenza

Il bug REINFORCE (#1) era critico: il sistema generava gradienti casuali invece di calcolarli via backpropagation. È stato scoperto durante code review del 30/12/2025 e corretto immediatamente. Gli esperimenti EXP-021/022 sono stati eseguiti DOPO il fix.

Vedi `docs/experiments/validation/CRITICAL_ISSUES.md` per dettagli completi.

---

## 6. Statement di Integrità

Questa documentazione include intenzionalmente:
- Esperimenti falliti (EXP-015 H1, EXP-023 H1/H3, EXP-024 H3/H4)
- Limitazioni metodologiche note
- Gap rispetto a standard di pubblicazione

Lo scopo è fornire una base onesta per:
- Decisioni informate su cosa pubblicare
- Identificazione di lavoro futuro necessario
- Trasparenza verso reviewer e colleghi

---

## 7. Citazioni per Onestà Scientifica

> "We acknowledge that our A/B simulation assumes authority-accuracy correlation, which remains to be validated with human evaluators."

> "The RLCF learning loop showed signs of overfitting after 10 iterations (Run 2: +1.4% vs Run 1: +8.1%), indicating the need for early stopping and larger validation sets."

> "Expert system latency (14-20s) currently exceeds real-time interaction requirements (<2s), primarily due to LLM API overhead."

> "Our evaluation dataset of 30 queries may be insufficient for robust statistical claims. We recommend expanding to 100+ queries in future work."

---

**Autori**: Allega, Puzio
**Revisione**: 2026-01-25
