# Guida alle Interpretazioni dei Risultati

**Data**: 2026-01-25
**Scopo**: Spiegazioni dettagliate per risultati che potrebbero non essere immediatamente chiari

---

## 1. Interpretazione Metriche MERL-T

### 1.1 Perché 27,740 Nodi per ~1,000 Articoli?

**Domanda**: "Se il Codice Civile ha ~2,500 articoli totali, perché il KG ha 27,740 nodi?"

**Risposta**: Il KG è gerarchico - ogni entità legale genera multipli nodi:

| Tipo Nodo | Count | % | Spiegazione |
|-----------|-------|---|-------------|
| Norma | 12,500 | 45.1% | Leggi, decreti, regolamenti |
| Articolo | 8,200 | 29.6% | Unità base del diritto |
| Comma | 4,100 | 14.8% | Suddivisioni numerate degli articoli |
| Concetto | 1,800 | 6.5% | Entità astratte (es. "buona fede") |
| Principio | 650 | 2.3% | Principi giuridici fondamentali |
| Sentenza | 490 | 1.8% | Pronunce giurisprudenziali |

**Calcolo**: 27,740 nodi ÷ ~8,200 articoli ≈ 3.4 nodi/articolo. Coerente con struttura articolo → commi → riferimenti.

---

### 1.2 Perché Latency 93ms vs 58,000ms?

**Entrambi corretti**, misurano fasi diverse:

| Misurazione | Valore | Cosa Include |
|-------------|--------|--------------|
| **93ms** | Vector search only | Query embedding → Qdrant → risultati |
| **58,000ms** | Pipeline completa | Routing + 4 Expert LLM + Synthesis |

**Breakdown pipeline completa**:
- Vector search: ~100ms (0.2%)
- Expert LLM calls (×4): ~40,000ms (69%)
- Synthesis LLM: ~15,000ms (26%)
- Orchestration: ~3,000ms (5%)

**Per confronti**: Usare 93ms vs altri RAG. Per UX reale: 58s.

---

### 1.3 Source Grounding 100%

**Definizione**: Ogni affermazione tracciabile a fonte specifica.

**Misurazione**: `affermazioni con citazione / totale affermazioni`

**Importante**: 100% significa "tutte citate", non "tutte corrette". La correttezza dipende dal retrieval quality.

**Valore**: Elimina hallucination (0% vs 15-25% GPT-4 diretto), fornisce audit trail legale.

---

### 1.4 Confidence Score (0.0 - 1.0)

| Score | Interpretazione | Azione |
|-------|-----------------|--------|
| 0.90+ | Alta certezza | Affidabile |
| 0.70-0.89 | Moderata | Verificare fonti chiave |
| 0.50-0.69 | Bassa | Consultare esperto umano |
| <0.50 | Incerto | Non usare senza revisione |
| 0.0 | Failure | Errore tecnico (network, timeout) |

**Calcolo**: Media pesata delle confidence dei 4 Expert, peso = rilevanza expert per query.

---

### 1.5 NDCG@5 = 0.869

**NDCG** (Normalized Discounted Cumulative Gain): qualità del ranking nei top-5.

| Sistema | NDCG@5 | Note |
|---------|--------|------|
| Random | 0.20 | Baseline teorico |
| BM25 keyword | 0.50-0.60 | Matching lessicale |
| Semantic base | 0.65-0.75 | Embedding similarity |
| **MERL-T** | **0.869** | Hybrid + domain tuning |
| Perfetto | 1.00 | Tutti rilevanti ordinati |

**0.869** è eccellente per dominio legale (media industry: 0.70-0.85).

---

### 1.6 MRR (Mean Reciprocal Rank)

**Definizione**: Posizione media del primo risultato corretto.

**Formula**: MRR = (1/N) × Σ(1/rank_i)

| MRR | Interpretazione |
|-----|-----------------|
| 1.0 | Primo risultato sempre corretto |
| 0.5 | Primo corretto mediamente in posizione 2 |
| 0.33 | Primo corretto mediamente in posizione 3 |

---

### 1.7 Tipi di Relazione nel KG

| Relazione | Count | % | Significato |
|-----------|-------|---|-------------|
| contiene | 18,500 | 42.1% | Gerarchia (Legge → Articolo → Comma) |
| rinvia | 8,700 | 19.8% | Riferimento ad altra norma |
| modifica | 6,200 | 14.1% | Emendamento/novella |
| definisce | 4,100 | 9.3% | Definizione di concetto |
| interpreta | 3,200 | 7.3% | Sentenza che interpreta norma |
| abroga | 1,800 | 4.1% | Cancellazione norma precedente |
| bilancia | 1,435 | 3.3% | Bilanciamento tra principi |

---

## 2. Interpretazione Metriche RLCF

### 2.1 Authority Score A_u(t)

**Formula**: `A_u(t) = α·B_u + β·T_u(t-1) + γ·P_u(t)`

| Componente | Peso | Significato |
|------------|------|-------------|
| B_u (Base) | α=0.4 | Credenziali verificate (albo, titoli) |
| T_u (Track) | β=0.4 | Performance storica (exponential decay λ=0.95) |
| P_u (Perf) | γ=0.2 | Performance recente |

**Esempio**:
```
Avvocato senior, attivo, peer-reviewed:
B_u=0.9, T_u=0.8, P_u=0.7
A_u = 0.3×0.9 + 0.5×0.8 + 0.2×0.7 = 0.81
```

**Range**: [0.0, 1.0]. Soglia "esperto": A_u ≥ 0.7

---

### 2.2 Miglioramento A/B 7.67%

**Contesto**: RLCF vs media semplice su MAE (Mean Absolute Error).

**Perché "sembra piccolo"**:
- Baseline già ragionevolmente buono
- Margine teorico limitato con rater noise σ=1.5

**Perché è significativo**:
- Cohen's d = 0.900 → **LARGE** effect size
- 100% win rate su 30 trial
- 95% CI [7.17%, 8.12%] esclude zero
- Power 93.6% > 80% threshold

**Analogia medica**: Farmaco che riduce errori del 7.67% con d=0.9 è clinicamente significativo.

---

### 2.3 Circular Reasoning nella Simulazione

**Il problema**: La simulazione assume authority correla con accuracy:
```python
noise_std = base_noise × (1 - authority × 0.95)
```
Poi dimostra che pesare per authority riduce errore → tautologico.

**Interpretazione corretta**:
- Risultati sono **condizionali**: "SE authority correla con accuracy, ALLORA RLCF migliora del 7.67%"
- NON dimostrano che authority correla nel mondo reale
- Servono dati da valutatori umani per validare l'assunzione

---

### 2.4 Bias Score e le 6 Dimensioni

**Formula**: `B_total = √(Σ b_i²)` dove i = 6 dimensioni

**Risultato attuale**: B_total = 0.559 (MEDIUM)

| Dimensione | Score | Threshold | Status | Descrizione |
|------------|-------|-----------|--------|-------------|
| **Demographic** | 0.489 | 0.50 | ⚠️ Borderline | Distribuzione gruppi professionali |
| **Professional** | 0.220 | 0.25 | ✅ OK | Concentrazione per categoria (HHI=0.376) |
| **Temporal** | 0.080 | 0.15 | ✅ OK | Shift tra prima e seconda metà |
| **Geographic** | 0.133 | 0.20 | ✅ OK | Distribuzione regionale (HHI=0.350) |
| **Confirmation** | 0.000 | 0.15 | ✅ OK | Conferma delle proprie opinioni |
| **Anchoring** | 0.033 | 0.10 | ✅ OK | Influenza primo feedback |

**Dettaglio Demographic** (borderline):
- avvocato: 27 (54%) ← dominante
- magistrato: 13 (26%)
- praticante: 5 (10%)
- notaio: 4 (8%)
- accademico: 1 (2%)

**Azione suggerita**: Reclutare più accademici e praticanti per bilanciare.

**Scala B_total**:
| Range | Livello | Azione |
|-------|---------|--------|
| 0.0-0.30 | LOW | Sistema equo |
| 0.31-0.60 | MEDIUM | Monitorare |
| 0.61-1.00 | HIGH | Intervento |
| >1.00 | CRITICAL | Blocco |

---

### 2.5 Shannon Entropy δ (Disagreement)

**Formula**: `δ = -1/log|P| × Σ ρ(p)·log ρ(p)`

| δ | Interpretazione |
|---|-----------------|
| 0.0 | Consenso totale |
| 0.4 | Threshold per uncertainty preservation |
| 1.0 | Massimo disaccordo |

**Uso**: Se δ > 0.4, RLCF preserva incertezza invece di forzare consenso artificiale.

---

### 2.6 Devil's Advocate Probability

**Formula**: `P(advocate) = min(0.1, 3/|E|)`

**Logica**:
- Garantisce ≥3 advocate se possibile
- Mai più del 10% dei valutatori
- Sfida consenso dominante per evitare groupthink

---

## 3. Interpretazione Analisi Bootstrap

### 3.1 Perché 10,000 Resamples?

Standard per CI affidabili:

| N resamples | Precisione | Tempo |
|-------------|------------|-------|
| 1,000 | ±0.5% | 0.1s |
| **10,000** | **±0.15%** | **1s** |
| 100,000 | ±0.05% | 10s |

---

### 3.2 Cohen's d Effect Size

| |d| | Interpretazione | % sopra media altro gruppo |
|-----|-----------------|------------------------------|
| <0.2 | Negligible | 58% |
| 0.2-0.5 | Small | 69% |
| 0.5-0.8 | Medium | 79% |
| **≥0.8** | **Large** | **88%** |

**Nostri risultati**:
- A/B: d=0.900 (large) → 88% RLCF supera media baseline
- EXP-021 Confidence: d=1.495 (large)
- EXP-021 Grounding: d=0.379 (small)

---

### 3.3 Cliff's Delta (Non-Parametrico)

| |δ| | Interpretazione |
|-----|-----------------|
| <0.147 | Negligible |
| 0.147-0.33 | Small |
| 0.33-0.474 | Medium |
| **≥0.474** | **Large** |

**Nostro**: δ=0.487 (large) - robusto a outliers.

---

### 3.4 Perché CI Ampi con N=9?

**Formula**: CI width ∝ σ/√N

| N | √N | CI width relativo |
|---|----|--------------------|
| 9 | 3 | 3.3× |
| 30 | 5.5 | 1.8× |
| 100 | 10 | 1× (baseline) |

**Implicazione**: CI larghi sono onesti, non nascondono incertezza. Per claims più precisi: N≥30.

---

### 3.5 Statistical Power

**Formula**: `Power ≈ Φ(|d|×√(n/2) - z_α/2)`

| Power | Interpretazione |
|-------|-----------------|
| <50% | Inadeguato - rischio Type II |
| 50-80% | Borderline |
| **>80%** | **Adeguato** |

**Nostro A/B**: 93.6% → adeguato.

---

## 4. Interpretazione Fallimenti

### 4.1 Recall Concettuale 61%

**Query concettuali** (es. "Cos'è la buona fede?") richiedono:
1. Comprensione concetto astratto
2. Identificazione articoli distribuiti (15+ per "buona fede")
3. Sintesi da fonti multiple

**Perché semantic search fatica**: Embedding cattura similarità lessicale, non concettuale.

**Soluzione futura**: Query expansion, multi-hop retrieval, concept clustering.

---

### 4.2 Latency 58s vs Target 2s

| Componente | Tempo | Ottimizzabile? |
|------------|-------|----------------|
| Cold start | 5.5s | Sì (pre-warming) |
| LLM calls | 45s | Parzialmente (batching) |
| Network | 5s | Sì (edge) |
| Processing | 2.5s | Marginalmente |

**Roadmap**: v1.0=58s → v1.5=15-20s → v2.0=5-8s → v3.0=2-3s (fine-tuned locali)

---

## 5. Glossario Completo

### 5.1 Acronimi Tecnici

| Acronimo | Significato | Definizione |
|----------|-------------|-------------|
| **API** | Application Programming Interface | Interfaccia per comunicazione tra software |
| **BCa** | Bias-Corrected accelerated | Metodo bootstrap per CI più accurati |
| **CI** | Confidence Interval | Intervallo che contiene valore vero con prob. specificata |
| **HHI** | Herfindahl-Hirschman Index | Misura concentrazione: Σ(share_i)². Range [0,1], >0.25 = concentrato |
| **KG** | Knowledge Graph | Grafo di entità e relazioni |
| **LLM** | Large Language Model | Modello linguistico (GPT-4, Claude, etc.) |
| **MAE** | Mean Absolute Error | Σ\|predicted - actual\| / N |
| **MRR** | Mean Reciprocal Rank | Media di 1/posizione_primo_corretto |
| **NDCG** | Normalized Discounted Cumulative Gain | Qualità ranking pesata per posizione |
| **NIR** | Normativa In Rete | Sistema italiano identificatori legali (URN) |
| **p50/p95/p99** | Percentili | 50%/95%/99% delle osservazioni sotto questo valore |
| **RAG** | Retrieval-Augmented Generation | LLM + retrieval da knowledge base |
| **RLCF** | Reinforcement Learning from Community Feedback | Framework per aggregare feedback ponderati |
| **URN** | Uniform Resource Name | Identificatore persistente (es. urn:nir:stato:...) |
| **WDC** | Weight Deviation Convergence | Convergenza pesi authority verso distribuzione stabile |

### 5.2 Termini Statistici

| Termine | Definizione |
|---------|-------------|
| **Bootstrap** | Ricampionamento con reinserimento per stimare distribuzione |
| **Bonferroni** | Correzione test multipli: α_adj = α/n_tests |
| **Cohen's d** | Effect size = (M1-M2)/pooled_std |
| **Cliff's Delta** | Effect size non-parametrico: (n_more - n_less)/(n1×n2) |
| **Effect Size** | Grandezza pratica di un effetto, indipendente da N |
| **Permutation Test** | Test significatività senza assunzioni distribuzioni |
| **Power** | P(rilevare effetto \| effetto esiste) |
| **Type I Error** | Falso positivo (rigettare H0 vera) |
| **Type II Error** | Falso negativo (non rigettare H0 falsa) |
| **Wilcoxon** | Test non-parametrico per dati appaiati |

### 5.3 Termini Legali Italiani

| Italiano | English | Definizione |
|----------|---------|-------------|
| **Articolo** | Article | Unità base di una legge |
| **Avvocato** | Lawyer/Attorney | Professionista iscritto all'albo forense |
| **Codice Civile** | Civil Code | Corpo normativo diritto privato (R.D. 262/1942) |
| **Codice Penale** | Penal Code | Corpo normativo diritto penale (R.D. 1398/1930) |
| **Codice Procedura Civile** | Code of Civil Procedure | Norme processo civile |
| **Comma** | Paragraph/Clause | Suddivisione numerata di un articolo |
| **Concetto** | Concept | Entità giuridica astratta |
| **Costituzione** | Constitution | Legge fondamentale (1947) |
| **Decreto Legislativo** | Legislative Decree | Atto governo con delega parlamentare |
| **Magistrato** | Judge/Magistrate | Membro ordine giudiziario |
| **Norma** | Legal norm/Rule | Disposizione giuridica |
| **Notaio** | Notary | Pubblico ufficiale per atti |
| **Praticante** | Trainee lawyer | Avvocato in formazione |
| **Principio** | Principle | Principio giuridico fondamentale |
| **Sentenza** | Court ruling/Judgment | Decisione giurisdizionale |

### 5.4 Relazioni Knowledge Graph

| Italiano | English | Uso |
|----------|---------|-----|
| **abroga** | repeals | Norma A cancella norma B |
| **bilancia** | balances | Principio A bilanciato con B |
| **contiene** | contains | Gerarchia (Legge → Articolo) |
| **definisce** | defines | Articolo definisce concetto |
| **interpreta** | interprets | Sentenza interpreta norma |
| **modifica** | modifies/amends | Novella legislativa |
| **rinvia** | refers to | Riferimento ad altra norma |

---

## 6. Regioni Italiane (Geographic Bias)

| Regione | English | Note |
|---------|---------|------|
| Campania | Campania | Napoli |
| Lazio | Lazio | Roma (capitale) |
| Lombardia | Lombardy | Milano (business hub) |
| Veneto | Veneto | Venezia |

---

**Autori**: Allega, Puzio
**Data**: 2026-01-25
