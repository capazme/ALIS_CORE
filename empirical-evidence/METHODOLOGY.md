# Metodologia - Evidenze Empiriche

**Data**: 2026-01-25
**Versione**: 1.0

---

## 1. Obiettivo

Generare evidenze empiriche che dimostrino:
1. L'architettura MERL-T è implementata e funzionante
2. Il framework RLCF è operativo con le formule matematiche del paper

---

## 2. Metodologia Generale

### 2.1 Approccio

- **Evidenze da sistema reale**: Utilizziamo la codebase ALIS_CORE già implementata
- **Dati sintetici controllati**: Per RLCF, generiamo dati con distribuzioni note
- **Riproducibilità**: Ogni esperimento è documentato e ripetibile

### 2.2 Validazione

- **Consistenza interna**: I risultati sono coerenti tra loro
- **Tracciabilità**: Ogni output è collegato al codice sorgente
- **Verificabilità**: Terze parti possono riprodurre i risultati

---

## 3. Evidenze MERL-T

### 3.1 KG Statistics

**Obiettivo**: Dimostrare che il Knowledge Graph è popolato e strutturato.

**Metodo**:
1. Connessione a FalkorDB (porta 6380)
2. Esecuzione query Cypher per statistiche
3. Aggregazione risultati

**Query eseguite**:
```cypher
-- Conteggio totale nodi
MATCH (n) RETURN count(n) as total_nodes

-- Conteggio per tipo
MATCH (n) RETURN labels(n)[0] as type, count(*) as count

-- Conteggio relazioni
MATCH ()-[r]->() RETURN type(r) as type, count(*) as count

-- Metriche connettività
MATCH (n)-[r]-() RETURN n, count(r) as degree ORDER BY degree DESC LIMIT 10
```

**Output atteso**: JSON con statistiche, MD con interpretazione.

### 3.2 Expert Pipeline Trace

**Obiettivo**: Dimostrare che i 4 Expert funzionano in pipeline.

**Metodo**:
1. Definizione 10 query giuridiche diversificate
2. Esecuzione via MultiExpertOrchestrator
3. Cattura trace completo (routing → expert → synthesis)

**Query di test** (categorie):
- Definitional: "Cos'è il contratto?"
- Literal: "Cosa dice l'art. 1453 c.c.?"
- Systemic: "Come si collega l'art. 2043 con l'art. 1218?"
- Constitutional: "Quali principi tutela l'art. 41 Cost.?"
- Precedent: "Cosa dice la Cassazione sulla buona fede?"

**Output atteso**: JSON trace per query, MD report aggregato.

### 3.3 Latency Benchmark

**Obiettivo**: Misurare performance del sistema.

**Metodo**:
1. Warm-up sistema
2. Misurazione cold start (prima esecuzione)
3. Misurazione cached (esecuzioni successive)
4. Calcolo percentili (p50, p95, p99)

**Metriche**:
- Embedding generation
- Vector search (Qdrant)
- Graph traversal (FalkorDB)
- LLM inference
- End-to-end pipeline

**Output atteso**: JSON con metriche, MD con tabelle.

---

## 4. Evidenze RLCF

### 4.1 Implementation Evidence

**Obiettivo**: Dimostrare che le formule del paper sono implementate.

**Metodo**:
1. Identificazione formule nel paper
2. Localizzazione nel codice sorgente
3. Estrazione snippet con contesto

**Formule verificate**:

| Formula | Paper | File:Linea |
|---------|-------|------------|
| A_u(t) = α·B_u + β·T_u(t) + γ·P_u(t) | Eq. 1 | authority.py:162-206 |
| δ = H(ρ)/log\|P\| | Eq. 2 | aggregation.py:10-46 |
| B_total = √(Σ b_i²) | Eq. 3 | bias_detection.py:768-770 |
| P(advocate) = min(0.1, 3/\|E\|) | Eq. 4 | devils_advocate.py:350-371 |

**Output atteso**: JSON mapping formula→codice, MD con snippet.

### 4.2 A/B Simulation

**Obiettivo**: Confrontare RLCF (authority-weighted) vs baseline (uniform).

**Metodo**:
1. Generazione 100 utenti sintetici con authority distribuita
2. Generazione 50 task con ground truth
3. Simulazione feedback con rumore
4. Aggregazione con authority weighting (A) vs uniforme (B)
5. Confronto convergenza al ground truth

**Configurazione**:
```yaml
num_users: 100
num_tasks: 50
iterations: 20
authority_distribution: pareto(alpha=2.0)
noise_level: 0.2
```

**Metriche**:
- Convergence rate
- Final accuracy
- Iterations to converge
- Variance reduction

**Output atteso**: JSON risultati, MD con grafici comparativi.

### 4.3 Bias Detection Demo

**Obiettivo**: Dimostrare che il sistema rileva bias.

**Metodo**:
1. Generazione 50 feedback sintetici con bias intenzionali
2. Esecuzione BiasDetector
3. Analisi output 6-dimensionale

**Bias introdotti**:
- b1 (demographic): Concentrazione per professione
- b2 (professional): Clustering per specializzazione
- b3 (temporal): Drift temporale
- b4 (geographic): Concentrazione regionale
- b5 (confirmation): Pattern di conferma
- b6 (anchoring): Effetto ancoraggio

**Output atteso**: JSON con 6 scores, MD con interpretazione.

---

## 5. Limitazioni

1. **Dati sintetici**: Per RLCF, i dati sono generati, non da utenti reali
2. **Scala ridotta**: 100 utenti, 50 task (sufficiente per demo)
3. **LLM variabilità**: Output LLM non deterministici (mitigato con temperature=0)
4. **Tempo limitato**: Scadenza oggi, alcune evidenze semplificate

---

## 6. Riproducibilità

### Environment

```
Python: 3.11+
FalkorDB: latest (Docker)
Qdrant: latest (Docker)
merlt: installed from source
```

### Random Seeds

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

### Versioning

Tutti i file includono timestamp e versione della codebase.
