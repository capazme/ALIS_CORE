---
stepsCompleted: [1, 2, 3]
inputDocuments:
  - docs/project-documentation/index.md
  - docs/project-documentation/00-project-overview.md
  - docs/project-documentation/02-merlt-experts.md
  - docs/project-documentation/03-rlcf.md
session_topic: "Architettura e riorganizzazione dei componenti core di ALIS"
session_goals: "Codebase riorganizzata, architettura modulare, documentazione scientifica per tesi"
selected_approach: "progressive-flow"
techniques_used:
  - phase1: "first-principles-thinking"
  - phase2: "mind-mapping"
  - phase3: "scamper-method"
  - phase4: "decision-tree-mapping"
phase_status:
  phase1: "completed"
  phase2: "completed"
  phase3: "completed"
  phase4: "completed"
session_status: "completed"
ideas_generated:
  - "Principio di Stratificazione architetturale"
  - "Principio di Tracciabilità (AI come processo)"
  - "Principio di Sequenzialità Canonica"
  - "Principio del Diritto Vivente (RLCF)"
  - "Principio della Nuova Scuola (glossatori algoritmici)"
  - "Isomorfismo shortest path ↔ significato proprio"
  - "Isomorfismo prudentes/responsa ↔ authority/feedback"
context_file: "docs/project-scan-report.json"
---

# Brainstorming Session: ALIS Core Architecture

**Date:** 2026-01-23
**Facilitator:** AI Brainstorming Coach
**Participant:** Gpuzio

---

## Session Overview

**Topic:** Architettura e riorganizzazione dei componenti core di ALIS

**Goals:**
- Riorganizzare le codebase per coerenza e manutenibilità
- Creare un'architettura funzionante, chiara e modulare
- Preparare per distribuzione controllata (open-source vs proprietario)
- Documentare con rigore scientifico per la tesi di laurea

### Context

Questo brainstorming supporta una tesi di laurea in **Metodologia delle Scienze Giuridiche** presso un'importante università privata italiana. Il progetto ALIS (Artificial Legal Intelligence System) implementa computazionalmente i canoni ermeneutici dell'Art. 12 delle Preleggi.

**Stakeholder:** ~20 professionisti/colleghi nell'associazione ALIS

**Academic Foundation:**
- MERL-T: Multi-Expert architecture (4 canoni ermeneutici)
- RLCF: Reinforcement Learning from Community Feedback (4 pilastri)
- Knowledge Commoditization Paradox

### Session Setup

**Approach Selected:** Progressive Flow (Flusso Progressivo)
- Start broad with divergent exploration
- Systematically narrow toward concrete solutions
- Layer techniques for depth

---

## Technique Selection

**Approach:** Progressive Technique Flow
**Journey Design:** Sviluppo sistematico dall'esplorazione all'azione

**Progressive Techniques:**

| Fase | Tecnica | Scopo |
|------|---------|-------|
| **Fase 1 - Esplorazione** | First Principles Thinking | Ricostruire dai fondamenti giuridici |
| **Fase 2 - Pattern Recognition** | Mind Mapping | Visualizzare connessioni tra componenti |
| **Fase 3 - Sviluppo** | SCAMPER Method | Raffinamento sistematico architettura |
| **Fase 4 - Action Planning** | Decision Tree Mapping | Roadmap implementativa |

---

## Brainstorming Content

### FASE 1: First Principles Thinking

#### Substrato Filosofico

*Estratto dai contributi accademici del partecipante*

**Principi Fondamentali:**

1. **AI come PROCESSO, mai come AGENTE**
   - "Un singolare ammasso di minerali elettrificati"
   - Responsabilità sempre riconducibile all'umano
   - Implicazione architetturale: ogni output deve essere tracciabile a decisioni umane

2. **Bias come grammatica dell'apprendimento**
   - Non difetto, ma struttura inevitabile della cognizione artificiale
   - Implicazione: rendere espliciti i bias, non eliminarli

3. **Creatività stratificata** (*layered creativity*)
   - Ogni contributo è composizione di contributi precedenti
   - Opera derivata e composta simultaneamente
   - Implicazione: architettura deve riflettere stratificazione delle fonti

4. **Volksgeist algoritmico**
   - LLM come cristallizzazione di valori culturali
   - "Corpus interrogabile" del sapere giuridico
   - Implicazione: community feedback come voce del Volksgeist vivente

5. **Diritto vivente di Ehrlich → RLCF**
   - Comportamento emergente vs guardrails espliciti
   - Il diritto "smette di essere teoria e diventa tecnica, se non addirittura framework"

---

#### Parallelo Storico: Irnerius e i Glossatori (1100)

> "Come Irnerio che nel 1100 creò una nuova metodologia per sistematizzare l'antica conoscenza, fondando la scuola dei glossatori per sistematizzare il Corpus Juris Civilis, così noi siamo la nuova comunità dei sapientes del diritto algoritmico."

**Metodologia di Irnerio:**
- Lettura del testo ad alta voce → studenti copiano
- Excursus esplicativo sotto forma di **glosse**
- Glosse interlineari → glosse marginali → corpus sistematico
- Risultato: diritto europeo scritto, sistematico, completo e razionale

**Parallelo con ALIS:**

| Glossatori (1100) | ALIS (2026) |
|-------------------|-------------|
| Corpus Juris Civilis | Corpus legislativo italiano |
| Glosse marginali | Expert annotations |
| Metodologia sistematica | MERL-T pipeline |
| Ius commune europeo | Interpretazione algoritmica condivisa |
| Lucerna juris (Irnerio) | Artificial Legal Intelligence |

---

#### Disputatio Fori e Prudentes

**Il modello romano:**
- *Prudentes*: giuristi che "respondēre" - analisi del caso concreto per elaborare regola giuridica autorevole
- *Ius publice respondendi ex auctoritate principis* (Augusto): responsa sottoscritti da giuristi "patentati"
- *Disputatio fori*: discussione nel foro che genera diritto vivente

**Isomorfismo con RLCF:**

| Diritto Romano | RLCF Framework |
|----------------|----------------|
| Prudentes con ius respondendi | Utenti con Authority Score |
| Responsa signata | Feedback verificato e pesato |
| Disputatio fori | Community discussion threads |
| Autorevolezza per competenza | Dynamic Authority Scoring |
| Creazione giurisprudenziale | Policy learning from feedback |

**Insight chiave:** L'ordinamento romano era "caratterizzato da una componente fortemente giurisprudenziale" dove "l'interpretazione giurisprudenziale finiva per essere una vera e propria attività di creazione del diritto."

→ RLCF implementa tecnicamente questo principio: la community ALIS crea diritto interpretativo attraverso feedback strutturato.

---

#### Gerarchia Sequenziale dei Canoni

**Dottrina consolidata (Art. 12 Preleggi):**

```
1. Significato proprio delle parole (letterale)
        ↓ (se insufficiente)
2. Connessione delle parole (sistematico)
        ↓ (se insufficiente)
3. Intenzione del legislatore (teleologico)
        ↓ (sempre applicabile)
4. Precedenti e prassi (prudenziale)

[Art. 14: eccezioni per leggi penali e speciali]
```

**Implicazione architetturale:** Gli Expert devono rispettare questa gerarchia sequenziale, non operare in parallelo equipollente.

---

#### "Significato Proprio delle Parole" come Shortest Path

**Ipotesi computazionale:**

Nel contesto dell'iperspazio vettoriale semantico, il "significato proprio delle parole" può essere interpretato come lo **shortest path** - il percorso più breve tra il significante e il significato tecnico-giuridico.

**Fondamento in computational linguistics:**

- **Semantic similarity** in vector space: "words that occur in the same contexts tend to have similar meanings" (Firth)
- **Shortest path approaches**: distanza geodetica tra nodi in un grafo tassonomico ontologico
- **IC-weighted path distance**: pesatura del percorso con Information Content

**Isomorfismo proposto:**

| Concetto giuridico | Implementazione vettoriale |
|--------------------|---------------------------|
| Significato proprio | Shortest path nel knowledge graph |
| Accezione tecnica | Cluster specialistico (legal domain) |
| Ambiguità semantica | Distanza tra cluster alternativi |
| Interpretazione estensiva | Path più lungo, attraverso nodi intermedi |

**Ricerca necessaria:** Formalizzare l'isomorfismo tra:
- Cosine similarity in embedding space
- Path distance in knowledge graph
- Concetto giuridico di "significato proprio"

---

#### First Principles: Domande Fondamentali

**Q1: Cosa significa "interpretare" computazionalmente?**
→ Navigare uno spazio semantico strutturato secondo regole epistemiche codificate

**Q2: Qual è l'unità atomica dell'interpretazione giuridica?**
→ Il *responsum*: query + contesto + risposta autorevole + traccia di ragionamento

**Q3: Come si stratifica l'autorevolezza?**
→ Dottrina consolidata → Giurisprudenza → Prassi → Opinione comunitaria (RLCF weights)

**Q4: Qual è il confine tra open-source e proprietario?**
→
- **Open:** Metodologia (MERL-T paper), infrastruttura generica, canoni universali
- **Proprietario:** Pesi addestrati, knowledge graph popolato, authority scores, policy checkpoints

---

#### Sintesi Fase 1: Principi Architetturali Emergenti

1. **Principio di Stratificazione**
   - L'architettura deve riflettere la stratificazione delle fonti giuridiche
   - Ogni layer aggiunge autorevolezza e specificità

2. **Principio di Tracciabilità**
   - Ogni output riconducibile a: fonte normativa + expert + reasoning trace
   - Nessuna "black box" - AI come processo, non agente

3. **Principio di Sequenzialità Canonica**
   - Expert pipeline rispetta gerarchia Art. 12
   - LiteralExpert → SystemicExpert → PrinciplesExpert → PrecedentExpert

4. **Principio del Diritto Vivente**
   - RLCF come implementazione tecnica del diritto vivente di Ehrlich
   - Community feedback modifica policy, non hard-coded rules

5. **Principio della Nuova Scuola**
   - ALIS come "nuova comunità dei sapientes del diritto algoritmico"
   - Documentazione come moderne glosse marginali
   - Tesi come manifesto metodologico

---

**Fonti consultate:**
- [Irnerius - Wikipedia](https://en.wikipedia.org/wiki/Irnerius)
- [La Rivoluzione di Irnerio e la Scuola dei Glossatori](https://www.avvfiorenzoauteri.com/post/la-rivoluzione-di-irnerio-e-la-scuola-dei-glossatori-civilisti)
- [Iuris Prudentes - Wikipedia](https://it.wikipedia.org/wiki/Iuris_Prudentes)
- [Responsa Prudentium nel diritto romano](https://www.iurisprudentes.it/2019/08/30/breve-nota-sulla-natura-giuridica-dei-responsa-prudentium-nel-diritto-privato-romano/)
- [Semantic Similarity - Wikipedia](https://en.wikipedia.org/wiki/Semantic_similarity)
- [Vector Space Semantics](https://alvinntnu.github.io/NTNU_ENC2036_LECTURES/vector-space-representation.html)

---

### FASE 2: Mind Mapping

*Visualizzazione delle connessioni tra principi filosofici, componenti tecnici e architettura*

#### Mappa Centrale: ALIS Core

```
                                    ┌─────────────────────────────────────┐
                                    │         THESIS FRAMEWORK            │
                                    │   "Metodologia delle Scienze        │
                                    │         Giuridiche"                 │
                                    └───────────────┬─────────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │                               │                               │
                    ▼                               ▼                               ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
    │    THEORETICAL LAYER      │   │    COMPUTATIONAL LAYER    │   │    COMMUNITY LAYER        │
    │                           │   │                           │   │                           │
    │  • Art. 12 Preleggi       │   │  • MERL-T Framework       │   │  • RLCF Framework         │
    │  • Canoni ermeneutici     │   │  • Expert Pipeline        │   │  • ~20 professionisti     │
    │  • Dottrina consolidata   │   │  • Knowledge Graph        │   │  • Authority Scoring      │
    │  • Ehrlich diritto vivente│   │  • Vector Search          │   │  • Feedback loops         │
    └───────────┬───────────────┘   └───────────┬───────────────┘   └───────────┬───────────────┘
                │                               │                               │
                └───────────────────────────────┼───────────────────────────────┘
                                                │
                                                ▼
                            ┌───────────────────────────────────────┐
                            │           ALIS MONOREPO               │
                            │                                       │
                            │   merlt ─── visualex-api ─── platform │
                            │      │           │               │    │
                            │   models    visualex-merlt    frontend│
                            └───────────────────────────────────────┘
```

---

#### Cluster 1: Fondamenti Giuridici → Componenti Tecnici

```
CANONI ERMENEUTICI (Art. 12)              MERL-T EXPERTS
═══════════════════════════               ══════════════

┌─────────────────────────┐               ┌─────────────────────────┐
│ 1. LETTERALE            │──────────────▶│ LiteralExpert           │
│    "significato proprio"│               │ • Shortest path search  │
│                         │               │ • Qdrant vectors        │
└─────────────────────────┘               └─────────────────────────┘
            │
            ▼ (se insufficiente)
┌─────────────────────────┐               ┌─────────────────────────┐
│ 2. SISTEMATICO          │──────────────▶│ SystemicExpert          │
│    "connessione parole" │               │ • FalkorDB graph        │
│                         │               │ • Relazioni normative   │
└─────────────────────────┘               └─────────────────────────┘
            │
            ▼ (se insufficiente)
┌─────────────────────────┐               ┌─────────────────────────┐
│ 3. TELEOLOGICO          │──────────────▶│ PrinciplesExpert        │
│    "intenzione"         │               │ • Lavori preparatori    │
│                         │               │ • Ratio legis           │
└─────────────────────────┘               └─────────────────────────┘
            │
            ▼ (sempre)
┌─────────────────────────┐               ┌─────────────────────────┐
│ 4. PRUDENZIALE          │──────────────▶│ PrecedentExpert         │
│    "precedenti, prassi" │               │ • Case law embedding    │
│                         │               │ • Massime giurisp.      │
└─────────────────────────┘               └─────────────────────────┘
```

---

#### Cluster 2: Paralleli Storici → Architettura

```
                    GLOSSATORI (1100)                    ALIS (2026)
                    ═════════════════                    ═══════════

                    Corpus Juris Civilis ─────────────▶ Corpus Legislativo IT
                           │                                    │
                           ▼                                    ▼
                    ┌─────────────┐                     ┌─────────────┐
                    │   GLOSSE    │                     │  EXPERT     │
                    │ interlineari│ ═══════════════════▶│ ANNOTATIONS │
                    │  marginali  │                     │ + reasoning │
                    └─────────────┘                     └─────────────┘
                           │                                    │
                           ▼                                    ▼
                    ┌─────────────┐                     ┌─────────────┐
                    │   SCUOLA    │                     │    ALIS     │
                    │  BOLOGNA    │ ═══════════════════▶│ COMMUNITY   │
                    │  studenti   │                     │  20 membri  │
                    └─────────────┘                     └─────────────┘
                           │                                    │
                           ▼                                    ▼
                    ┌─────────────┐                     ┌─────────────┐
                    │    IUS      │                     │ DIRITTO     │
                    │   COMMUNE   │ ═══════════════════▶│ ALGORITMICO │
                    │   europeo   │                     │  condiviso  │
                    └─────────────┘                     └─────────────┘


                    DIRITTO ROMANO                       RLCF
                    ══════════════                       ════

                    ┌─────────────┐                     ┌─────────────┐
                    │  PRUDENTES  │ ═══════════════════▶│   USERS +   │
                    │ (patentati) │                     │ AUTH SCORE  │
                    └─────────────┘                     └─────────────┘
                           │                                    │
                           ▼                                    ▼
                    ┌─────────────┐                     ┌─────────────┐
                    │  RESPONSA   │                     │  FEEDBACK   │
                    │   signata   │ ═══════════════════▶│  verificato │
                    └─────────────┘                     └─────────────┘
                           │                                    │
                           ▼                                    ▼
                    ┌─────────────┐                     ┌─────────────┐
                    │ DISPUTATIO  │                     │  COMMUNITY  │
                    │    FORI     │ ═══════════════════▶│  THREADS    │
                    └─────────────┘                     └─────────────┘
```

---

#### Cluster 3: Monorepo Structure → Principi Architetturali

```
PRINCIPI FASE 1                        COMPONENTI MONOREPO
═══════════════                        ════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│                           STRATIFICAZIONE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   LAYER 3: Presentation       ─────────▶  visualex-platform/frontend   │
│                                           visualex-merlt (plugin)       │
│                                                                         │
│   LAYER 2: Application        ─────────▶  visualex-platform/backend    │
│                                           merlt/api (FastAPI)           │
│                                           visualex-api (Quart)          │
│                                                                         │
│   LAYER 1: Core ML            ─────────▶  merlt/experts                │
│                                           merlt/rlcf                    │
│                                           merlt-models                  │
│                                                                         │
│   LAYER 0: Data               ─────────▶  PostgreSQL, FalkorDB,        │
│                                           Qdrant, Redis                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           TRACCIABILITÀ                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Query ──▶ MERL-T API ──▶ Expert(s) ──▶ Response                 │
│        │              │              │              │                   │
│        ▼              ▼              ▼              ▼                   │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐               │
│   │trace_id │   │ expert  │   │reasoning│   │ sources │               │
│   │timestamp│   │  type   │   │  trace  │   │  cited  │               │
│   │user_auth│   │ params  │   │  steps  │   │  URNs   │               │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘               │
│                                                                         │
│   ──────────────────▶ rlcf_traces table ◀──────────────────            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      SEQUENZIALITÀ CANONICA                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ExpertRouter Decision Logic:                                          │
│                                                                         │
│   ┌──────────────┐                                                     │
│   │ LiteralExpert│ ──▶ response_sufficient? ──▶ YES ──▶ RETURN         │
│   └──────────────┘              │                                       │
│                                 NO                                      │
│                                 ▼                                       │
│   ┌───────────────┐                                                    │
│   │ SystemicExpert│ ──▶ response_sufficient? ──▶ YES ──▶ RETURN        │
│   └───────────────┘              │                                      │
│                                 NO                                      │
│                                 ▼                                       │
│   ┌──────────────────┐                                                 │
│   │ PrinciplesExpert │ ──▶ response_sufficient? ──▶ YES ──▶ RETURN     │
│   └──────────────────┘              │                                   │
│                                 NO/ALWAYS                               │
│                                 ▼                                       │
│   ┌─────────────────┐                                                  │
│   │ PrecedentExpert │ ──▶ INTEGRATE + RETURN                           │
│   └─────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

#### Cluster 4: Open-Source vs Proprietario

```
                           DISTRIBUTION STRATEGY
                           ═════════════════════

        ┌─────────────────────────────────────────────────────┐
        │                   OPEN SOURCE                        │
        │                   (GitHub Public)                    │
        ├─────────────────────────────────────────────────────┤
        │                                                      │
        │   📄 Papers (MERL-T, RLCF, ALIS)                    │
        │   📐 Architecture diagrams                          │
        │   🔧 Generic infrastructure                          │
        │      • Expert base classes                          │
        │      • RLCF framework (sans weights)                │
        │      • Plugin system interfaces                     │
        │   📚 Documentation (tesi-ready)                     │
        │   🧪 Test fixtures (anonymized)                     │
        │                                                      │
        └─────────────────────────────────────────────────────┘
                                │
                     ╔══════════╧══════════╗
                     ║   BOUNDARY LINE     ║
                     ║  "Valore Aggiunto"  ║
                     ╚══════════╤══════════╝
                                │
        ┌─────────────────────────────────────────────────────┐
        │                  PROPRIETARY                         │
        │               (ALIS Association)                     │
        ├─────────────────────────────────────────────────────┤
        │                                                      │
        │   🧠 Trained model weights                          │
        │      • merlt-models/*.safetensors                   │
        │   📊 Populated knowledge graph                       │
        │      • FalkorDB data                                │
        │      • Qdrant collections                           │
        │   👤 Authority scores                                │
        │      • user_authority table                         │
        │   ⚙️ Policy checkpoints                              │
        │      • policy_checkpoints table                     │
        │   🔐 ALIS member access credentials                 │
        │                                                      │
        └─────────────────────────────────────────────────────┘
```

---

#### Cluster 5: Data Flow Completo

```
                              ALIS DATA FLOW
                              ══════════════

    ┌─────────┐
    │  USER   │
    │ (member)│
    └────┬────┘
         │ query
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        VISUALEX-PLATFORM                                │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐│
│  │    Frontend     │◀────▶│     Backend     │◀────▶│   PostgreSQL    ││
│  │  (React + MERLT)│      │    (Express)    │      │  (users, auth)  ││
│  └────────┬────────┘      └─────────────────┘      └─────────────────┘│
└───────────│────────────────────────────────────────────────────────────┘
            │ analyze request
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           MERLT API                                     │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     EXPERT ROUTER                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │  │
│  │  │ Literal  │  │ Systemic │  │Principles│  │Precedent │       │  │
│  │  │ Expert   │  │  Expert  │  │  Expert  │  │  Expert  │       │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │  │
│  │       │              │              │              │            │  │
│  │       └──────────────┴──────────────┴──────────────┘            │  │
│  │                              │                                   │  │
│  │                    ┌─────────┴─────────┐                        │  │
│  │                    │  GATING NETWORK   │                        │  │
│  │                    │  (weight combine) │                        │  │
│  │                    └─────────┬─────────┘                        │  │
│  │                              │                                   │  │
│  │                    ┌─────────┴─────────┐                        │  │
│  │                    │    SYNTHESIZER    │                        │  │
│  │                    │  (final response) │                        │  │
│  │                    └───────────────────┘                        │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│           │                    │                    │                  │
│           ▼                    ▼                    ▼                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │    Qdrant       │  │   FalkorDB      │  │   PostgreSQL    │       │
│  │  (embeddings)   │  │   (graph)       │  │  (RLCF data)    │       │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
            │ response + trace_id
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           RLCF LOOP                                     │
│                                                                         │
│    response ──▶ USER ──▶ feedback ──▶ RLCF ──▶ policy update          │
│                   │                      │                              │
│                   │                      ▼                              │
│                   │            ┌─────────────────┐                     │
│                   │            │ authority_score │                     │
│                   │            │     update      │                     │
│                   │            └─────────────────┘                     │
│                   │                      │                              │
│                   ▼                      ▼                              │
│           ┌─────────────────────────────────────────┐                  │
│           │        DIRITTO VIVENTE ALGORITMICO      │                  │
│           │   (community shapes interpretation)     │                  │
│           └─────────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

#### Mind Map: Connessioni Trasversali

```
                    CONNESSIONI EMERGENTI
                    ═════════════════════

    EHRLICH (1913)                          VECTOR SPACE (2024)
    "Living Law"                            "Semantic Distance"
         │                                        │
         │                                        │
         └───────────┐              ┌─────────────┘
                     │              │
                     ▼              ▼
              ┌─────────────────────────────┐
              │                             │
              │    SIGNIFICATO PROPRIO      │
              │    ══════════════════       │
              │                             │
              │    Shortest path =          │
              │    Diritto vivente nel      │
              │    momento dell'uso         │
              │                             │
              └─────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   Il "significato proprio" non è statico,      │
    │   ma emerge dall'uso comunitario (Ehrlich)     │
    │   e può essere misurato come distanza          │
    │   vettoriale dal cluster tecnico-giuridico     │
    │                                                 │
    └─────────────────────────────────────────────────┘

    IRNERIUS (1100)                         ALIS (2026)
    "Lucerna Juris"                         "Artificial Legal Intelligence"
         │                                        │
         │                                        │
         └───────────┐              ┌─────────────┘
                     │              │
                     ▼              ▼
              ┌─────────────────────────────┐
              │                             │
              │      NUOVA SCUOLA           │
              │      ═══════════            │
              │                             │
              │    Sistematizzazione del    │
              │    corpus giuridico con     │
              │    metodologia innovativa   │
              │                             │
              └─────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   Bologna 1100: glosse marginali sul CJC       │
    │   ALIS 2026: expert annotations sul corpus     │
    │   legislativo italiano                         │
    │                                                 │
    │   Entrambe: creare diritto comune attraverso   │
    │   annotazione sistematica                      │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

---

#### Sintesi Fase 2: Pattern Emergenti

1. **Pattern Stratificazione Verticale**
   - Layer 0 (Data) → Layer 1 (ML Core) → Layer 2 (API) → Layer 3 (UI)
   - Ogni layer aggiunge valore e astrazione

2. **Pattern Flusso Sequenziale**
   - Query → Expert cascade (gerarchia Art. 12) → Synthesis → Response → Feedback → Learning

3. **Pattern Boundary Open/Proprietary**
   - Metodologia e infrastruttura: open
   - Dati, pesi, authority: proprietari

4. **Pattern Parallelo Storico**
   - Glossatori:ALIS = CJC:Corpus IT = Glosse:Annotations = Scuola:Community

5. **Pattern Diritto Vivente Computazionale**
   - Ehrlich + Vector Space = significato proprio come shortest path dinamico

---

### FASE 3: SCAMPER Method

*Raffinamento sistematico dell'architettura attraverso 7 lenti creative*

#### S - SUBSTITUTE (Sostituire)

**Cosa possiamo sostituire per migliorare?**

| Componente Attuale | Sostituzione Proposta | Beneficio |
|--------------------|----------------------|-----------|
| FalkorDB (Redis-based) | Neo4j Community | Cypher più maturo, community più ampia |
| Qdrant | Milvus/Weaviate | Alternativa se scaling issues |
| Express backend | FastAPI unificato | Single Python stack |
| 4 Expert separati | Expert modularizzati con hot-swap | Runtime expert switching |
| LLM provider singolo | Multi-provider con fallback | Resilienza, costo ottimizzato |
| Trained weights fissi | Fine-tuned LoRA adapters | Aggiornamento incrementale |

**Decisione architetturale:**
- ✅ MANTENERE FalkorDB (già integrato, Cypher compatibile)
- ⚠️ VALUTARE Express → FastAPI per uniformità stack
- ✅ IMPLEMENTARE multi-provider LLM con fallback

---

#### C - COMBINE (Combinare)

**Cosa possiamo unire per sinergie?**

| Elementi da Combinare | Risultato | Impatto |
|----------------------|-----------|---------|
| visualex-api + merlt/api | Unified Legal API | Single point of entry |
| LiteralExpert + NER | Literal+NER Expert | Identificazione entità contestuale |
| Knowledge Graph + Vector DB | Hybrid Search | Graph-enhanced RAG |
| RLCF + Authority | Unified Feedback Loop | Semplificazione policy |
| visualex-merlt + platform | Plugin nativamente integrato | Meno boundary crossing |

**Architettura Combinata Proposta:**

```
PRIMA (5 componenti separati):
  merlt ─── visualex-api ─── visualex-platform ─── visualex-merlt ─── merlt-models

DOPO (3 componenti logici):
  ┌─────────────────────────────────────────────────────────────────┐
  │                      ALIS-CORE                                   │
  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
  │  │   alis-ml        │  │   alis-api       │  │  alis-web    │  │
  │  │  (experts+rlcf)  │  │  (unified API)   │  │  (platform)  │  │
  │  └──────────────────┘  └──────────────────┘  └──────────────┘  │
  │                                                                 │
  │  + alis-models (separato, proprietario)                        │
  └─────────────────────────────────────────────────────────────────┘
```

---

#### A - ADAPT (Adattare)

**Cosa possiamo adattare da altri domini?**

| Fonte | Concetto | Adattamento ALIS |
|-------|----------|------------------|
| **Medicina - Consenso clinico** | Expert panels con voting pesato | GatingNetwork con confidence-weighted voting |
| **Wikipedia - Citation needed** | Flag per affermazioni non verificate | Uncertainty markers in responses |
| **Stack Overflow - Reputation** | Karma progressivo | Dynamic Authority refinement |
| **Legal Tech - Document Assembly** | Template con placeholder | Response templates con slots giuridici |
| **Academic Peer Review** | Double-blind review | Devil's Advocate pillar |
| **Blockchain - Audit trail** | Immutable transaction log | rlcf_traces come audit trail |

**Adattamento prioritario:**
- ✅ Wikipedia "citation needed" → Ogni affermazione expert deve avere fonte URN
- ✅ Stack Overflow reputation → Authority Score con decay temporale
- ✅ Academic peer review → Devil's Advocate come reviewer obbligatorio

---

#### M - MODIFY/MAGNIFY/MINIFY (Modificare)

**Cosa possiamo enfatizzare o ridurre?**

| Aspetto | Azione | Motivazione |
|---------|--------|-------------|
| **Tracciabilità** | MAGNIFY | Core thesis value - ogni output spiegabile |
| **UI complexity** | MINIFY | Focus su funzionalità core, non feature creep |
| **Expert autonomy** | MAGNIFY | Ogni expert deve poter operare standalone |
| **Database coupling** | MINIFY | Astrazioni per swap database |
| **Documentation** | MAGNIFY | Tesi-quality, every decision documented |
| **Real-time features** | MINIFY | Batch processing OK per uso professionale |
| **RLCF feedback granularity** | MAGNIFY | Feedback su singoli statement, non solo response |

**Priorità di modifica:**

```
HIGH MAGNIFY                          HIGH MINIFY
═══════════                           ═══════════

Tracciabilità ████████████            UI complexity ████░░░░░░
Documentation ████████████            Real-time ████░░░░░░░░
Expert autonomy ██████████            Database coupling ███░░░░░
Feedback granularity █████            Feature creep ██░░░░░░░░
```

---

#### P - PUT TO OTHER USES (Altri usi)

**Come possiamo riutilizzare componenti per altri scopi?**

| Componente | Uso Attuale | Altri Usi Possibili |
|------------|-------------|---------------------|
| **Expert Framework** | Interpretazione Art. 12 | Qualsiasi multi-expert domain (medical, financial) |
| **RLCF** | Legal feedback | Academic grading, peer review systems |
| **Knowledge Graph** | Norme italiane | EU law, international treaties |
| **Authority Scoring** | User competence | Document reliability scoring |
| **GatingNetwork** | Expert routing | Any ensemble decision making |
| **visualex-api scrapers** | Italian legal sources | EU legal scraping |

**Riuso strategico per tesi:**
- ✅ MERL-T come framework generale → metodologia esportabile
- ✅ RLCF come paradigma → paper su "Community-driven AI alignment"
- ✅ Knowledge Graph schema → ontologia giuridica italiana pubblicabile

---

#### E - ELIMINATE (Eliminare)

**Cosa possiamo rimuovere senza perdere valore?**

| Candidato | Eliminare? | Motivazione |
|-----------|------------|-------------|
| Neural Gating (PyTorch) | ⚠️ SIMPLIFY | Rule-based routing sufficient for v1 |
| Multi-language UI | ✅ ELIMINATE | Only Italian for thesis scope |
| Complex plugin system | ⚠️ SIMPLIFY | Direct integration, less abstraction |
| OAuth providers | ✅ ELIMINATE | Simple JWT sufficient |
| Real-time notifications | ✅ ELIMINATE | Polling OK for ~20 users |
| Kubernetes deployment | ✅ ELIMINATE | Docker Compose sufficient |
| Microservices full-blown | ⚠️ MERGE | Modular monolith better for team size |

**Architettura Semplificata:**

```
ELIMINATE                              KEEP ESSENTIAL
═════════                              ═══════════════

❌ Neural Gating complex               ✅ Rule-based expert routing
❌ Multi-language                      ✅ Italian only
❌ OAuth providers                     ✅ JWT auth
❌ Real-time WebSocket                 ✅ REST polling
❌ Kubernetes                          ✅ Docker Compose
❌ Microservices overhead              ✅ Modular monolith
```

---

#### R - REVERSE/REARRANGE (Invertire/Riorganizzare)

**Cosa possiamo riordinare per migliorare il flusso?**

| Processo Attuale | Riorganizzazione | Beneficio |
|------------------|------------------|-----------|
| Expert parallel → synthesize | Expert sequential (Art. 12) → early exit | Rispetto gerarchia canonica |
| Feedback post-response | Feedback inline (durante lettura) | Granularità maggiore |
| Model load at query | Model preload at startup | Latency reduction |
| Scraping on-demand | Scraping batch + cache | Reliability, compliance |
| Documentation post-facto | Documentation-driven development | Tesi-ready from start |

**Flusso Riorganizzato:**

```
ATTUALE:
Query → [All Experts in parallel] → Gating → Synthesis → Response → Feedback

PROPOSTA (Art. 12 compliant):
Query → LiteralExpert → sufficient?
            │                 ├─ YES → Response → Inline Feedback
            │                 └─ NO ↓
            └─────────── SystemicExpert → sufficient?
                                    ├─ YES → Response + Literal context
                                    └─ NO ↓
                              PrinciplesExpert → sufficient?
                                          ├─ YES → Response + context
                                          └─ NO ↓
                                    PrecedentExpert → Final Response
                                                            ↓
                                                     Inline Feedback
```

---

#### Sintesi Fase 3: Raccomandazioni SCAMPER

**Alta Priorità (Implementare):**

1. **COMBINE:** Unificare visualex-api + merlt/api in single API layer
2. **ELIMINATE:** Rimuovere complessità non necessaria (multi-lang, OAuth, k8s)
3. **REVERSE:** Implementare expert sequenziale per rispettare Art. 12
4. **MAGNIFY:** Tracciabilità e documentazione come first-class citizens

**Media Priorità (Pianificare):**

5. **ADAPT:** Authority scoring con decay temporale (Stack Overflow model)
6. **MODIFY:** Feedback granulare per statement, non solo response
7. **SUBSTITUTE:** Valutare multi-provider LLM con fallback

**Bassa Priorità (Considerare):**

8. **PUT TO OTHER USES:** Packaging MERL-T come framework generico
9. **COMBINE:** Hybrid search (graph + vector) per retrieval

---

#### SCAMPER Decision Matrix

```
                        IMPACT
                   Low    Med    High
              ┌─────────────────────────┐
         Low  │   P      A      R      │
   EFFORT     │                        │
         Med  │   M      C      S      │
              │                        │
         High │   -      E      -      │
              └─────────────────────────┘

QUICK WINS (High Impact, Low Effort):
• R - Sequential expert flow
• A - Citation-required pattern

BIG BETS (High Impact, High Effort):
• E - Architecture simplification

FILL-INS (Low Impact, Low Effort):
• P - Document reuse potential
• M - Minor modifications

STRATEGIC (High Impact, Med Effort):
• S - Multi-provider LLM
• C - API unification
```

---

### FASE 4: Decision Tree Mapping

*Roadmap implementativa con decision points e milestone*

#### Albero Decisionale Principale

```
                            ┌─────────────────────────────────────┐
                            │          ALIS REFACTORING           │
                            │         Decision Tree 2026          │
                            └────────────────┬────────────────────┘
                                             │
                            ┌────────────────┴────────────────┐
                            │                                 │
                            ▼                                 ▼
                ┌───────────────────────┐       ┌───────────────────────┐
                │  TRACK A: ACADEMIC    │       │  TRACK B: PRODUCTION  │
                │  (Tesi + Papers)      │       │  (ALIS Association)   │
                └───────────┬───────────┘       └───────────┬───────────┘
                            │                               │
                            ▼                               ▼
        ┌───────────────────────────────┐   ┌───────────────────────────────┐
        │ A1: Documentation-First       │   │ B1: Code Refactoring          │
        │     • Methodology papers      │   │     • Monolith modularization │
        │     • Architecture docs       │   │     • API unification         │
        │     • Tesi chapters           │   │     • Expert sequentialization│
        └───────────┬───────────────────┘   └───────────┬───────────────────┘
                    │                                   │
                    ▼                                   ▼
        ┌───────────────────────────────┐   ┌───────────────────────────────┐
        │ A2: Open-Source Prep          │   │ B2: Feature Completion        │
        │     • Code cleanup            │   │     • RLCF full implementation│
        │     • License selection       │   │     • UI polish               │
        │     • README/CONTRIBUTING     │   │     • Integration tests       │
        └───────────┬───────────────────┘   └───────────┬───────────────────┘
                    │                                   │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      MILESTONE: MVP THESIS    │
                    │                               │
                    │  • Working demo for committee │
                    │  • Academic papers submitted  │
                    │  • Association deployment     │
                    └───────────────────────────────┘
```

---

#### Decision Point 1: Architettura Monorepo

```
                    ┌─────────────────────────────────┐
                    │  DP1: MONOREPO STRUCTURE        │
                    │  "Come organizzare i 5 repo?"   │
                    └────────────────┬────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│ OPTION A:         │    │ OPTION B:         │    │ OPTION C:         │
│ Keep 5 separate   │    │ Merge to 3 repos  │    │ True monorepo     │
│                   │    │                   │    │                   │
│ • merlt           │    │ • alis-ml         │    │ • alis-core/      │
│ • merlt-models    │    │ • alis-api        │    │   ├── packages/   │
│ • visualex-api    │    │ • alis-web        │    │   │   ├── ml/     │
│ • visualex-platform│   │ + alis-models     │    │   │   ├── api/    │
│ • visualex-merlt  │    │   (separate)      │    │   │   └── web/    │
│                   │    │                   │    │   └── models/     │
├───────────────────┤    ├───────────────────┤    ├───────────────────┤
│ PRO:              │    │ PRO:              │    │ PRO:              │
│ • No migration    │    │ • Logical grouping│    │ • Single version  │
│ • Clear boundaries│    │ • Easier deploys  │    │ • Atomic commits  │
│                   │    │ • Moderate effort │    │ • Shared tooling  │
├───────────────────┤    ├───────────────────┤    ├───────────────────┤
│ CON:              │    │ CON:              │    │ CON:              │
│ • Version sync    │    │ • Some migration  │    │ • Major migration │
│ • Cross-repo deps │    │ • Hybrid state    │    │ • Complex CI/CD   │
└───────────────────┘    └───────────────────┘    └───────────────────┘
        │                            │                            │
        │                            ▼                            │
        │               ╔═══════════════════════╗                │
        │               ║ RECOMMENDED: OPTION B ║                │
        │               ║ "Modular Consolidation"║                │
        │               ╚═══════════════════════╝                │
        │                            │                            │
        └────────────────────────────┴────────────────────────────┘
```

**Rationale Option B:**
- Bilancia semplicità e separazione di concern
- `alis-ml`: tutto il ML (experts, rlcf, pipeline)
- `alis-api`: unified FastAPI (legal scraping + ML API)
- `alis-web`: platform + plugin integrato
- `alis-models`: separato per IP protection

---

#### Decision Point 2: Expert Execution Model

```
                    ┌─────────────────────────────────┐
                    │  DP2: EXPERT EXECUTION          │
                    │  "Parallel vs Sequential?"      │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ PARALLEL        │    │ SEQUENTIAL      │    │ HYBRID          │
    │ (current)       │    │ (Art. 12)       │    │                 │
    │                 │    │                 │    │                 │
    │ All experts     │    │ Literal first   │    │ Literal+Systemic│
    │ run together    │    │ then cascade    │    │ parallel, then  │
    │ → Gating        │    │ if insufficient │    │ Principles+Prec │
    │ → Synthesis     │    │                 │    │ if needed       │
    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤
    │ PRO:            │    │ PRO:            │    │ PRO:            │
    │ • Fast response │    │ • Art.12 compliant│  │ • Balanced      │
    │ • Simple impl   │    │ • Explainable   │    │ • Moderate speed│
    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤
    │ CON:            │    │ CON:            │    │ CON:            │
    │ • Not Art.12    │    │ • Slower        │    │ • Complex logic │
    │ • Academic issue│    │ • Early exit?   │    │ • Harder explain│
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                      │                      │
              │                      ▼                      │
              │         ╔═══════════════════════╗          │
              │         ║ RECOMMENDED: SEQUENTIAL║          │
              │         ║ "Dottrina Compliant"  ║          │
              │         ╚═══════════════════════╝          │
              │                      │                      │
              └──────────────────────┴──────────────────────┘
```

**Rationale Sequential:**
- Accademicamente difendibile (gerarchia Art. 12)
- Tracciabilità migliore (quale expert ha risposto)
- Early exit se Literal sufficiente = risparmio compute
- Thesis-ready: ogni decisione giustificabile

---

#### Decision Point 3: Distribution Strategy

```
                    ┌─────────────────────────────────┐
                    │  DP3: OPEN SOURCE STRATEGY      │
                    │  "Cosa rendere pubblico?"       │
                    └────────────────┬────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ FULL OPEN       │       │ FRAMEWORK OPEN  │       │ PAPERS ONLY     │
│                 │       │                 │       │                 │
│ Everything      │       │ Architecture    │       │ Academic papers │
│ except models   │       │ + base classes  │       │ describing      │
│                 │       │ + docs          │       │ methodology     │
│                 │       │ + empty weights │       │                 │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ PRO:            │       │ PRO:            │       │ PRO:            │
│ • Max visibility│       │ • Reproducible  │       │ • Full IP prot  │
│ • Community     │       │ • Balanced IP   │       │ • Academic focus│
│ • Thesis impact │       │ • Thesis example│       │                 │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ CON:            │       │ CON:            │       │ CON:            │
│ • IP concerns   │       │ • Partial value │       │ • No adoption   │
│ • Competitor use│       │ • Some effort   │       │ • Limited impact│
└─────────────────┘       └─────────────────┘       └─────────────────┘
         │                           │                           │
         │                           ▼                           │
         │              ╔═══════════════════════╗               │
         │              ║ RECOMMENDED: FRAMEWORK ║               │
         │              ║ "Academic Reproducibility"║            │
         │              ╚═══════════════════════╝               │
         │                           │                           │
         └───────────────────────────┴───────────────────────────┘
```

**Open Source Manifest:**

```
OPEN (MIT/Apache 2.0)                 PROPRIETARY (ALIS Association)
═════════════════════                 ═════════════════════════════

📄 /papers/*                          🧠 /alis-models/*
   • MERL-T methodology                  • Trained weights
   • RLCF framework paper                • Fine-tuned LoRA
   • ALIS system paper

📐 /docs/*                            📊 /data/*
   • Architecture                        • FalkorDB populated graph
   • API specifications                  • Qdrant collections
   • Deployment guides

🔧 /alis-ml/experts/base_*.py        🔧 /alis-ml/experts/trained_*.py
   • Abstract classes                    • Domain-specific implementations
   • Interface definitions               • Legal domain knowledge

🔧 /alis-api/                         👤 /alis-api/rlcf/policy_checkpoints/
   • Generic endpoints                   • Authority scores
   • Scraper interfaces                  • Training sessions

🧪 /tests/fixtures/                   🔐 /config/production/
   • Anonymized examples                 • API keys, secrets
   • Mock data                           • Member credentials
```

---

#### Implementation Roadmap

```
                              2026 TIMELINE
    ════════════════════════════════════════════════════════════════

    JAN         FEB         MAR         APR         MAY         JUN
    │           │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼           ▼

    ┌─────────────────────────────────────────────────────────────┐
    │                    PHASE 1: FOUNDATION                       │
    │                    (Jan-Feb 2026)                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Week 1-2: Documentation Audit                              │
    │  ├── Inventario docs esistenti                              │
    │  ├── Gap analysis vs tesi requirements                      │
    │  └── Template standardizzati                                │
    │                                                              │
    │  Week 3-4: Architecture Decision Records                    │
    │  ├── ADR-001: Monorepo strategy                            │
    │  ├── ADR-002: Expert execution model                       │
    │  └── ADR-003: Distribution strategy                        │
    │                                                              │
    │  Week 5-6: Repository Restructure                           │
    │  ├── Create alis-ml, alis-api, alis-web structure          │
    │  ├── Move code with git history preservation               │
    │  └── Update import paths                                    │
    │                                                              │
    │  Week 7-8: CI/CD Setup                                      │
    │  ├── GitHub Actions for each component                      │
    │  ├── Docker Compose update                                  │
    │  └── Integration test suite                                 │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    PHASE 2: CORE REFACTORING                │
    │                    (Mar-Apr 2026)                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Sprint 1: Expert Pipeline                                  │
    │  ├── Implement sequential execution                         │
    │  ├── Add sufficiency scoring                                │
    │  ├── Trace generation per expert                           │
    │  └── Unit tests for each expert                            │
    │                                                              │
    │  Sprint 2: RLCF Completion                                  │
    │  ├── Authority scoring finalization                         │
    │  ├── Feedback granularity (per-statement)                  │
    │  ├── Devil's Advocate integration                          │
    │  └── Policy checkpoint management                          │
    │                                                              │
    │  Sprint 3: API Unification                                  │
    │  ├── Merge visualex-api + merlt endpoints                  │
    │  ├── OpenAPI spec generation                               │
    │  ├── Rate limiting & auth                                  │
    │  └── API versioning strategy                               │
    │                                                              │
    │  Sprint 4: Integration & Testing                            │
    │  ├── End-to-end test scenarios                             │
    │  ├── Performance benchmarks                                │
    │  ├── Security audit (OWASP)                                │
    │  └── Documentation sync                                    │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    PHASE 3: THESIS PREP                     │
    │                    (May-Jun 2026)                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Academic Deliverables:                                     │
    │  ├── Chapter drafts alignment with implementation          │
    │  ├── Methodology validation with working code              │
    │  ├── Performance metrics & evaluation                      │
    │  └── Committee demo preparation                            │
    │                                                              │
    │  Open Source Release:                                       │
    │  ├── Code cleanup & comments                               │
    │  ├── License files (MIT/Apache)                            │
    │  ├── CONTRIBUTING.md                                       │
    │  └── GitHub release v1.0                                   │
    │                                                              │
    │  Association Deployment:                                    │
    │  ├── Production environment setup                          │
    │  ├── User onboarding (20 members)                          │
    │  ├── Feedback collection system                            │
    │  └── Monitoring & alerting                                 │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
```

---

#### Milestone Definitions

| Milestone | Criteria | Date |
|-----------|----------|------|
| **M1: Architecture Decided** | ADRs approved, repo restructured | End Feb 2026 |
| **M2: Core Refactored** | Sequential experts, RLCF complete | End Apr 2026 |
| **M3: Open Source Ready** | Public repo, docs, license | Mid May 2026 |
| **M4: Thesis Demo** | Working system for committee | End May 2026 |
| **M5: Association Live** | 20 users active, feedback flowing | End Jun 2026 |

---

#### Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Thesis deadline pressure | HIGH | HIGH | Phase 1 prioritizes docs |
| Code migration breaks | MEDIUM | MEDIUM | Git history preservation, tests |
| Community adoption slow | LOW | LOW | Association early adopters |
| LLM provider changes | MEDIUM | MEDIUM | Multi-provider abstraction |
| IP concerns from association | LOW | HIGH | Clear license boundaries |

---

#### Sintesi Fase 4: Action Items

**Immediate (This Week):**
1. Create ADR template and draft ADR-001 (Monorepo)
2. Inventory all existing documentation
3. Define thesis chapter ↔ code mapping

**Short Term (This Month):**
4. Finalize Option B repo structure
5. Begin sequential expert implementation
6. Set up GitHub Actions for new structure

**Medium Term (This Quarter):**
7. Complete RLCF implementation
8. Unify API layer
9. Prepare open-source release

---

## Session Conclusion

### Key Decisions

1. **Philosophical Foundation:** AI come processo (mai agente), tracciabilità totale
2. **Historical Parallel:** Nuova scuola dei sapientes del diritto algoritmico
3. **Architecture:** 3-repo consolidation (alis-ml, alis-api, alis-web)
4. **Expert Model:** Sequential execution per Art. 12 compliance
5. **Distribution:** Framework open, models proprietary

### Next Steps

1. **Proceed to BMM Phase 2:** Research workflow for deeper technical validation
2. **Create PRD:** Formalize requirements from brainstorming insights
3. **Architecture Document:** Detailed technical specifications
4. **Sprint Planning:** Break down into actionable stories

---

**Session Status:** ✅ COMPLETED
**Duration:** Progressive Flow (4 techniques)
**Output:** 5 architectural principles, 3 major decisions, 6-month roadmap
