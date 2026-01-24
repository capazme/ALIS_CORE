---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/architecture.md
  - docs/project-documentation/index.md
  - docs/project-documentation/00-project-overview.md
  - docs/project-documentation/01-architecture.md
  - docs/project-documentation/02-merlt-experts.md
  - docs/project-documentation/03-rlcf.md
  - Legacy/VisuaLexAPI/frontend/README-UX.md
  - Legacy/VisuaLexAPI/CLAUDE.md
  - Legacy/VisuaLexAPI/frontend/src/utils/citationParser.ts
  - Legacy/VisuaLexAPI/frontend/src/utils/citationMatcher.ts
  - Legacy/MERL-T_alpha/merlt/rlcf/ner_rlcf_integration.py
  - papers/markdown/DA GP - RLCF.md
workflowType: 'ux-design'
project_name: 'ALIS_CORE'
user_name: 'Gpuzio'
date: '2026-01-24'
lastStep: 'step-06-validation-handoff'
---

# UX Design Specification ALIS_CORE

**Author:** Gpuzio
**Date:** 2026-01-24

---

## Executive Summary

### Project Vision

ALIS è una piattaforma di interpretazione giuridica computazionale che implementa i canoni ermeneutici dell'Art. 12 Preleggi come pipeline AI. La proposta di valore core è la **tracciabilità completa** del ragionamento giuridico: ogni affermazione è riconducibile a Expert specifico → Fonte → Reasoning chain.

**Unique Value:** "Posso usare questa traccia di ragionamento in un atto legale."

---

### Conceptual Framework: IDE per Giuristi

ALIS adotta il paradigma dell'**Integrated Development Environment (IDE)** come metafora guida per l'esperienza utente. Come VS Code o IntelliJ trasformano la scrittura di codice in un'esperienza assistita e produttiva, ALIS trasforma il lavoro giuridico.

#### Mapping IDE → Legal IDE

| IDE Feature | Legal IDE Equivalent | Implementazione ALIS |
|-------------|---------------------|----------------------|
| **Code Editor** | Norm Viewer | VisuaLex core, Study Mode |
| **Syntax Highlighting** | Citation Highlighting | NER + linking automatico |
| **IntelliSense/Autocomplete** | AI Suggestions | MERL-T hints, norme correlate |
| **Linting/Error Detection** | Consistency Check | Conflitti normativi, abrogazioni |
| **Debugging** | Trace Reasoning | Expert Accordion, Devil's Advocate |
| **Terminal/Console** | Query Interface | Barra ricerca → MERL-T pipeline |
| **Workspace/Project** | Dossier | Raccolta norme per caso |
| **Version Control** | Norm Versioning | Navigazione temporale vigenze |
| **Git Blame** | Legislative Intent | Storico modifiche, ratio legis |
| **Diff View** | Norm Comparison | Confronto versioni articolo |
| **Split View** | Side-by-Side Analysis | Confronto norme correlate |
| **Snippets** | Template Clausole | Clausole tipo, formule ricorrenti |
| **Extensions** | Expert Modules | LiteralExpert, PrecedentExpert, etc. |
| **Breakpoints** | Annotation Points | Punti di analisi nel reasoning |
| **Find All References** | Cross-Reference Search | "Chi cita questo articolo?" |
| **Go to Definition** | Go to Source | Click su citazione → norma originale |
| **Peek Definition** | Hover Preview | Tooltip su citazione |
| **Problems Panel** | Issues Panel | Conflitti, abrogazioni, modifiche |
| **Output Panel** | Expert Trace | Log del reasoning MERL-T |
| **Settings/Preferences** | Profile Selector | 4 profili modalità |

#### Paradigmi UX Derivati

**1. Command Palette (Ctrl+Shift+P)**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  > Cerca comando...                                                     │
│  ─────────────────────────────────────────────────────────────────────  │
│  📖 Apri norma...                                                       │
│  🔍 Analizza con MERL-T                                                │
│  📁 Aggiungi a Dossier...                                              │
│  ⏱️ Mostra versione storica...                                         │
│  📊 Confronta con...                                                    │
│  😈 Mostra interpretazioni alternative                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**2. Minimap / Document Outline**
```
┌────────────────────┐
│  STRUTTURA NORMA   │
│  ─────────────────  │
│  ▸ Capo I          │
│    Art. 1 ●        │  ← Posizione corrente
│    Art. 2          │
│    Art. 3 ⚠️       │  ← Modificato di recente
│  ▸ Capo II         │
│    Art. 4          │
│    Art. 5 🔗       │  ← Ha citazioni rilevate
└────────────────────┘
```

**3. Problems Panel / Issues**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  PROBLEMI (3)                                               [Filtra ▼] │
│  ─────────────────────────────────────────────────────────────────────  │
│  ⚠️ Art. 5 - Modificato da L. 123/2024 (in vigore dal 01/03/2024)     │
│  ⚠️ Art. 12 comma 3 - Abrogato da D.Lgs. 45/2023                       │
│  ℹ️ Art. 8 - Interpretazione controversa (vedi Devil's Advocate)       │
└─────────────────────────────────────────────────────────────────────────┘
```

**4. Peek Definition (Hover + F12)**
```
Hovering su "art. 2043 c.c.":

┌────────────────────────────────────────────────────────────┐
│  Art. 2043 - Risarcimento per fatto illecito              │
│  ──────────────────────────────────────────────────────── │
│  Qualunque fatto doloso o colposo che cagiona ad altri    │
│  un danno ingiusto, obbliga colui che ha commesso il      │
│  fatto a risarcire il danno.                              │
│  ──────────────────────────────────────────────────────── │
│  [Apri] [Peek References] [Aggiungi a Dossier]            │
└────────────────────────────────────────────────────────────┘
```

**5. Split Editor**
```
┌───────────────────────────────┬───────────────────────────────┐
│  Art. 1453 c.c.               │  Art. 1455 c.c.               │
│  Risolubilità del contratto   │  Importanza inadempimento     │
│  ─────────────────────────────│───────────────────────────────│
│  [Testo articolo...]          │  [Testo articolo...]          │
│                               │                               │
│                               │                               │
│  🔗 Collegamento sistemico    │  🔗 Collegamento sistemico    │
│  rilevato da SystemicExpert   │  rilevato da SystemicExpert   │
└───────────────────────────────┴───────────────────────────────┘
```

**6. Git Blame → Legislative History**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  L. 15/2005    │  Qualunque fatto doloso o colposo che cagiona        │
│  (originale)   │  ad altri un danno ingiusto, obbliga colui che       │
│                │  ha commesso il fatto a risarcire il danno.          │
│  ─────────────────────────────────────────────────────────────────────  │
│  D.Lgs 28/2010 │  [nessuna modifica a questo comma]                   │
│  ─────────────────────────────────────────────────────────────────────  │
│  L. 123/2024   │  [modifica al comma 2 - non visibile qui]            │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Keyboard-First Design

Come un IDE, ALIS deve essere navigabile interamente da tastiera:

| Shortcut | Azione |
|----------|--------|
| `Ctrl+K` | Focus barra ricerca (Command Palette) |
| `Ctrl+P` | Quick Open (cerca norma per nome) |
| `Ctrl+Shift+P` | Tutti i comandi |
| `F12` | Go to Definition (apri citazione) |
| `Alt+F12` | Peek Definition (preview inline) |
| `Shift+F12` | Find All References (chi cita questa norma) |
| `Ctrl+\` | Split view |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+J` | Toggle Expert panel |
| `Ctrl+Shift+M` | Problems panel |
| `Ctrl+D` | Aggiungi a Dossier |
| `Ctrl+Shift+A` | Analizza con MERL-T |

#### Implicazioni di Design

1. **Densità informativa alta**: I giuristi, come i developer, preferiscono vedere più informazioni contemporaneamente
2. **Customizzazione layout**: Pannelli ridimensionabili, tema chiaro/scuro
3. **Keyboard shortcuts**: Produttività senza mouse
4. **Context menus ricchi**: Right-click con azioni contestuali
5. **Status bar informativa**: Profilo, sessione, ultima sync
6. **Extensions/Plugins**: Architettura aperta per Expert aggiuntivi

#### Differenze da IDE Tradizionale

| IDE Tradizionale | Legal IDE |
|-----------------|-----------|
| Codice = testo modificabile | Norma = testo immutabile (read-only) |
| Compile/Run = esecuzione | Analizza = interpretazione |
| Errors = bug | Errors = conflitti normativi |
| Debug = step-through | Debug = trace reasoning |
| Git = versioning codice | Versioning = vigenze normative |
| Tests = unit test | Validation = peer review (RLCF) |

### Target Users

| Persona | Profilo | Entry Point | Contribuzione |
|---------|---------|-------------|---------------|
| **Legal Professional** | Avvocato, praticante | Query specifica ("presupposti art. 1453?") | Media (usa, feedback occasionale) |
| **Legal Researcher** | Dottorando, accademico | Esplorazione norme correlate | Alta (contribuisce dati tesi) |
| **Association Member** | Membro ALIS | Mix di entrambi | Variabile (sceglie profilo) |
| **System Admin** | Referente tecnico | Dashboard admin | N/A |
| **API Developer** | Integratore esterno | Documentazione API | N/A |

**Caratteristiche comuni:**
- Desktop-first (studio, ufficio, ricerca)
- Competenza legale medio-alta
- Familiarità con citazioni e URN
- Sensibilità alla precisione e tracciabilità

### Key Design Challenges

#### 1. Dual Entry Point Architecture
Gli utenti arrivano sia con **domande** ("come funziona X?") sia cercando **norme specifiche** (Art. 2043 c.c.). L'UX deve supportare entrambi senza friction:
- Query naturale → MERL-T pipeline
- Ricerca norma → VisuaLex browse → opt-in per enrichment AI

#### 2. Layer Simbiontico con Controllo Utente
MERL-T non è separato da VisuaLex ma un **layer simbiontico** che si attiva all'opt-in. L'utente deve poter:
- Scegliere il livello di coinvolgimento AI
- Cambiare modalità in qualsiasi momento
- Lavorare senza interruzioni quando necessario

#### 3. Synthesizer + Expert Trace
La risposta finale viene dal **Synthesizer** (aggregazione pesata). I ragionamenti dei singoli Expert sono nel **trace** espandibile. Sfida: bilanciare sintesi leggibile vs. trasparenza metodologica.

#### 4. Latency Management (<3min first, <500ms cached)
Prima risposta può richiedere fino a 3 minuti. Sfida: mantenere engagement durante attesa con progressive loading (norma base immediata, enrichment graduale).

### Design Opportunities

#### 1. Trust Through Transparency
La tracciabilità Expert → Source → Reasoning può diventare un **differenziatore competitivo**. Nessun Legal AI mostra così chiaramente il "perché" delle risposte.

#### 2. Profile-Based Personalization
Il modello a 4 profili permette **personalizzazione senza complessità**. L'utente sceglie una volta, il sistema si adatta.

#### 3. Progressive Enrichment as Feature
La latenza può diventare **feature**: "Stiamo consultando 4 Expert..." con animazione che mostra il processo metodologico (Art. 12 compliance visiva).

#### 4. Reuse VisuaLex Patterns
L'UI VisuaLex esistente (Study Mode, Dossier, Selection Popup) è matura. Possiamo **estendere** pattern familiari invece di reinventare.

#### 5. Authority as Recognition
L'authority score può essere presentato come **riconoscimento professionale**, non gamification. "Il tuo contributo plasma l'interpretazione della community."

---

## Core UX Decisions

### Decision 1: Sistema a 4 Profili Modalità

**Rationale:** Dopo analisi Tree of Thoughts con alternative (toggle binario, slider, automatico, override), la decisione finale è un sistema a **4 profili predefiniti** che bilanciano semplicità, controllo utente, e contesto professionale legale.

#### I 4 Profili

| # | Nome | Icona | Target Use Case |
|---|------|-------|-----------------|
| **1** | **Consultazione Rapida** | ⚡ | Udienza, verifica veloce, lavoro sotto pressione |
| **2** | **Ricerca Assistita** | 📖 | Lavoro quotidiano, browse + hint |
| **3** | **Analisi Esperta** | 🔍 | Studio approfondito, preparazione atti |
| **4** | **Contributore Attivo** | 🎓 | Ricercatore, membro attivo associazione |

#### Profilo 1: ⚡ Consultazione Rapida

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PROFILO 1: ⚡ Consultazione Rapida                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  API chiamate:     Solo VisuaLex (no MERL-T)                           │
│  Latenza:          <500ms sempre                                        │
│  UI elementi:      Norma + Brocardi (se cached)                        │
│  Feedback UI:      Nascosto                                             │
│  Expert trace:     Non disponibile                                      │
│  Consent richiesto: Nessuno (no AI)                                     │
│  Use case:         "Devo verificare un articolo in 30 secondi"         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Profilo 2: 📖 Ricerca Assistita

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PROFILO 2: 📖 Ricerca Assistita                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  API chiamate:     VisuaLex + MERL-T (solo cache/hints)                │
│  Latenza:          <1s (cache hit), <30s (cache miss leggero)          │
│  UI elementi:      Norma + Brocardi + "Articoli correlati" (AI hint)   │
│  Feedback UI:      Minimale [👍👎] su hints, non intrusivo             │
│  Expert trace:     Summary only (no dettaglio Expert)                  │
│  Consent richiesto: Base (AI analysis)                                  │
│  Use case:         "Sto cercando norme, aiutami a navigare"            │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Profilo 3: 🔍 Analisi Esperta

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PROFILO 3: 🔍 Analisi Esperta                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  API chiamate:     VisuaLex + MERL-T full pipeline                     │
│  Latenza:          <3min (progressive loading)                          │
│  UI elementi:      Norma + Sintesi + Expert Accordion + Fonti          │
│  Feedback UI:      Opzionale [👍👎💬] sulla sintesi                    │
│  Expert trace:     Completo, espandibile per Expert                    │
│  Consent richiesto: Full (AI + audit trail)                            │
│  Use case:         "Ho una questione giuridica, voglio analisi citabile"│
└─────────────────────────────────────────────────────────────────────────┘
```

#### Profilo 4: 🎓 Contributore Attivo

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PROFILO 4: 🎓 Contributore Attivo                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  API chiamate:     VisuaLex + MERL-T full + RLCF granulare             │
│  Latenza:          <3min (progressive loading)                          │
│  UI elementi:      Tutto Profilo 3 + feedback inline per Expert        │
│  Feedback UI:      Granulare: rating per Expert + commenti + correzioni │
│  Expert trace:     Completo + metriche confidence + disagreement       │
│  Consent richiesto: Full + RLCF feedback + research (opzionale)        │
│  Use case:         "Voglio contribuire a migliorare il sistema"        │
│  Bonus:            Vede authority score, badge contributore            │
└─────────────────────────────────────────────────────────────────────────┘
```

#### UI: Profile Selector

```
┌─────────────────────────────────────────────────────────────────────────┐
│  VisuaLex                              Modalità: [🔍 Analisi Esperta ▼] │
│                                        ┌────────────────────────────────┤
│                                        │ ⚡ Consultazione Rapida        │
│                                        │    Zero AI, massima velocità   │
│                                        │────────────────────────────────│
│                                        │ 📖 Ricerca Assistita           │
│                                        │    Suggerimenti intelligenti   │
│                                        │────────────────────────────────│
│                                        │ 🔍 Analisi Esperta    ✓        │
│                                        │    4 Expert + trace citabile   │
│                                        │────────────────────────────────│
│                                        │ 🎓 Contributore Attivo         │
│                                        │    Feedback granulare + RLCF   │
│                                        └────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────┘
```

**Comportamento:**
- Il profilo selezionato **persiste** tra sessioni
- L'utente può cambiarlo **in qualsiasi momento** dal dropdown
- Il cambio è **immediato** (no reload pagina)
- Al primo accesso: default **📖 Ricerca Assistita** (bilancia velocità e valore)

---

### Decision 2: Progressive Loading Pattern

Per gestire la latenza <3min nei profili 3 e 4, implementiamo progressive loading a 3 livelli:

#### Livello 1 - Immediato (0-500ms)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📖 Norma Base                                                          │
│  Art. 1453 c.c. - Risolubilità del contratto                           │
│  ───────────────────────────────────────────────────────────────────── │
│  [Testo articolo visibile immediatamente da VisuaLex]                  │
│                                                                         │
│  ⏳ Analisi Esperta in corso...                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Livello 2 - Progressivo (500ms-30s)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 I 4 Expert stanno analizzando...                                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ ✓ LiteralExpert       completato                                  │ │
│  │ ⏳ SystemicExpert      in corso...                                 │ │
│  │ ○ PrinciplesExpert     in attesa                                  │ │
│  │ ○ PrecedentExpert      in attesa                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Livello 3 - Educativo (>30s)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  💡 Sapevi che...                                                       │
│  L'analisi segue i canoni dell'Art. 12 Preleggi: prima il significato │
│  letterale, poi il contesto sistematico, poi i principi, infine i     │
│  precedenti giurisprudenziali.                                         │
│                                                                         │
│  [████████░░] 80% - Ancora ~30 secondi                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Decision 3: Summary + Accordion Pattern

Per visualizzare il risultato MERL-T (profili 3 e 4):

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 ANALISI ESPERTA                                                     │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SINTESI                                                                │
│  I presupposti per la risoluzione ex art. 1453 c.c. sono:              │
│  (1) inadempimento, (2) non scarsa importanza ex art. 1455,            │
│  (3) costituzione in mora. La Cassazione (n. 12345/2020) conferma     │
│  questa interpretazione consolidata.                                    │
│                                                                         │
│  Confidence: ████████░░ 85%                                            │
│  Fonti: 3 articoli, 2 massime                                          │
│                                                                         │
│  ▸ Dettaglio Expert (click per espandere)                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  ▸ LiteralExpert: Analisi testuale art. 1453                          │
│  ▸ SystemicExpert: Collegamenti art. 1455, 1218                       │
│  ▸ PrinciplesExpert: Principio di proporzionalità                     │
│  ▸ PrecedentExpert: Cass. 12345/2020, Cass. 6789/2019                 │
│                                                                         │
│  [📋 Copia Trace] [📁 Salva in Dossier] [💬 Feedback]                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Comportamento:**
- **Default:** Sintesi visibile, trace collapsed
- **Click su Expert:** Espande il reasoning completo di quell'Expert
- **Click su fonte:** Naviga alla norma/massima in VisuaLex
- **Profilo 4:** Mostra anche `[👍👎💬]` inline per ogni Expert

---

### Decision 4: Feedback Adattivo per Profilo

| Profilo | Feedback UI | Comportamento |
|---------|-------------|---------------|
| ⚡ Consultazione | Nessuno | Zero interruzioni |
| 📖 Ricerca | `[👍][👎]` su hints | Non intrusivo, post-azione |
| 🔍 Analisi | `[👍][👎][💬]` su sintesi | Opzionale, visibile ma non richiesto |
| 🎓 Contributore | Inline per Expert + correzioni | Granulare, authority-building |

**Profilo 🎓 - UI Feedback Granulare:**

```
▾ LiteralExpert: Analisi testuale art. 1453
  ───────────────────────────────────────────────────────────────────
  Il termine "inadempimento" nell'art. 1453 c.c. indica...
  [reasoning completo]

  Fonti citate: Art. 1453 c.c., Art. 1218 c.c.
  Confidence: 92%

  [👍 12] [👎 1] [💬 Commenta] [✏️ Suggerisci correzione]
```

---

### Decision 5: Earned Opt-In per Upgrade Profilo

Il sistema suggerisce upgrade profilo basandosi su comportamento, non al primo accesso:

**Trigger per suggerimento:**
- Utente in Profilo 2 che espande trace > 5 volte → suggerisci Profilo 3
- Utente in Profilo 3 che dà feedback > 10 volte → suggerisci Profilo 4
- Utente inattivo per 2+ settimane in Profilo 4 → gentle reminder

**UI Nudge (non bloccante):**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  💡 Notiamo che usi spesso l'analisi approfondita.                     │
│     Vuoi passare a "Contributore Attivo" per dare feedback granulare?  │
│                                                                         │
│     [Sì, attiva] [Non ora] [Non mostrare più]                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Decision 6: NER via RLCF - Evoluzione Citation Parsing

**Rationale:** Il sistema rules-based esistente (`citationParser.ts`) ha confidence scoring ma raggiunge ~85% accuracy. L'obiettivo è evolvere verso un NER ML-based trainato tramite RLCF feedback, mantenendo il rules-based come fallback istantaneo.

#### Architettura Two-Tier

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CURRENT STATE                        TARGET STATE                      │
│  ─────────────────                    ────────────                      │
│                                                                         │
│  [User Text Input]                    [User Text Input]                 │
│        │                                    │                           │
│        ▼                                    ▼                           │
│  ┌──────────────┐                    ┌──────────────┐                  │
│  │ Rules-Based  │                    │ Rules-Based  │ Tier 1           │
│  │   Parser     │                    │   Parser     │ (instant)        │
│  │ conf: 0.6-0.95│                   │              │                  │
│  └──────────────┘                    └──────┬───────┘                  │
│        │                                    │                           │
│        ▼                              conf < 0.85?                      │
│  [Linked Norms]                            │ Yes                        │
│                                            ▼                            │
│                                     ┌──────────────┐                   │
│                                     │   ML NER     │ Tier 2            │
│                                     │  (SpaCy)     │ (low-conf only)   │
│                                     └──────┬───────┘                   │
│                                            │                            │
│                                            ▼                            │
│                                     [Linked Norms]                      │
│                                     + inline ✓ per                      │
│                                       Profilo 🎓                        │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Entity Types per NER Legale

| Entity | Esempio | Descrizione |
|--------|---------|-------------|
| `ART_PREFIX` | "art.", "articolo", "artt." | Prefisso articolo |
| `ART_NUM` | "1453", "12-bis", "2043" | Numero articolo |
| `ACT_TYPE_ABBR` | "c.c.", "c.p.", "d.lgs." | Tipo atto abbreviato |
| `ACT_TYPE_FULL` | "codice civile", "decreto legislativo" | Tipo atto esteso |
| `ACT_NUMBER` | "231", "81" | Numero atto |
| `ACT_YEAR` | "2001", "2008" | Anno atto |
| `COMMA` | "comma 1", "co. 2" | Riferimento comma |
| `LETTERA` | "lett. a)", "lettera b" | Riferimento lettera |

#### Copertura Norme (tutte supportate ab initio)

**Codici Principali (40+):**
- Codice Civile, Penale, Procedura Civile, Procedura Penale
- Codice della Navigazione, Codice della Strada
- Codice del Consumo, Codice Privacy, Codice Antimafia
- (lista completa da `Legacy/VisuaLexAPI/visualex/tools/map.py`)

**Atti Generici:**
- Legge (l.), Decreto Legge (d.l.), Decreto Legislativo (d.lgs.)
- Regio Decreto (r.d.), DPR, DPCM, DM
- Regolamenti UE, Direttive UE

#### Feedback Collection per Profilo

| Profilo | NER Behavior | Feedback UI |
|---------|--------------|-------------|
| ⚡ Consultazione | Solo rules-based (instant) | Nessuno |
| 📖 Ricerca | Rules + ML se low-conf | `[✓]` inline discreto |
| 🔍 Analisi | Rules + ML sempre | `[✓][✗]` post-action |
| 🎓 Contributore | Rules + ML + active learning | `[✓][✗][✏️]` inline prominente |

#### UI: Inline Confirmation (Profilo 🎓)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Nel testo dell'art. 1453 c.c. si prevede che...                       │
│                   ▲──────────▲                                          │
│                   │          │                                          │
│                   │    ┌─────┴─────┐                                   │
│                   │    │ ✓ Corretto │ ← click conferma (authority+1)   │
│                   │    │ ✗ Errato   │ ← apre correzione                │
│                   │    │ ✏️ Modifica │ ← selezione manuale              │
│                   │    └───────────┘                                   │
│                   │                                                     │
│                   └── Norma riconosciuta (conf: 92%)                   │
│                       Link: Art. 1453 Codice Civile                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Comportamento click:**
- `✓ Corretto`: Invia `confirmation` feedback → authority +1
- `✗ Errato`: Apre `CitationCorrectionCard` per correzione dettagliata
- `✏️ Modifica`: Permette selezione manuale del testo corretto

#### UI: Active Learning Prompt (Profilo 🎓, >10 norme/sessione)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🎓 Hai consultato 12 norme oggi.                                      │
│     Vuoi confermare i collegamenti riconosciuti?                       │
│                                                                         │
│     [Conferma tutti (12)] [Rivedi singolarmente] [Non ora]             │
│                                                                         │
│     I tuoi contributi migliorano l'accuratezza per tutti.              │
│     Authority attuale: ████████░░ 82%                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Authority Weighting (Formula RLCF)

Ogni feedback è pesato secondo la formula del paper RLCF:

```
A_u(t) = α·B_u + β·T_u(t) + γ·P_u(t)

dove:
  α = 0.3  (peso baseline credentials)
  β = 0.5  (peso track record)
  γ = 0.2  (peso performance recente)
  λ = 0.95 (decay factor per track record)

B_u = Baseline credentials (qualifica, anni esperienza)
T_u(t) = Track record con exponential smoothing
P_u(t) = Performance recente (ultimi N feedback)
```

**Backend già implementato** in `ner_rlcf_integration.py`:
- `_get_user_authority()` calcola A_u(t)
- `_persist_to_rlcf_db()` salva in PostgreSQL
- `_update_user_track_record()` aggiorna T_u(t)

#### Session Counter (Profilo 🎓)

```
┌───────────────────────────────────────────┐
│  Sessione: 12 norme | 8 confermate | 2 corrette
│  Authority: ████████░░ 82% (+3 oggi)     │
└───────────────────────────────────────────┘
```

Posizione: Footer della pagina o sidebar, sempre visibile in Profilo 🎓.

#### Roadmap Implementazione

| Sprint | Focus | Deliverable |
|--------|-------|-------------|
| **1** | Data Collection | UI feedback inline, persistence (già fatto backend) |
| **2** | Training Pipeline | Export SpaCy format, baseline model |
| **3** | Inference | Endpoint NER ML, two-tier routing |
| **4** | Active Learning | Prompts intelligenti, uncertainty sampling |

---

### Decision 7: Devil's Advocate System

**Rationale:** Il Devil's Advocate è il quarto pilastro del framework RLCF. Previene groupthink e silos disciplinari presentando sistematicamente interpretazioni contrarie quando emerge consenso eccessivo. Non è un "bug" ma una feature metodologica che preserva il pluralismo interpretativo.

#### Quando si Attiva

Il Devil's Advocate si attiva quando:
1. **Consenso alto** (δ < 0.2): Gli Expert concordano troppo → rischio echo chamber
2. **Confidence eccessiva** (>95%): Interpretazione "troppo sicura" per materia complessa
3. **Storico monotono**: Stessa interpretazione dominante per >N query simili

**Formula di attivazione** (dal paper RLCF):
```
N_da = min(⌈|E| · r⌉, N_max)

dove:
  |E| = numero di Expert/evaluator
  r = target ratio (10-20%)
  N_max = max assoluto (evita che i devils diventino maggioranza)
```

#### Presentazione UI per Profilo

| Profilo | Devil's Advocate Visibility |
|---------|----------------------------|
| ⚡ Consultazione | Mai (zero friction) |
| 📖 Ricerca | Mai (solo hints positivi) |
| 🔍 Analisi | Collapsato, opt-in ("Vedi interpretazioni alternative") |
| 🎓 Contributore | Espanso di default, feedback richiesto |

#### UI: Devil's Advocate Card (Profilo 🔍)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 ANALISI ESPERTA                                                     │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SINTESI (Consenso: 94%)                                               │
│  I presupposti per la risoluzione ex art. 1453 c.c. sono...            │
│  [sintesi principale]                                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  😈 Interpretazione Alternativa                          [▸]    │   │
│  │  Un orientamento minoritario sostiene che...                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  [📋 Copia Trace] [📁 Salva in Dossier] [💬 Feedback]                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Click su [▸]** espande:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  😈 Interpretazione Alternativa                                  [▾]   │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  POSIZIONE CONTRARIA                                                    │
│  Secondo un orientamento dottrinale minoritario (Tizio, 2018;          │
│  contra Cass. 12345/2020), la risoluzione ex art. 1453 non             │
│  richiede necessariamente la previa costituzione in mora quando...     │
│                                                                         │
│  Fonti: Tizio, "Sulla risoluzione contrattuale", 2018                  │
│         Cass. 9876/2015 (obiter dictum)                                │
│                                                                         │
│  ⚠️ Questa interpretazione sfida il consenso dominante.                │
│     Ha valore se il tuo caso presenta elementi atipici.                │
│                                                                         │
│  [Utile per il mio caso] [Non rilevante] [Approfondisci]               │
└─────────────────────────────────────────────────────────────────────────┘
```

#### UI: Devil's Advocate Card (Profilo 🎓)

Per il Contributore Attivo, il Devil's Advocate è sempre visibile e richiede valutazione:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  😈 DEVIL'S ADVOCATE                                    Consenso: 94%  │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Il sistema ha rilevato alto consenso. Per preservare il pluralismo   │
│  interpretativo, presentiamo una posizione contraria:                  │
│                                                                         │
│  POSIZIONE CONTRARIA                                                    │
│  [Testo dell'interpretazione alternativa...]                           │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  🎓 La tua valutazione contribuisce a calibrare il sistema:           │
│                                                                         │
│  Questa interpretazione alternativa è:                                 │
│  ○ Valida e sottorappresentata (merita più peso)                      │
│  ○ Tecnicamente corretta ma superata                                   │
│  ○ Errata o fuorviante                                                 │
│  ○ Non posso valutare (fuori dal mio ambito)                          │
│                                                                         │
│  [💬 Aggiungi commento]                                                │
│                                                                         │
│  [Invia valutazione]                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Feedback Options e Impatto

| Risposta | Impatto sul Sistema |
|----------|---------------------|
| "Valida e sottorappresentata" | ↑ Peso interpretazione alternativa |
| "Tecnicamente corretta ma superata" | Mantiene come storico, ↓ peso attuale |
| "Errata o fuorviante" | ↓ Peso, flag per review |
| "Non posso valutare" | Nessun impatto (onestà epistemica) |

**Authority weighting**: Ogni valutazione pesata per A_u(t) dell'utente.

#### Frequenza e Non-Intrusività

Per evitare fatigue:
- **Max 1 Devil's Advocate per sessione** (Profilo 🔍)
- **Max 3 per sessione** (Profilo 🎓)
- **Cooldown**: Se utente clicca "Non rilevante" 3x consecutive → pausa 1 settimana
- **Smart targeting**: Mostra su query dove l'utente ha expertise (basato su storico)

#### Icona e Naming

| Opzione | Pro | Contro |
|---------|-----|--------|
| 😈 Devil's Advocate | Riconoscibile, playful | Potrebbe sembrare negativo |
| ⚖️ Contraddittorio | Neutro, legale | Meno memorabile |
| 🔄 Interpretazione Alternativa | Descrittivo | Generico |
| 🎭 Altra Voce | Evocativo | Poco chiaro |

**Raccomandazione**: Usare **😈** nell'header con label "Interpretazione Alternativa" per bilanciare riconoscibilità e professionalità.

#### Integrazione con Expert Trace

Il Devil's Advocate può emergere da:
1. **Dissenso tra Expert**: PrecedentExpert trova giurisprudenza contraria
2. **Generazione dedicata**: Prompt specifico per contrarian view
3. **Storico RLCF**: Interpretazioni minoritarie validate dalla community

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Dettaglio Expert                                                       │
│  ─────────────────────────────────────────────────────────────────────  │
│  ▸ LiteralExpert (92%)                                                 │
│  ▸ SystemicExpert (88%)                                                │
│  ▸ PrinciplesExpert (85%)                                              │
│  ▸ PrecedentExpert (90%)                                               │
│  ─────────────────────────────────────────────────────────────────────  │
│  😈 Devil's Advocate                                                   │
│  ▸ Fonte: PrecedentExpert (minority opinion)                          │
│     Cass. 9876/2015 suggerisce interpretazione diversa...              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Interaction Patterns

### Navigation Architecture

#### Primary Navigation Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│  VisuaLex                    [🔍]          Modalità: [🔍 Analisi ▼]    │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌────────────────────────────────────────────────┐│
│  │ SIDEBAR         │  │ MAIN CONTENT                                   ││
│  │                 │  │                                                 ││
│  │ 📖 Codici       │  │  [Contenuto contestuale]                       ││
│  │   ├ Civile      │  │                                                 ││
│  │   ├ Penale      │  │                                                 ││
│  │   └ ...         │  │                                                 ││
│  │                 │  │                                                 ││
│  │ 📁 Dossier      │  │                                                 ││
│  │   ├ Caso A      │  │                                                 ││
│  │   └ Caso B      │  │                                                 ││
│  │                 │  │                                                 ││
│  │ 🕐 Recenti      │  │                                                 ││
│  │                 │  │                                                 ││
│  │ ⚙️ Impostazioni │  │                                                 ││
│  └─────────────────┘  └────────────────────────────────────────────────┘│
│  ──────────────────────────────────────────────────────────────────────  │
│  [Footer: Session stats per Profilo 🎓 | Authority | Consent status]   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Dual Entry Points

| Entry Point | Trigger | Flow |
|-------------|---------|------|
| **Query naturale** | Barra ricerca globale | Input → MERL-T → Risultato con trace |
| **Browse norma** | Sidebar / link diretto | Norma → Opt-in enrichment (se Profilo 2+) |

**Query Search Bar:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  [🔍 Cerca norma o fai una domanda...]                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  Suggerimenti:                                                          │
│    "art. 1453 c.c."           → [Browse norma]                         │
│    "presupposti risoluzione"  → [Query MERL-T]                         │
│    "l. 241/1990"              → [Browse norma]                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Micro-Interactions

#### 1. Citation Link Hover

```
Hover su "art. 1453 c.c." nel testo:

┌──────────────────────────────────┐
│  Art. 1453 - Codice Civile      │
│  Risolubilità del contratto     │
│  ─────────────────────────────  │
│  [Apri] [Aggiungi a Dossier]    │
│                                  │
│  Confidence: 92% ✓              │ ← Solo Profilo 🎓
└──────────────────────────────────┘
```

**Timing:**
- Delay hover: 300ms (evita flash accidentali)
- Fade-in: 150ms
- Fade-out on leave: 100ms

#### 2. Expert Accordion Expand

```
Click su "▸ LiteralExpert":

▾ LiteralExpert (92%)              ← Rotazione 90° dell'icona
  ──────────────────────────────────
  [Contenuto slide-down 200ms]

  Height animation: ease-out
  Scroll-into-view: se fuori viewport
```

#### 3. Feedback Button States

```
Initial:     [👍]  [👎]  [💬]      ← Outline, gray
Hover:       [👍]  [👎]  [💬]      ← Fill color hint
Selected:    [👍✓] [👎]  [💬]      ← Filled, checkmark
Submitted:   [👍✓] ──────────      ← Fade altri, toast conferma
```

**Toast Feedback:**
```
┌────────────────────────────────┐
│  ✓ Grazie! Authority +1       │  ← Slide-in right, 2s auto-dismiss
└────────────────────────────────┘
```

#### 4. Profile Switch

```
Click su dropdown profilo:

Transizione: 200ms opacity + scale
Effetto: Shimmer su elementi UI che cambiano
No page reload
```

#### 5. Progressive Loading States

| Stato | UI | Interazione |
|-------|-----|-------------|
| Loading | Skeleton + spinner | Cancellabile (X) |
| Partial | Norma + "Analisi in corso..." | Norma già navigabile |
| Complete | Full content | Tutti controlli attivi |
| Error | Toast + retry | [Riprova] button |
| Timeout (>3min) | Fallback message | [Consulta solo norma] |

### State Management (UI States)

#### Page States

```typescript
type PageState =
  | { status: 'idle' }                           // Pagina iniziale
  | { status: 'loading'; progress?: number }     // Caricamento
  | { status: 'partial'; norma: Norma }          // Norma senza enrichment
  | { status: 'enriching'; norma: Norma; experts: ExpertProgress[] }
  | { status: 'complete'; norma: Norma; analysis: MERLTResult }
  | { status: 'error'; error: Error; retry: () => void };
```

#### Expert Progress

```typescript
type ExpertProgress =
  | { expert: string; status: 'pending' }
  | { expert: string; status: 'running' }
  | { expert: string; status: 'complete'; confidence: number }
  | { expert: string; status: 'error'; message: string };
```

#### Feedback State

```typescript
type FeedbackState =
  | { status: 'none' }
  | { status: 'hovering'; target: 'up' | 'down' | 'comment' }
  | { status: 'selected'; value: 'up' | 'down' }
  | { status: 'commenting'; draft: string }
  | { status: 'submitting' }
  | { status: 'submitted'; response: FeedbackResponse };
```

### Error Handling UX

#### Error Categories

| Categoria | Messaggio | Azione |
|-----------|-----------|--------|
| **Network** | "Connessione interrotta" | [Riprova] + cache offline |
| **Timeout** | "L'analisi sta richiedendo più tempo del previsto" | [Attendi] [Solo norma] |
| **Not Found** | "Norma non trovata" | Suggerimenti alternativi |
| **Rate Limit** | "Troppe richieste, attendi X secondi" | Countdown + auto-retry |
| **Server Error** | "Errore temporaneo" | [Riprova] + ID errore |

#### Error Toast Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ⚠️ Connessione interrotta                                             │
│  ─────────────────────────────────────────────────────────────────────  │
│  Le modifiche non salvate saranno ripristinate automaticamente.        │
│                                                                         │
│  [Riprova ora]  [Lavora offline]                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Posizione:** Top-right, slide-in
**Auto-dismiss:** Solo per successi (3s), errori persistono

### Accessibility (A11y)

#### Keyboard Navigation

| Tasto | Azione |
|-------|--------|
| `Tab` | Naviga tra elementi interattivi |
| `Enter/Space` | Attiva elemento focussato |
| `Escape` | Chiudi modal/dropdown/popover |
| `Arrow ↑↓` | Naviga lista/accordion |
| `Ctrl+K` | Focus barra ricerca |
| `?` | Mostra shortcuts (quando non in input) |

#### Focus Management

```css
/* Focus visibile sempre */
:focus-visible {
  outline: 2px solid var(--color-focus);
  outline-offset: 2px;
}

/* Skip link per screen reader */
.skip-link:focus {
  position: fixed;
  top: 0;
  left: 0;
  z-index: 9999;
}
```

#### ARIA Labels

```html
<!-- Expert accordion -->
<button
  aria-expanded="false"
  aria-controls="literal-expert-content"
  aria-label="Espandi analisi LiteralExpert, confidence 92%">
  ▸ LiteralExpert (92%)
</button>

<!-- Feedback buttons -->
<button aria-label="Questo risultato è utile" aria-pressed="false">
  👍
</button>

<!-- Progress -->
<div
  role="progressbar"
  aria-valuenow="80"
  aria-valuemin="0"
  aria-valuemax="100"
  aria-label="Analisi in corso, 80% completato">
</div>
```

#### Color Contrast

| Elemento | Ratio Minimo | Note |
|----------|--------------|------|
| Testo normale | 4.5:1 | WCAG AA |
| Testo grande (18px+) | 3:1 | WCAG AA |
| Elementi UI | 3:1 | Bordi, icone |
| Focus indicator | 3:1 | Outline |

#### Screen Reader Announcements

```typescript
// Annuncia completamento analisi
announceToScreenReader(`Analisi completata.
  Confidence ${result.confidence}%.
  ${result.sources.length} fonti trovate.`);

// Annuncia feedback inviato
announceToScreenReader(`Feedback inviato. Grazie per il contributo.`);
```

### Responsive Breakpoints

| Breakpoint | Layout | Note |
|------------|--------|------|
| < 768px | Sidebar collassata, single column | Mobile (non prioritario) |
| 768px - 1024px | Sidebar mini, main expanded | Tablet |
| > 1024px | Full layout | Desktop (primario) |
| > 1440px | Max-width container, centered | Large desktop |

**Desktop-First Approach:**
- Target primario: 1280px+ (studio legale, ufficio)
- Mobile: funzionale ma non ottimizzato
- Touch: supportato ma keyboard/mouse prioritari

---

## Visual Design System

### Design Philosophy

ALIS adotta un design **professionale e sobrio** adatto all'ambiente legale:
- **Leggibilità prima di tutto**: testi lunghi, citazioni, riferimenti
- **Gerarchia chiara**: distinzione immediata tra norma, analisi, fonti
- **Credibilità**: aspetto autorevole senza essere austero
- **Efficienza**: ridurre cognitive load durante il lavoro intenso

### Color Palette

#### Primary Colors

```css
:root {
  /* Brand - Blu istituzionale */
  --color-primary-50: #eff6ff;
  --color-primary-100: #dbeafe;
  --color-primary-500: #3b82f6;
  --color-primary-600: #2563eb;  /* Primary action */
  --color-primary-700: #1d4ed8;
  --color-primary-900: #1e3a8a;

  /* Accent - Verde conferma */
  --color-success-500: #22c55e;
  --color-success-600: #16a34a;

  /* Warning */
  --color-warning-500: #f59e0b;
  --color-warning-600: #d97706;

  /* Error */
  --color-error-500: #ef4444;
  --color-error-600: #dc2626;
}
```

#### Semantic Colors

| Uso | Light Mode | Dark Mode |
|-----|------------|-----------|
| Background page | `#ffffff` | `#0f172a` (slate-900) |
| Background card | `#f8fafc` | `#1e293b` (slate-800) |
| Text primary | `#0f172a` | `#f1f5f9` |
| Text secondary | `#64748b` | `#94a3b8` |
| Border | `#e2e8f0` | `#334155` |
| Focus ring | `#3b82f6` | `#60a5fa` |

#### Expert-Specific Colors

Ogni Expert ha un colore identificativo per il trace:

| Expert | Colore | Uso |
|--------|--------|-----|
| LiteralExpert | `#3b82f6` (blue) | Analisi testuale |
| SystemicExpert | `#8b5cf6` (violet) | Connessioni sistemiche |
| PrinciplesExpert | `#f59e0b` (amber) | Principi costituzionali |
| PrecedentExpert | `#10b981` (emerald) | Giurisprudenza |
| Synthesizer | `#6366f1` (indigo) | Sintesi finale |

#### Profile Colors

| Profilo | Colore Badge | Uso |
|---------|--------------|-----|
| ⚡ Consultazione | `#64748b` (slate) | Neutro, veloce |
| 📖 Ricerca | `#3b82f6` (blue) | Standard, professionale |
| 🔍 Analisi | `#8b5cf6` (violet) | Approfondito |
| 🎓 Contributore | `#f59e0b` (amber) | Premium, riconoscimento |

### Typography

#### Font Stack

```css
:root {
  /* Headings - Serif per autorevolezza legale */
  --font-heading: 'Source Serif Pro', 'Georgia', serif;

  /* Body - Sans-serif per leggibilità */
  --font-body: 'Inter', 'system-ui', sans-serif;

  /* Code/Citations - Monospace */
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}
```

#### Type Scale

| Nome | Size | Weight | Line Height | Uso |
|------|------|--------|-------------|-----|
| `h1` | 30px | 700 | 1.2 | Titolo pagina |
| `h2` | 24px | 600 | 1.3 | Titolo sezione |
| `h3` | 20px | 600 | 1.4 | Titolo articolo |
| `h4` | 18px | 500 | 1.4 | Titolo Expert |
| `body` | 16px | 400 | 1.6 | Testo principale |
| `body-sm` | 14px | 400 | 1.5 | Metadati, caption |
| `caption` | 12px | 400 | 1.4 | Label, hint |

#### Text Styles

```css
/* Norma Title */
.norma-title {
  font-family: var(--font-heading);
  font-size: 24px;
  font-weight: 600;
  color: var(--color-text-primary);
}

/* Norma Body - Ottimizzato per lettura lunga */
.norma-body {
  font-family: var(--font-body);
  font-size: 16px;
  line-height: 1.75;
  color: var(--color-text-primary);
  max-width: 70ch; /* Larghezza ottimale lettura */
}

/* Citation Link */
.citation-link {
  font-family: var(--font-mono);
  font-size: 14px;
  color: var(--color-primary-600);
  text-decoration: underline;
  text-decoration-style: dotted;
}

/* Expert Reasoning */
.expert-reasoning {
  font-family: var(--font-body);
  font-size: 15px;
  line-height: 1.7;
  color: var(--color-text-secondary);
  padding-left: 16px;
  border-left: 3px solid var(--expert-color);
}
```

### Spacing System

Basato su scala 4px (Tailwind default):

| Token | Value | Uso |
|-------|-------|-----|
| `space-1` | 4px | Micro gap (icon + text) |
| `space-2` | 8px | Tight (form fields) |
| `space-3` | 12px | Compact (list items) |
| `space-4` | 16px | Default (card padding) |
| `space-6` | 24px | Loose (section gap) |
| `space-8` | 32px | Large (page sections) |
| `space-12` | 48px | XL (major sections) |

### Component Patterns

#### Card Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  CARD HEADER                                                      │  │
│  │  ─────────────────────────────────────────────────────────────── │  │
│  │  Titolo                                            [Action]       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  CARD BODY                                                              │
│  Contenuto principale del card                                          │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  CARD FOOTER                                                      │  │
│  │  Metadati | Azioni secondarie                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

CSS:
- border-radius: 8px
- border: 1px solid var(--color-border)
- box-shadow: 0 1px 3px rgba(0,0,0,0.1)
- padding: 16px (body), 12px (header/footer)
```

#### Button Variants

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PRIMARY        SECONDARY      GHOST          DANGER                   │
│  ┌─────────┐    ┌─────────┐   ┌─────────┐    ┌─────────┐              │
│  │ ██████ │    │ ░░░░░░ │   │         │    │ ██████ │              │
│  │ Azione │    │ Azione │   │ Azione  │    │ Elimina│              │
│  └─────────┘    └─────────┘   └─────────┘    └─────────┘              │
│                                                                         │
│  - bg-primary    - bg-gray-100  - transparent   - bg-error             │
│  - text-white    - text-gray    - text-primary  - text-white           │
│  - hover:darker  - hover:gray   - hover:bg-gray - hover:darker         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Input Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LABEL                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Placeholder text...                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  Helper text o error message                                            │
│                                                                         │
│  STATES:                                                                │
│  - Default: border-gray-300                                             │
│  - Focus: border-primary-500 + ring-2 ring-primary-200                  │
│  - Error: border-error-500 + ring-2 ring-error-200                      │
│  - Disabled: bg-gray-50 + opacity-50                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Badge Pattern

```
Confidence Badge:
┌────────────────┐
│ 92% ████████░░ │  → Verde se >80%, Giallo 60-80%, Rosso <60%
└────────────────┘

Expert Badge:
┌──────────────────┐
│ 🔍 LiteralExpert │  → Colore expert-specific
└──────────────────┘

Profile Badge:
┌──────────────────────┐
│ 🎓 Contributore      │  → Colore profilo
└──────────────────────┘
```

### Iconography

#### Icon Library

**Primaria:** Lucide Icons (open source, consistenti)
**Fallback:** Heroicons

#### Icon Sizing

| Size | px | Uso |
|------|-----|-----|
| `xs` | 12px | Inline con testo small |
| `sm` | 16px | Inline con testo body |
| `md` | 20px | Button icons |
| `lg` | 24px | Navigation, headers |
| `xl` | 32px | Feature icons |

#### Icon Usage

```
Azione     Icona
────────────────────
Cerca      🔍 Search
Salva      💾 Save
Condividi  📤 Share
Copia      📋 Copy
Espandi    ▸ ChevronRight
Comprimi   ▾ ChevronDown
Chiudi     ✕ X
Feedback+  👍 ThumbsUp
Feedback-  👎 ThumbsDown
Commento   💬 MessageCircle
Modifica   ✏️ Edit
Conferma   ✓ Check
Errore     ⚠️ AlertTriangle
Info       ℹ️ Info
```

### Animation & Motion

#### Timing Functions

```css
:root {
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-in-out: cubic-bezier(0.65, 0, 0.35, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}
```

#### Duration Scale

| Token | ms | Uso |
|-------|-----|-----|
| `fast` | 100ms | Micro-interactions (hover) |
| `normal` | 200ms | Standard transitions |
| `slow` | 300ms | Complex animations |
| `slower` | 500ms | Page transitions |

#### Motion Patterns

```css
/* Fade In */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* Slide Up (toast, modal) */
@keyframes slideUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Accordion expand */
@keyframes expandHeight {
  from { height: 0; opacity: 0; }
  to { height: var(--content-height); opacity: 1; }
}

/* Skeleton shimmer */
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
```

#### Reduce Motion

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

### Dark Mode

#### Implementation Strategy

1. **CSS Variables** per tutti i colori semantici
2. **`dark:` prefix** Tailwind per override
3. **System preference** di default, con toggle manuale
4. **Persistence** in localStorage

#### Dark Mode Palette

```css
.dark {
  --color-bg-page: #0f172a;      /* slate-900 */
  --color-bg-card: #1e293b;      /* slate-800 */
  --color-bg-elevated: #334155;  /* slate-700 */
  --color-text-primary: #f1f5f9; /* slate-100 */
  --color-text-secondary: #94a3b8; /* slate-400 */
  --color-border: #475569;       /* slate-600 */
}
```

#### Dark Mode Guidelines

- **Mai nero puro** (`#000`): usa slate-900 (`#0f172a`)
- **Ridurre contrasto**: non bianco puro, usa slate-100
- **Colori desaturati**: primari leggermente più scuri
- **Ombre ridotte**: box-shadow più sottili o eliminati
- **Test leggibilità**: verificare contrast ratio

### Figma/Design Handoff

#### Token Export

Design tokens esportati in formato compatibile:

```json
{
  "colors": {
    "primary": { "500": "#3b82f6", "600": "#2563eb" },
    "text": { "primary": "#0f172a", "secondary": "#64748b" }
  },
  "typography": {
    "heading": { "fontFamily": "Source Serif Pro", "h1": "30px" },
    "body": { "fontFamily": "Inter", "base": "16px" }
  },
  "spacing": { "1": "4px", "2": "8px", "4": "16px" },
  "radii": { "sm": "4px", "md": "8px", "lg": "12px" }
}
```

#### Component Checklist

Per ogni componente documentare:
- [ ] Stati (default, hover, focus, active, disabled)
- [ ] Varianti (size, color, style)
- [ ] Responsive behavior
- [ ] Dark mode appearance
- [ ] Animation/transitions
- [ ] A11y considerations

---

## Validation & Handoff

### Summary of Core Decisions

| # | Decision | Rationale | Impact |
|---|----------|-----------|--------|
| **1** | 4-Profile System | Bilancia semplicità e controllo utente | Architettura API, UI components |
| **2** | Progressive Loading | Gestisce latenza <3min senza friction | Frontend state management |
| **3** | Summary + Accordion | Trasparenza metodologica mantenendo leggibilità | UI trace component |
| **4** | Feedback Adattivo | Ottimizza raccolta dati senza intrusione | RLCF integration |
| **5** | Earned Opt-In | Upgrade profilo basato su comportamento | Analytics, nudge system |
| **6** | NER via RLCF | Evoluzione citation parsing → ML NER | Backend training, UI feedback |
| **7** | Devil's Advocate | Preserva pluralismo, previene groupthink | RLCF Pillar IV, UI contrarian |

### Alignment Matrix

#### Con PRD (Product Requirements)

| PRD Requirement | UX Solution | Status |
|-----------------|-------------|--------|
| FR-001: Query MERL-T | Dual entry point (search + browse) | ✅ Addressed |
| FR-002: Expert trace | Accordion pattern con confidence | ✅ Addressed |
| FR-003: Feedback collection | Profilo-specific feedback UI | ✅ Addressed |
| FR-004: Authority tracking | Session counter, badge 🎓 | ✅ Addressed |
| NFR-001: <3min latency | Progressive loading pattern | ✅ Addressed |
| NFR-002: WCAG AA | A11y section completa | ✅ Addressed |
| NFR-003: GDPR consent | Consent per profilo, opt-in | ✅ Addressed |

#### Con Architecture

| ADR | UX Implication | Status |
|-----|----------------|--------|
| ADR-001: Circuit Breaker | Error states, retry UX | ✅ Addressed |
| ADR-002: GDPR Consent | Profili 3-4 richiedono consent | ✅ Addressed |
| ADR-003: API Versioning | Transparent to UX | N/A |
| ADR-004: Audit Trail | Session persistence, history | ✅ Addressed |
| ADR-005: Warm-Start Cache | Latenza ridotta Profili 1-2 | ✅ Addressed |

### Implementation Recommendations

#### Priority Order (Sprint Planning)

| Priority | Component | Complexity | Dependencies |
|----------|-----------|------------|--------------|
| **P0** | Profile Selector | Low | localStorage, API |
| **P0** | Progressive Loading | Medium | SSE/WebSocket |
| **P1** | Expert Accordion | Low | None |
| **P1** | Feedback Buttons | Low | API endpoint |
| **P1** | Citation Link Hover | Medium | citationParser |
| **P2** | NER Inline Confirmation | Medium | NER backend |
| **P2** | Active Learning Prompts | Low | Analytics |
| **P3** | Session Counter | Low | State management |
| **P3** | Authority Badge | Low | API |

#### Component Reuse from VisuaLex

I seguenti componenti esistono già in `Legacy/VisuaLexAPI/frontend`:

| Component | Location | Adaptation Needed |
|-----------|----------|-------------------|
| `CitationCorrectionCard` | `components/features/merlt/` | Minor (add confirm) |
| `StudyModePanel` | `components/features/` | Extend for Expert |
| `SelectionPopup` | `components/features/` | Reuse as-is |
| `DossierSidebar` | `components/features/` | Reuse as-is |
| `Toast` | `components/ui/` | Reuse as-is |
| `Button` | `components/ui/` | Add variants |
| `Card` | `components/ui/` | Add states |

#### New Components Required

| Component | Complexity | Notes |
|-----------|------------|-------|
| `ProfileSelector` | Low | Dropdown + persistence |
| `ProgressiveLoader` | Medium | SSE integration |
| `ExpertAccordion` | Medium | Color-coded, expandable |
| `FeedbackBar` | Low | 3 buttons + states |
| `ConfidenceBadge` | Low | Gradient bar |
| `SessionStats` | Low | Footer component |

### Success Metrics

#### Quantitative KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first interaction | < 500ms | Analytics |
| Feedback submission rate (🎓) | > 30% | API logs |
| Profile upgrade conversion | > 10% | Analytics |
| Error recovery rate | > 90% | Error tracking |
| Accessibility score | > 90 | Lighthouse |

#### Qualitative Indicators

- [ ] Utenti capiscono la differenza tra profili senza spiegazione
- [ ] Expert trace è consultato (non ignorato)
- [ ] Feedback percepito come contributo, non fatica
- [ ] Latenza 3min tollerata grazie a progressive loading
- [ ] Citation linking percepito come accurato

### Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Latenza >3min | Medium | High | Timeout + fallback a norma base |
| Feedback fatigue | Medium | Medium | Earned opt-in, batch confirm |
| Profile confusion | Low | Medium | Onboarding tooltip |
| Accessibility gaps | Low | High | Automated testing CI |
| Dark mode contrast | Low | Low | Manual review |

### Open Questions for Implementation

1. **SSE vs WebSocket per progressive loading?**
   - Raccomandazione: SSE (più semplice, unidirezionale sufficiente)

2. **Gestione offline?**
   - Raccomandazione: Cache norma base, disable MERL-T features

3. **Animazioni custom o libreria?**
   - Raccomandazione: Framer Motion per React

4. **State management?**
   - Raccomandazione: Zustand (leggero) o React Query per server state

### Next Steps

1. **Immediate**: Review con stakeholder (PM, Dev Lead)
2. **Short-term**: Prototipo Figma per user testing
3. **Medium-term**: Sprint 1 implementation (Profile Selector, Progressive Loading)
4. **Long-term**: NER via RLCF evolution (Sprint 2-4)

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Expert** | Uno dei 4 agenti AI (Literal, Systemic, Principles, Precedent) |
| **Synthesizer** | Componente che aggrega le risposte degli Expert |
| **RLCF** | Reinforcement Learning from Community Feedback |
| **Authority** | Score utente basato su credentials + track record |
| **NER** | Named Entity Recognition (riconoscimento citazioni) |
| **URN** | Uniform Resource Name (identificativo norma) |
| **Trace** | Catena di reasoning esplicita di un Expert |

### References

- Architecture Document: `_bmad-output/planning-artifacts/architecture.md`
- PRD: `_bmad-output/planning-artifacts/prd.md`
- RLCF Paper: `papers/markdown/DA GP - RLCF.md`
- VisuaLex UX: `Legacy/VisuaLexAPI/frontend/README-UX.md`
- Citation Parser: `Legacy/VisuaLexAPI/frontend/src/utils/citationParser.ts`
- NER RLCF Backend: `Legacy/MERL-T_alpha/merlt/rlcf/ner_rlcf_integration.py`

---

**Document Status:** Complete
**Last Updated:** 2026-01-24
**Next Review:** Before Sprint 1 kickoff
