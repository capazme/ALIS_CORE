# ALIS - Artificial Legal Intelligence System

> **Sistema di Intelligenza Artificiale per l'Informatica Giuridica Italiana**

---

## Cos'è ALIS?

ALIS (Artificial Legal Intelligence System) è un ecosistema software per l'analisi automatizzata del diritto italiano. Il progetto nasce nell'ambito di una ricerca sulla **sociologia computazionale del diritto**, con l'obiettivo di creare strumenti che permettano di studiare il sistema giuridico attraverso metodi computazionali.

### Obiettivo della Ricerca

Il sistema implementa computazionalmente i **canoni ermeneutici** dell'Art. 12 delle Preleggi al Codice Civile, trasformando i metodi interpretativi tradizionali in algoritmi eseguibili:

| Canone Ermeneutico | Implementazione Computazionale |
|-------------------|-------------------------------|
| **Interpretazione letterale** | `LiteralExpert` - Analisi testuale e definizioni |
| **Interpretazione sistematica** | `SystemicExpert` - Contesto normativo e collegamenti |
| **Intenzione del legislatore** | `PrinciplesExpert` - Principi costituzionali e ratio legis |
| **Giurisprudenza applicativa** | `PrecedentExpert` - Precedenti e orientamenti |

---

## Struttura del Progetto

Il progetto è organizzato in **5 repository specializzati** + un archivio legacy:

```
ALIS_CORE/
├── visualex-api/        # Libreria Python per accesso a fonti giuridiche
├── visualex-platform/   # Piattaforma web per ricerca giuridica
├── merlt/               # Framework ML per analisi giuridica (MERL-T)
├── merlt-models/        # Modelli addestrati e configurazioni
├── visualex-merlt/      # Integrazione piattaforma + MERL-T
└── Legacy/              # Codice originale (archivio storico)
```

### Mappa Concettuale

```
┌─────────────────────────────────────────────────────────────────┐
│                         UTENTE                                   │
│                    (Giurista/Ricercatore)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VISUALEX-PLATFORM                              │
│              Interfaccia Web per Ricerca Giuridica              │
│                                                                  │
│  • Ricerca testi normativi    • Gestione dossier                │
│  • Cronologia e preferiti     • Esportazione PDF                │
│  • Annotazioni personali      • Dark mode                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│      VISUALEX-API       │   │        VISUALEX-MERLT           │
│   Accesso Fonti Dati    │   │   Plugin Sistema Esperto        │
│                         │   │                                  │
│ • Normattiva.it         │   │ • Multi-Expert Analysis         │
│ • Brocardi.it           │   │ • Knowledge Graph               │
│ • EUR-Lex               │   │ • Feedback Loop (RLCF)          │
└─────────────────────────┘   └───────────────┬─────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────────┐
                              │             MERLT                  │
                              │   Multi-Expert Legal Retrieval    │
                              │          Transformer              │
                              │                                    │
                              │  ┌─────────┐  ┌─────────┐         │
                              │  │Literal  │  │Systemic │         │
                              │  │Expert   │  │Expert   │         │
                              │  └────┬────┘  └────┬────┘         │
                              │       │            │               │
                              │  ┌────┴────┐  ┌────┴────┐         │
                              │  │Principles│ │Precedent│         │
                              │  │Expert   │  │Expert   │         │
                              │  └─────────┘  └─────────┘         │
                              │                                    │
                              │  Knowledge Graph + Vector Search   │
                              └───────────────────────────────────┘
```

---

## Repository e Loro Funzione

### 1. visualex-api (Libreria Python)
**Licenza**: MIT (Open Source)
**Pubblicata su**: PyPI (`pip install visualex`)

Libreria per l'accesso programmatico alle fonti del diritto italiano:
- **Normattiva.it**: Testi ufficiali della legislazione italiana
- **Brocardi.it**: Commenti dottrinali e massime giurisprudenziali
- **EUR-Lex**: Normativa dell'Unione Europea

```python
# Esempio d'uso (per chi programma)
from visualex import NormattivaScraper
scraper = NormattivaScraper()
articolo = await scraper.get_article("codice civile", "1453")
```

📄 [Documentazione visualex-api](./visualex-api/README.md)

---

### 2. visualex-platform (Piattaforma Web)
**Licenza**: Proprietaria
**Stack**: React + TypeScript + Express + PostgreSQL

Applicazione web completa per la ricerca giuridica:
- Interfaccia moderna e responsive
- Gestione utenti e autenticazione
- Dossier e preferiti
- Esportazione documenti

📄 [Documentazione visualex-platform](./visualex-platform/README.md)

---

### 3. merlt (Framework Machine Learning)
**Licenza**: Apache 2.0 (Open Source)
**Pubblicata su**: PyPI (`pip install merlt`)

**MERL-T** = **M**ulti-**E**xpert **L**egal **R**etrieval **T**ransformer

Framework per l'analisi giuridica basata su intelligenza artificiale:
- **Sistema Multi-Expert**: 4 esperti che replicano i canoni ermeneutici
- **Knowledge Graph**: Rappresentazione delle relazioni tra norme
- **RLCF**: Apprendimento dal feedback di esperti giuridici
- **RAG**: Retrieval-Augmented Generation per risposte fondate

📄 [Documentazione merlt](./merlt/README.md)

---

### 4. merlt-models (Modelli Addestrati)
**Licenza**: Proprietaria
**Contenuto**: Pesi dei modelli, checkpoint, configurazioni

Contiene i modelli addestrati utilizzati da MERL-T:
- Embedding models per il dominio giuridico
- Configurazioni degli Expert
- Checkpoint di addestramento

📄 [Documentazione merlt-models](./merlt-models/README.md)

---

### 5. visualex-merlt (Layer di Integrazione)
**Licenza**: Proprietaria
**Architettura**: Sistema a Plugin

Collega la piattaforma VisuaLex con il framework MERL-T:
- **8 Plugin Slots**: Punti di estensione nell'interfaccia
- **25 Eventi**: Comunicazione tra componenti
- **Pannelli Expert**: Visualizzazione analisi multi-expert

📄 [Documentazione visualex-merlt](./visualex-merlt/README.md)

---

## Documentazione per la Ricerca

Per comprendere il progetto da una prospettiva di ricerca:

| Documento | Contenuto | Per chi |
|-----------|-----------|---------|
| [GLOSSARIO.md](./GLOSSARIO.md) | Terminologia tecnico-giuridica | Tutti |
| [ARCHITETTURA.md](./ARCHITETTURA.md) | Architettura sistema (non tecnica) | Ricercatori |
| [GUIDA_NAVIGAZIONE.md](./GUIDA_NAVIGAZIONE.md) | Come orientarsi nel codice | Revisori |
| [Legacy/MERL-T_alpha/docs/](./Legacy/MERL-T_alpha/docs/) | Documentazione metodologica originale | Ricercatori |

---

## Concetti Chiave

### Knowledge Graph Giuridico

Il sistema costruisce un **grafo della conoscenza giuridica** dove:
- **Nodi** = Articoli, commi, definizioni, concetti
- **Archi** = Relazioni semantiche (RINVIA, MODIFICA, DEROGA, ATTUA, etc.)

Questo permette di navigare il diritto non solo linearmente, ma seguendo le **connessioni logiche** tra norme.

### Multi-Expert System

Quattro "esperti virtuali" analizzano ogni questione giuridica secondo approcci diversi:

1. **LiteralExpert**: "Cosa dice esattamente il testo?"
2. **SystemicExpert**: "Come si colloca nel sistema normativo?"
3. **PrinciplesExpert**: "Qual era l'intenzione del legislatore?"
4. **PrecedentExpert**: "Come è stato interpretato dalla giurisprudenza?"

Le risposte vengono poi **sintetizzate** pesando il contributo di ciascun esperto.

### RLCF (Reinforcement Learning from Community Feedback)

Sistema di apprendimento che migliora nel tempo grazie al feedback della **comunità** di esperti:
- Gli esperti giuridici valutano le risposte del sistema
- Il sistema apprende a pesare meglio i contributi degli Expert
- L'**autorità** di ogni valutatore è calcolata in base alla sua competenza

---

## Stack Tecnologico (Panoramica)

| Layer | Tecnologie | Scopo |
|-------|------------|-------|
| **Frontend** | React, TypeScript, Tailwind | Interfaccia utente |
| **Backend Web** | Express, PostgreSQL | API e persistenza |
| **Backend ML** | FastAPI, PyTorch | Servizi AI |
| **Database** | PostgreSQL, FalkorDB, Qdrant, Redis | Dati relazionali, grafi, vettori, cache |
| **Scraping** | BeautifulSoup, Playwright | Accesso fonti |

---

## Per Iniziare

### Se sei un Ricercatore (non programmatore)

1. Leggi [GLOSSARIO.md](./GLOSSARIO.md) per familiarizzare con la terminologia
2. Leggi [ARCHITETTURA.md](./ARCHITETTURA.md) per capire come funziona il sistema
3. Esplora [Legacy/MERL-T_alpha/docs/](./Legacy/MERL-T_alpha/docs/) per la metodologia

### Se vuoi esplorare il codice

1. Leggi [GUIDA_NAVIGAZIONE.md](./GUIDA_NAVIGAZIONE.md)
2. Ogni repository ha il suo README con dettagli specifici
3. Il codice è commentato in italiano dove rilevante

### Se vuoi contribuire

1. Il progetto usa Git per il version control
2. Le librerie open source (visualex-api, merlt) accettano contributi
3. Vedi i file CONTRIBUTING.md nei rispettivi repository

---

## Contesto della Ricerca

Questo progetto è sviluppato nell'ambito di una ricerca sulla **sociologia computazionale del diritto**. L'obiettivo è dimostrare che:

1. I **canoni ermeneutici** tradizionali possono essere formalizzati computazionalmente
2. Il **feedback collaborativo** di esperti può migliorare i sistemi di AI giuridica
3. I **Knowledge Graph** offrono una rappresentazione più ricca del diritto rispetto ai soli testi

La documentazione tecnica serve anche come **materiale di ricerca**, documentando le scelte metodologiche e architetturali.

---

## Licenze

| Repository | Licenza | Note |
|------------|---------|------|
| visualex-api | MIT | Completamente open source |
| merlt | Apache 2.0 | Open source con attribuzione |
| visualex-platform | Proprietaria | Codice chiuso |
| merlt-models | Proprietaria | Modelli addestrati |
| visualex-merlt | Proprietaria | Integrazione |

---

## Pubblicazioni Accademiche

Questo progetto è oggetto di ricerca accademica. I seguenti paper descrivono i fondamenti teorici e metodologici:

### ALIS - La Piattaforma

> Allega, D. (2025). The Artificial Legal Intelligence Society as an open, multi-sided platform for law-as-computation. In M. Panait, I. G. Rădulescu, B. Tudorică, C. Popescu, & M. C. Voica (Eds.), *Book of abstracts: Creativity and Innovation in Digital Economy 2025* (pp. 136–138). Petroleum-Gas University of Ploiești Publishing House. ISSN: 2971-9798

**Abstract**: Presenta ALIS come piattaforma multi-stakeholder per la computazione del diritto, con enfasi sulla dimensione sociologica e comunitaria.

### MERL-T - L'Architettura Multi-Expert

> Allega, D., & Puzio, G. (2025b). MERL-T: A multi-expert architecture for trustworthy artificial legal intelligence. In M. Panait, I. G. Rădulescu, B. Tudorică, C. Popescu, & M. C. Voica (Eds.), *Book of abstracts: Creativity and Innovation in Digital Economy 2025* (pp. 170–171). Petroleum-Gas University of Ploiești Publishing House. ISSN: 2971-9798

**Abstract**: Descrive l'architettura a 5 livelli (Scraping/ETL, Storage/Retrieval, Multi-Expert, Feedback, Governance) e i 4 Expert che implementano i canoni ermeneutici dell'Art. 12 Preleggi.

### RLCF - Il Sistema di Feedback

> Allega, D., & Puzio, G. (2025c). Reinforcement learning from community feedback (RLCF): A novel framework for artificial intelligence in social science domains. In M. Panait, I. G. Rădulescu, B. Tudorică, C. Popescu, & M. C. Voica (Eds.), *Book of abstracts: Creativity and Innovation in Digital Economy 2025* (pp. 92–94). Petroleum-Gas University of Ploiești Publishing House. ISSN: 2971-9798

**Abstract**: Propone un framework alternativo a RLHF (Reinforcement Learning from Human Feedback) basato su 4 pilastri: Dynamic Authority Scoring, Uncertainty Preservation, Constitutional Governance, Devil's Advocate System.

### Knowledge Commoditization - Fondamenti Teorici

> Allega, D., & Puzio, G. (2025a). The knowledge commoditization paradox: Theoretical and practical challenges of AI-driven value extraction in information-intensive organizations. In M. Panait, I. G. Rădulescu, B. Tudorică, C. Popescu, & M. C. Voica (Eds.), *Book of abstracts: Creativity and Innovation in Digital Economy 2025* (pp. 66–68). Petroleum-Gas University of Ploiești Publishing House. ISSN: 2971-9798

**Abstract**: Analizza il paradosso della commoditizzazione della conoscenza: la necessità e simultanea impossibilità di ridurre la conoscenza organizzativa complessa a rappresentazioni computazionali. Introduce il concetto di "entropia semantica".

---

## Riferimenti

- **Normattiva.it**: Portale della legge vigente (Istituto Poligrafico dello Stato)
- **Brocardi.it**: Enciclopedia giuridica online
- **EUR-Lex**: Accesso al diritto dell'Unione Europea
- **Art. 12 Preleggi**: Disposizioni sulla legge in generale (interpretazione)

---

*Ultimo aggiornamento: Gennaio 2026*
