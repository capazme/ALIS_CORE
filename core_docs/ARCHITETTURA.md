# Architettura del Sistema ALIS

> **Guida all'architettura per ricercatori non programmatori**

Questo documento spiega come funziona ALIS senza entrare nei dettagli tecnici del codice. L'obiettivo è permettere a un giurista informatico di comprendere la logica del sistema.

---

## Visione d'Insieme

ALIS è composto da **tre livelli** che collaborano:

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVELLO PRESENTAZIONE                     │
│                                                              │
│   L'interfaccia che l'utente vede e usa                     │
│   (visualex-platform)                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LIVELLO APPLICATIVO                       │
│                                                              │
│   La logica che elabora le richieste                        │
│   (visualex-api + visualex-merlt + merlt)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LIVELLO DATI                              │
│                                                              │
│   I database che memorizzano le informazioni                │
│   (PostgreSQL + FalkorDB + Qdrant + Redis)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Livello Presentazione: L'Interfaccia Utente

### Cosa fa
- Mostra i risultati delle ricerche
- Permette di navigare tra le norme
- Gestisce account e preferenze utente
- Visualizza le analisi degli Expert

### Come appare
L'interfaccia è una **applicazione web moderna** accessibile da browser:
- Design responsive (funziona su desktop e mobile)
- Dark mode disponibile
- Pannelli laterali per informazioni aggiuntive

### Punti di Estensione (Plugin Slots)
L'interfaccia ha 8 "slot" dove il plugin MERL-T può inserire contenuti:

| Slot | Dove appare | Cosa mostra |
|------|-------------|-------------|
| `article-toolbar` | Barra strumenti articolo | Pulsante "Analizza con MERL-T" |
| `article-sidebar` | Pannello laterale | Risultati analisi multi-expert |
| `article-content-overlay` | Sopra il testo | Correzioni citazioni |
| `profile-tabs` | Pagina profilo | Tab MERL-T |
| `admin-dashboard` | Dashboard admin | Pannello amministrazione |
| `bulletin-board` | Bacheca | Knowledge Graph explorer |
| `dossier-actions` | Pagina dossier | Esportazione dati training |
| `graph-view` | Vista grafo | Visualizzazione grafo |

---

## Livello Applicativo: La Logica del Sistema

### 1. Accesso alle Fonti (visualex-api)

**Funzione**: Recuperare i testi normativi dalle fonti ufficiali.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Normattiva.it  │     │   Brocardi.it    │     │     EUR-Lex      │
│   (Testi uff.)   │     │   (Commenti)     │     │   (Dir. UE)      │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │      visualex-api       │
                    │                         │
                    │  • Scarica i testi      │
                    │  • Estrae la struttura  │
                    │  • Normalizza il formato│
                    └─────────────────────────┘
```

**Processo**:
1. L'utente cerca "art. 1453 codice civile"
2. Il sistema interroga Normattiva.it
3. Il testo viene scaricato e "parsato" (analizzato)
4. La struttura (rubrica, commi, etc.) viene estratta
5. Il risultato viene restituito in formato standardizzato

### 2. Sistema Multi-Expert (merlt)

**Funzione**: Analizzare le questioni giuridiche secondo diversi approcci ermeneutici.

#### I Quattro Expert

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOMANDA GIURIDICA                            │
│           "Quando il debitore è responsabile?"                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  LITERAL EXPERT │  │ SYSTEMIC EXPERT │  │PRINCIPLES EXPERT│
│                 │  │                 │  │                 │
│ "Art. 1218 c.c.:│  │ "Nel sistema    │  │ "La ratio è     │
│  Il debitore... │  │  della resp.    │  │  tutelare il    │
│  è tenuto al    │  │  contrattuale..."│  │  creditore..."  │
│  risarcimento"  │  │                 │  │                 │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                     │                     │
         │           ┌─────────────────┐             │
         │           │PRECEDENT EXPERT │             │
         │           │                 │             │
         │           │ "Cass. 2023/123:│             │
         │           │  La Corte ha..."│             │
         │           └────────┬────────┘             │
         │                    │                      │
         └────────────────────┼──────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   SYNTHESIZER   │
                    │                 │
                    │ Combina le      │
                    │ risposte pesando│
                    │ ogni Expert     │
                    └─────────────────┘
```

#### Come funziona ogni Expert

| Expert | Canone Ermeneutico | Cosa cerca | Fonti preferite |
|--------|-------------------|------------|-----------------|
| **Literal** | Art. 12 co. 1 Preleggi | Significato letterale, definizioni | Testo normativo |
| **Systemic** | Interpretazione sistematica | Contesto, norme collegate | Grafo delle relazioni |
| **Principles** | Ratio legis | Intenzione legislatore, principi cost. | Lavori preparatori, Cost. |
| **Precedent** | Giurisprudenza | Interpretazioni consolidate | Massime, sentenze |

#### Il Processo di Sintesi

1. Ogni Expert produce una risposta con le sue fonti
2. Il **Synthesizer** pesa le risposte
3. I pesi dipendono da:
   - Tipo di domanda (letterale vs. sistematica)
   - Qualità delle fonti trovate
   - Feedback passati (RLCF)

### 3. Knowledge Graph

**Funzione**: Rappresentare le relazioni tra norme come un grafo navigabile.

```
                            ┌─────────────────┐
                            │  Art. 1453 c.c. │
                            │  (Risoluzione)  │
                            └────────┬────────┘
                                     │ RINVIA
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │Art. 1218 c.c.│ │Art. 1223 c.c.│ │Art. 1455 c.c.│
           │(Responsab.)  │ │(Risarcimento)│ │(Importanza)  │
           └──────┬───────┘ └──────────────┘ └──────────────┘
                  │ DEFINISCE
                  ▼
           ┌──────────────┐
           │"Inadempimento"│
           │  (Concetto)   │
           └──────────────┘
```

**Vantaggi del Grafo**:
- **Navigazione**: Dall'art. 1453 si può arrivare all'art. 1218 seguendo "RINVIA"
- **Scoperta**: Si trovano norme collegate anche se non cercate direttamente
- **Contesto**: Si capisce la posizione sistemica di una norma

### 4. Ricerca Ibrida (Vector + Graph)

**Funzione**: Combinare ricerca semantica e navigazione del grafo.

```
           DOMANDA: "responsabilità per inadempimento"
                              │
         ┌────────────────────┼────────────────────┐
         │                                         │
         ▼                                         ▼
┌─────────────────────┐               ┌─────────────────────┐
│   RICERCA VETTORIALE │               │   RICERCA NEL GRAFO  │
│                      │               │                      │
│ Trova testi con      │               │ Trova norme collegate│
│ significato simile   │               │ per relazioni        │
│                      │               │                      │
│ Risultati:           │               │ Risultati:           │
│ • Art. 1218 (0.92)   │               │ • Art. 1223 (2 hop)  │
│ • Art. 1453 (0.88)   │               │ • Art. 1455 (1 hop)  │
│ • Art. 2043 (0.75)   │               │ • Art. 1256 (3 hop)  │
└──────────┬───────────┘               └──────────┬───────────┘
           │                                       │
           └───────────────────┬───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   SCORE COMBINATO   │
                    │                     │
                    │ score = α×semantico │
                    │       + (1-α)×grafo │
                    └─────────────────────┘
```

**Perché serve**:
- **Solo vettoriale**: Trova "responsabilità medica" anche se cerco contrattuale
- **Solo grafo**: Non capisce sinonimi e parafrasi
- **Ibrido**: Combina i vantaggi di entrambi

### 5. RLCF (Apprendimento dalla Comunità)

**Funzione**: Migliorare il sistema grazie al feedback degli esperti.

```
┌────────────────────────────────────────────────────────────────────┐
│                         CICLO RLCF                                  │
└────────────────────────────────────────────────────────────────────┘

   ┌──────────┐     ┌──────────────┐     ┌──────────────┐
   │  UTENTE  │────►│   SISTEMA    │────►│   RISPOSTA   │
   │  (Query) │     │   (Expert)   │     │  (con fonti) │
   └──────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │   ESPERTO    │
                                         │  VALUTATORE  │
                                         │              │
                                         │ "Risposta    │
                                         │  corretta?"  │
                                         └──────┬───────┘
                                                │
                         ┌──────────────────────┼──────────────────────┐
                         │                      │                      │
                         ▼                      ▼                      ▼
                  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
                  │   LIVELLO    │      │   DOMINIO    │      │   AUTORITÀ   │
                  │              │      │              │      │              │
                  │ • Retrieval  │      │ • Civile     │      │ Calcolata in │
                  │ • Reasoning  │      │ • Penale     │      │ base a:      │
                  │ • Synthesis  │      │ • Admin      │      │ • Background │
                  │              │      │ • Costit.    │      │ • Coerenza   │
                  └──────────────┘      └──────────────┘      │ • Consenso   │
                         │                      │              └──────┬───────┘
                         │                      │                     │
                         └──────────────────────┼─────────────────────┘
                                                │
                                                ▼
                                    ┌────────────────────┐
                                    │  AGGIORNAMENTO     │
                                    │  PESI EXPERT       │
                                    │                    │
                                    │ Se LiteralExpert   │
                                    │ sbaglia spesso su  │
                                    │ domande sistemat., │
                                    │ peso ↓ per quel    │
                                    │ tipo di domanda    │
                                    └────────────────────┘
```

**Concetto chiave - Autorità**:
Non tutti i feedback hanno lo stesso peso. L'**autorità** di un valutatore dipende da:
- **Background**: Competenza dichiarata
- **Coerenza**: Feedback coerenti nel tempo
- **Consenso**: Allineamento con altri esperti

---

## Livello Dati: I Database

### Quattro Database Specializzati

| Database | Tipo | Cosa memorizza | Perché |
|----------|------|----------------|--------|
| **PostgreSQL** | Relazionale | Utenti, sessioni, feedback | Dati strutturati tradizionali |
| **FalkorDB** | Grafo | Nodi e relazioni del KG | Navigazione veloce del grafo |
| **Qdrant** | Vettoriale | Embedding dei testi | Ricerca semantica veloce |
| **Redis** | Cache | Risultati frequenti | Velocità |

### Perché più database?

Ogni tipo di dato ha il suo database ottimale:

```
         "Chi sono gli utenti?"          "Quali norme sono collegate?"
                  │                                  │
                  ▼                                  ▼
           ┌────────────┐                    ┌────────────┐
           │ PostgreSQL │                    │  FalkorDB  │
           │            │                    │            │
           │ Tabelle    │                    │ Grafi      │
           │ righe      │                    │ nodi+archi │
           │ colonne    │                    │            │
           └────────────┘                    └────────────┘

         "Trova testi simili"            "Cosa è stato chiesto prima?"
                  │                                  │
                  ▼                                  ▼
           ┌────────────┐                    ┌────────────┐
           │   Qdrant   │                    │   Redis    │
           │            │                    │            │
           │ Vettori    │                    │ Cache      │
           │ embedding  │                    │ veloce     │
           └────────────┘                    └────────────┘
```

---

## Flusso Completo: Dalla Domanda alla Risposta

Ecco cosa succede quando un utente fa una domanda:

```
1. UTENTE
   │
   │ "Quando il debitore è responsabile per inadempimento?"
   │
   ▼
2. FRONTEND (visualex-platform)
   │
   │ Mostra interfaccia, invia richiesta al backend
   │
   ▼
3. PREPROCESSING
   │
   │ • Estrae entità: "debitore", "responsabile", "inadempimento"
   │ • Identifica il dominio: diritto civile
   │ • Riconosce citazioni: nessuna
   │
   ▼
4. RETRIEVAL IBRIDO
   │
   │ • Qdrant: trova testi semanticamente simili
   │ • FalkorDB: trova norme collegate nel grafo
   │ • Combina i risultati
   │
   ▼
5. EXPERT SYSTEM
   │
   │ • LiteralExpert: analizza art. 1218 c.c.
   │ • SystemicExpert: colloca nel sistema della resp. contrattuale
   │ • PrinciplesExpert: individua ratio (tutela creditore)
   │ • PrecedentExpert: cita giurisprudenza rilevante
   │
   ▼
6. SYNTHESIZER
   │
   │ Combina le risposte pesando ogni Expert
   │
   ▼
7. RISPOSTA ALL'UTENTE
   │
   │ "Secondo l'art. 1218 c.c., il debitore che non esegue
   │  esattamente la prestazione dovuta è tenuto al risarcimento
   │  del danno, se non prova che l'inadempimento o il ritardo
   │  è stato determinato da impossibilità della prestazione
   │  derivante da causa a lui non imputabile.
   │
   │  Fonti: Art. 1218 c.c., Art. 1223 c.c., Cass. 2023/XXX"
   │
   ▼
8. FEEDBACK (opzionale)
   │
   │ L'utente valuta la risposta → RLCF aggiorna i pesi
```

---

## Modalità di Funzionamento

### Modalità Base (senza MERL-T)

La piattaforma può funzionare anche senza il sistema Multi-Expert:
- Ricerca testuale tradizionale
- Visualizzazione articoli
- Gestione dossier e preferiti

### Modalità Completa (con MERL-T)

Con MERL-T attivo, si aggiungono:
- Analisi multi-expert
- Navigazione Knowledge Graph
- Feedback RLCF
- Suggerimenti contestuali

---

## Considerazioni per la Ricerca

### Riproducibilità

Il sistema è progettato per essere **riproducibile**:
- Configurazioni esternalizzate in file YAML
- Seed fissi per esperimenti
- Log dettagliati di ogni operazione

### Tracciabilità

Ogni risposta include le **fonti**:
- L'Expert non può "inventare" articoli
- Ogni affermazione è collegata a una fonte verificabile
- Il grafo mostra il percorso di ragionamento

### Valutabilità

Il sistema RLCF permette di **valutare** le performance:
- Feedback strutturati per livello e dominio
- Metriche di accuratezza per Expert
- Tracciamento dell'autorità dei valutatori

---

## Prossimi Sviluppi

L'architettura è progettata per evolvere:

1. **Nuovi Expert**: Aggiungere esperti per nuovi canoni (es. comparatistico)
2. **Nuove Fonti**: Integrare altre banche dati (es. sentenze di merito)
3. **Multilingua**: Supporto per normativa UE in più lingue
4. **Multivigenza**: Analisi temporale delle modifiche normative

---

## Riferimenti Accademici

L'architettura descritta in questo documento è formalizzata nelle seguenti pubblicazioni:

- **Architettura a 5 livelli**: Allega, D., & Puzio, G. (2025b). *MERL-T: A multi-expert architecture for trustworthy artificial legal intelligence*. CIDE 2025.
- **Sistema RLCF**: Allega, D., & Puzio, G. (2025c). *Reinforcement learning from community feedback (RLCF)*. CIDE 2025.
- **Piattaforma ALIS**: Allega, D. (2025). *The Artificial Legal Intelligence Society as an open, multi-sided platform for law-as-computation*. CIDE 2025.

Per i riferimenti completi, vedi [README.md](./README.md#pubblicazioni-accademiche).

---

*Ultimo aggiornamento: Gennaio 2026*
