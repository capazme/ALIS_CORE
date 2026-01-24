# Glossario Tecnico-Giuridico ALIS

> **Guida terminologica per navigare tra diritto e informatica**

Questo glossario collega i concetti tecnici dell'informatica con quelli del diritto, facilitando la comprensione del progetto ALIS per chi proviene da un background giuridico.

---

## Indice

- [A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w)

---

## A

### API (Application Programming Interface)
**Informatica**: Interfaccia che permette a programmi diversi di comunicare tra loro.
**Analogia giuridica**: Come le "clausole contrattuali standard" che definiscono come due parti devono interagire - l'API definisce le "regole d'ingaggio" tra software.

### Arco (Edge)
**Informatica**: Connessione tra due nodi in un grafo.
**Nel progetto**: Relazione tra due norme (es. l'arco "RINVIA" collega l'art. 1453 c.c. all'art. 1218 c.c.).

### Async/Asincrono
**Informatica**: Operazione che non blocca l'esecuzione mentre attende un risultato.
**Esempio**: Come un avvocato che manda una PEC e continua a lavorare invece di aspettare la risposta.

### Authority (Autorità)
**Nel progetto RLCF**: Peso assegnato al feedback di un utente in base alla sua competenza dimostrata.
**Analogia giuridica**: Come l'autorevolezza di una fonte dottrinale - il parere di un professore ordinario "pesa" più di quello di uno studente.

---

## B

### Backend
**Informatica**: La parte del software che l'utente non vede, che gestisce dati e logica.
**Analogia**: Come la "cancelleria" di un tribunale - essenziale ma non visibile al pubblico.

### Bridge Table
**Nel progetto**: Tabella che collega i "chunk" di testo (nel database vettoriale) ai nodi del Knowledge Graph.
**Funzione**: Permette di sapere che il paragrafo X del codice civile corrisponde al nodo Y nel grafo delle relazioni.

---

## C

### Cache
**Informatica**: Memoria temporanea per dati usati frequentemente.
**Analogia**: Come la "massimario" che un avvocato tiene sulla scrivania invece di andare ogni volta in biblioteca.

### Chunk
**Informatica**: Porzione di testo di dimensione gestibile per l'elaborazione.
**Nel progetto**: Un articolo lungo viene diviso in chunk più piccoli per l'analisi semantica (es. comma per comma).

### Cypher
**Informatica**: Linguaggio di query per database a grafo (usato da FalkorDB/Neo4j).
**Analogia**: Come SQL è il linguaggio per i database tradizionali, Cypher serve per "interrogare" i grafi.

---

## D

### Database
**Informatica**: Sistema organizzato per memorizzare e recuperare dati.
**Tipi nel progetto**:
- **PostgreSQL**: Database relazionale (tabelle con righe e colonne)
- **FalkorDB**: Database a grafo (nodi e relazioni)
- **Qdrant**: Database vettoriale (per ricerca semantica)
- **Redis**: Cache veloce

### Docker
**Informatica**: Tecnologia per creare "contenitori" isolati in cui far girare software.
**Analogia**: Come un "ufficio prefabbricato" - tutto il necessario è già dentro, pronto all'uso.

---

## E

### Embedding
**Informatica**: Rappresentazione numerica (vettore) di un testo che ne cattura il significato.
**Nel progetto**: Ogni articolo di legge viene trasformato in un vettore di numeri. Articoli semanticamente simili avranno vettori "vicini".
**Esempio**: "inadempimento contrattuale" e "mancata esecuzione dell'obbligazione" avranno embedding simili.

### EventBus
**Nel progetto**: Sistema di comunicazione tra componenti tramite "eventi".
**Analogia**: Come un "albo pretorio digitale" - i componenti pubblicano eventi che altri possono leggere.

### Expert (Sistema Expert)
**Nel progetto MERL-T**: Modulo specializzato in un tipo di interpretazione giuridica.
**I 4 Expert**:
1. **LiteralExpert**: Interpretazione letterale (Art. 12 Preleggi, comma 1)
2. **SystemicExpert**: Interpretazione sistematica
3. **PrinciplesExpert**: Ratio legis e principi costituzionali
4. **PrecedentExpert**: Giurisprudenza consolidata

---

## F

### FalkorDB
**Informatica**: Database a grafo ad alte prestazioni (fork di Redis Graph).
**Nel progetto**: Memorizza il Knowledge Graph giuridico - nodi (articoli, concetti) e relazioni (RINVIA, MODIFICA, DEROGA).

### FastAPI
**Informatica**: Framework Python per creare API web veloci.
**Nel progetto**: Espone i servizi MERL-T come API accessibili via web.

### Feedback Loop
**Informatica**: Ciclo in cui l'output di un sistema viene usato come input per migliorarlo.
**Nel progetto RLCF**: Gli esperti valutano le risposte → il sistema apprende → le risposte migliorano → nuove valutazioni...

### Frontend
**Informatica**: La parte del software che l'utente vede e con cui interagisce.
**Analogia**: Come la "sala d'attesa e sportello" di un ufficio pubblico.

---

## G

### Git
**Informatica**: Sistema di controllo versione per tracciare modifiche al codice.
**Analogia**: Come il "registro delle modifiche" di un atto normativo - si può vedere chi ha modificato cosa e quando.

### Grafo (Graph)
**Informatica**: Struttura dati composta da nodi (punti) e archi (connessioni).
**Nel progetto**: Il diritto viene rappresentato come grafo dove:
- **Nodi** = Articoli, commi, concetti giuridici
- **Archi** = Relazioni (rinvio, deroga, modifica, abrogazione)

### GraphRAG
**Informatica**: Retrieval-Augmented Generation che usa grafi invece di sola ricerca testuale.
**Nel progetto**: La ricerca non trova solo testi simili, ma anche norme "connesse" nel grafo.

---

## H

### Hybrid Search (Ricerca Ibrida)
**Nel progetto**: Combina ricerca semantica (embedding) e ricerca nel grafo.
**Formula**: `score = α × similarity + (1-α) × graph_score`
**Vantaggio**: Trova norme sia "semanticamente simili" che "giuridicamente connesse".

---

## I

### Ingestion
**Informatica**: Processo di acquisizione e elaborazione dati da fonti esterne.
**Nel progetto**: Scaricare un articolo da Normattiva, parsarlo, estrarre entità, creare nodi nel grafo, generare embedding.

---

## K

### Knowledge Graph (Grafo della Conoscenza)
**Informatica**: Rappresentazione strutturata della conoscenza come rete di entità e relazioni.
**Nel progetto**: Il diritto italiano rappresentato come grafo dove si può "navigare" da una norma all'altra seguendo le relazioni.
**Esempio**: Da art. 1453 c.c. → (RINVIA) → art. 1218 c.c. → (DEFINISCE) → "inadempimento"

---

## L

### LLM (Large Language Model)
**Informatica**: Modello di intelligenza artificiale addestrato su grandi quantità di testo.
**Esempi**: GPT-4, Claude, Gemini.
**Nel progetto**: Gli Expert usano LLM per generare risposte, ma le **fonti** vengono dal database (mai inventate).

---

## M

### MERL-T (Multi-Expert Legal Retrieval Transformer)
**Nome del framework**: Sistema che combina più "esperti" AI per l'analisi giuridica.
**Componenti**:
- **M**ulti-Expert: 4 esperti specializzati
- **L**egal: Dominio giuridico
- **R**etrieval: Recupero informazioni
- **T**ransformer: Architettura AI

### Multivigenza
**Diritto**: Coesistenza di versioni diverse della stessa norma nel tempo.
**Nel progetto**: Il sistema traccia le modifiche temporali delle norme, permettendo di vedere "come era" un articolo in una data specifica.

---

## N

### NER (Named Entity Recognition)
**Informatica**: Identificazione automatica di "entità nominate" in un testo.
**Nel progetto**: Riconoscimento di riferimenti normativi ("art. 1453 c.c."), concetti giuridici ("inadempimento"), soggetti ("debitore").

### Nodo (Node)
**Informatica**: Punto in un grafo.
**Nel progetto**: Può essere un articolo, un comma, un concetto giuridico, una definizione.

---

## O

### Open Source
**Informatica**: Software il cui codice sorgente è pubblicamente accessibile.
**Nel progetto**: `visualex-api` e `merlt` sono open source (licenze MIT e Apache 2.0).

### Orchestrator
**Nel progetto MERL-T**: Componente che coordina i 4 Expert, decidendo quanto peso dare a ciascuno.

---

## P

### Plugin
**Informatica**: Componente aggiuntivo che estende le funzionalità di un software.
**Nel progetto**: MERL-T è integrato in VisuaLex come plugin - può essere attivato/disattivato senza modificare la piattaforma base.

### PluginSlot
**Nel progetto**: Punto dell'interfaccia dove un plugin può inserire contenuti.
**Esempio**: Lo slot `article-toolbar` permette a MERL-T di aggiungere il pulsante "Analizza".

### PostgreSQL
**Informatica**: Database relazionale open source.
**Nel progetto**: Memorizza utenti, sessioni, preferiti, feedback RLCF.

### PyPI (Python Package Index)
**Informatica**: Repository ufficiale dei pacchetti Python.
**Nel progetto**: `visualex` e `merlt` sono pubblicati su PyPI e installabili con `pip install`.

---

## Q

### Qdrant
**Informatica**: Database specializzato per ricerca vettoriale.
**Nel progetto**: Memorizza gli embedding degli articoli, permettendo ricerca per "somiglianza semantica".

### Query
**Informatica**: Richiesta di dati a un database.
**Tipi nel progetto**:
- SQL per PostgreSQL
- Cypher per FalkorDB
- Vector search per Qdrant

---

## R

### RAG (Retrieval-Augmented Generation)
**Informatica**: Tecnica che combina recupero di informazioni + generazione AI.
**Nel progetto**: Prima si recuperano le norme pertinenti (Retrieval), poi l'LLM genera la risposta (Generation) basandosi su di esse.
**Vantaggio**: Le risposte sono sempre **fondate** su fonti reali, mai inventate.

### React
**Informatica**: Libreria JavaScript per costruire interfacce utente.
**Nel progetto**: Il frontend di VisuaLex è costruito con React.

### Redis
**Informatica**: Database in-memory velocissimo.
**Nel progetto**: Usato come cache per velocizzare operazioni frequenti.

### Repository (Repo)
**Informatica**: Contenitore di codice sorgente gestito con Git.
**Nel progetto**: ALIS_CORE contiene 5 repository (cartelle) principali.

### RLCF (Reinforcement Learning from Community Feedback)
**Nel progetto**: Sistema di apprendimento che migliora grazie al feedback della comunità di esperti giuridici.
**Meccanismo**:
1. Un esperto valuta una risposta del sistema
2. Il sistema calcola l'**autorità** del valutatore
3. Il feedback viene aggregato con quelli di altri esperti
4. I pesi degli Expert vengono aggiornati
5. Le risposte future migliorano

---

## S

### Scraper
**Informatica**: Programma che estrae dati da siti web.
**Nel progetto**:
- `NormattivaScraper`: Estrae testi da Normattiva.it
- `BrocardiScraper`: Estrae commenti da Brocardi.it

### Slot
Vedi [PluginSlot](#pluginslot).

---

## T

### Traversal (Attraversamento)
**Informatica**: Navigazione di un grafo seguendo gli archi.
**Nel progetto**: Partendo da un articolo, si "attraversa" il grafo seguendo le relazioni per trovare norme collegate.
**Esempio**: Da "art. 1453 c.c." traverso la relazione "RINVIA" per arrivare a "art. 1218 c.c."

### TypeScript
**Informatica**: Linguaggio di programmazione (superset di JavaScript) con tipi statici.
**Nel progetto**: Usato per frontend (React) e backend (Express) di VisuaLex.

---

## U

### URN (Uniform Resource Name)
**Standard**: Identificatore univoco per risorse.
**Nel progetto**: Ogni norma ha un URN nel formato standard italiano.
**Esempio**: `urn:nir:stato:legge:2023-01-01;1` identifica la Legge 1/2023.

---

## V

### Vector (Vettore)
**Matematica/Informatica**: Lista ordinata di numeri.
**Nel progetto**: Gli embedding sono vettori che rappresentano il "significato" di un testo.
**Dimensione**: Tipicamente 768 o 1024 numeri per ogni testo.

### Vector Database
Vedi [Qdrant](#qdrant).

---

## W

### Workflow
**Informatica**: Sequenza di operazioni automatizzate.
**Nel progetto**: Pipeline di ingestion, analisi multi-expert, feedback loop.

---

## Relazioni nel Knowledge Graph

Il grafo giuridico usa queste relazioni principali:

| Relazione | Significato | Esempio |
|-----------|-------------|---------|
| `RINVIA` | Riferimento normativo | Art. 1453 → Art. 1218 |
| `MODIFICA` | Modifica normativa | L. 2023/1 → Art. 100 c.p. |
| `DEROGA` | Eccezione alla regola | Art. speciale → Art. generale |
| `ABROGA` | Abrogazione | L. nuova → Art. vecchio |
| `ATTUA` | Attuazione | D.M. → Legge delega |
| `DEFINISCE` | Definizione | Art. → Concetto |
| `INTERPRETA` | Interpretazione giurisprudenziale | Sentenza → Articolo |

---

## Acronimi Frequenti

| Acronimo | Significato |
|----------|-------------|
| ALIS | Artificial Legal Intelligence System |
| API | Application Programming Interface |
| c.c. | Codice Civile |
| c.p. | Codice Penale |
| DB | Database |
| KG | Knowledge Graph |
| LLM | Large Language Model |
| MERL-T | Multi-Expert Legal Retrieval Transformer |
| ML | Machine Learning |
| NER | Named Entity Recognition |
| RAG | Retrieval-Augmented Generation |
| RLCF | Reinforcement Learning from Community Feedback |
| URN | Uniform Resource Name |

---

*Ultimo aggiornamento: Gennaio 2026*
