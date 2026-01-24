# System Architecture

> **ALIS_CORE Technical Architecture**

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────┐    ┌──────────────────────────────┐       │
│   │    VISUALEX-PLATFORM       │    │     VISUALEX-MERLT           │       │
│   │    (React 19 + Vite 7)     │←──→│     (Plugin System)          │       │
│   │                            │    │     - 8 Slots                │       │
│   │  - Search UI               │    │     - 25 Events              │       │
│   │  - Article Viewer          │    │     - Expert Panels          │       │
│   │  - Dossier Management      │    │     - RLCF Web               │       │
│   │  - User Profile            │    │                              │       │
│   └──────────────┬─────────────┘    └────────────────┬─────────────┘       │
│                  │                                    │                      │
└──────────────────│────────────────────────────────────│──────────────────────┘
                   │                                    │
                   ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐ │
│   │  PLATFORM BACKEND    │  │      MERLT API       │  │  VISUALEX-API    │ │
│   │  (Express 5)         │  │    (FastAPI)         │  │    (Quart)       │ │
│   │                      │  │                      │  │                  │ │
│   │  - Auth (JWT)        │  │  - /api/v1/analyze   │  │  - Normattiva    │ │
│   │  - User CRUD         │  │  - /api/v1/experts   │  │  - Brocardi      │ │
│   │  - Dossier CRUD      │  │  - /api/v1/feedback  │  │  - EUR-Lex       │ │
│   │  - Preferences       │  │  - /api/v1/graph     │  │                  │ │
│   │                      │  │  - /api/v1/rlcf      │  │                  │ │
│   │  Port: 3001          │  │  Port: 8000          │  │  Port: 5000      │ │
│   └──────────┬───────────┘  └──────────┬───────────┘  └────────┬─────────┘ │
│              │                         │                        │           │
└──────────────│─────────────────────────│────────────────────────│───────────┘
               │                         │                        │
               ▼                         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│   │  PostgreSQL  │  │   FalkorDB   │  │    Qdrant    │  │    Redis     │   │
│   │              │  │              │  │              │  │              │   │
│   │ - Users      │  │ - Legal KG   │  │ - Vectors    │  │ - Cache      │   │
│   │ - Dossiers   │  │ - Norms      │  │ - Embeddings │  │ - Sessions   │   │
│   │ - Feedback   │  │ - Relations  │  │ - Chunks     │  │ - Rate Limit │   │
│   │ - RLCF Data  │  │              │  │              │  │              │   │
│   │              │  │ Cypher Query │  │ Vector Search│  │              │   │
│   │  Port: 5432  │  │  Port: 6379  │  │  Port: 6333  │  │  Port: 6379  │   │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Presentation Layer

#### VisuaLex Platform (Frontend)

| Aspect | Details |
|--------|---------|
| **Stack** | React 19, TypeScript 5.x, Vite 7, Tailwind CSS v4 |
| **State** | Zustand 5 |
| **Routing** | React Router 7 |
| **Charts** | Recharts, D3 |
| **Graph** | Reagraph |
| **Key File** | `ArticleTabContent.tsx` (~1200 LOC, core UI) |

**Key Components:**
- `SearchBar` - Query input with NER highlighting
- `ArticleViewer` - Legal text display with annotations
- `PluginSlot` - Extension points for MERL-T integration
- `DossierManager` - Document collection management
- `GraphView` - Knowledge graph visualization

#### VisuaLex-MERLT (Plugin)

**8 Plugin Slots:**
```typescript
type SlotName =
  | 'article-toolbar'        // Analysis button
  | 'article-sidebar'        // Expert panels
  | 'article-content-overlay' // Text annotations
  | 'profile-tabs'           // User RLCF stats
  | 'admin-dashboard'        // System metrics
  | 'bulletin-board'         // Community feedback
  | 'dossier-actions'        // Dossier integration
  | 'graph-view';            // Graph visualization
```

**25 Events:**
```typescript
// Emitted by Platform
'article:viewed'        // { urn, data }
'article:text-selected' // { text, position }
'citation:detected'     // { citations[] }
'dossier:updated'       // { dossierId }
'search:performed'      // { query, results }

// Emitted by MERL-T
'merlt:analysis-started'    // { urn }
'merlt:analysis-complete'   // { urn, result }
'merlt:expert-response'     // { expert, response }
'merlt:citation-correction' // { original, corrected }
'merlt:graph-node-selected' // { nodeId }
```

---

### 2. Application Layer

#### Platform Backend (Express)

**Responsibilities:**
- User authentication (JWT)
- User/Dossier CRUD
- Rate limiting
- Static file serving

**Key Routes:**
| Route | Method | Description |
|-------|--------|-------------|
| `/api/auth/login` | POST | Login |
| `/api/auth/register` | POST | Register |
| `/api/users/:id` | GET/PUT | User profile |
| `/api/dossiers` | CRUD | Dossier management |
| `/api/preferences` | GET/PUT | User preferences |

**Database:** Prisma ORM → PostgreSQL

#### MERL-T API (FastAPI)

**Responsibilities:**
- Expert analysis
- Knowledge graph queries
- RLCF feedback collection
- Pipeline execution

**Key Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze` | POST | Multi-expert analysis |
| `/api/v1/experts` | GET | Available experts |
| `/api/v1/experts/{type}/analyze` | POST | Single expert |
| `/api/v1/feedback` | POST | Submit feedback |
| `/api/v1/feedback/{trace_id}` | GET | Get feedback |
| `/api/v1/rlcf/authority/{user_id}` | GET | Authority score |
| `/api/v1/graph/neighbors/{urn}` | GET | Graph neighbors |
| `/api/v1/graph/search` | POST | Graph search |
| `/api/v1/pipeline/ingest` | POST | Ingest document |

#### VisuaLex API (Quart)

**Responsibilities:**
- Legal source scraping
- URN generation
- Document parsing

**Scrapers:**
| Source | Method | Description |
|--------|--------|-------------|
| Normattiva.it | Playwright | Official legislation |
| Brocardi.it | BeautifulSoup | Commentary, case law |
| EUR-Lex | API/Scraping | EU legislation |

---

### 3. Data Layer

#### PostgreSQL

**Schema (Platform):**
```sql
-- Prisma-managed tables
User (id, email, password_hash, created_at, ...)
Dossier (id, user_id, name, articles[], ...)
Preference (id, user_id, theme, language, ...)
```

**Schema (RLCF):**
```sql
-- SQLAlchemy-managed tables
rlcf_traces (id, query_id, expert_type, execution_data, ...)
rlcf_feedback (id, trace_id, user_id, rating, feedback_type, ...)
policy_checkpoints (id, policy_name, weights_blob, timestamp, ...)
training_sessions (id, policy_type, status, metrics, ...)
user_authority (id, user_id, domain, score, ...)
```

#### FalkorDB (Knowledge Graph)

**Node Types:**
- `Norma` - Legislative act
- `Articolo` - Article
- `Comma` - Clause
- `Definizione` - Definition
- `Concetto` - Concept
- `Sentenza` - Court decision
- `Massima` - Legal maxim

**Edge Types:**
- `RIFERIMENTO` - References
- `MODIFICA` - Modifies
- `MODIFICATO_DA` - Modified by
- `DEROGA` - Derogates
- `ABROGA` - Abrogates
- `CITATO_DA` - Cited by
- `DEFINISCE` - Defines
- `ATTUA` - Implements
- `PRINCIPIO` - Principle relation

**Example Query:**
```cypher
MATCH (a:Articolo {urn: $urn})-[r]-(b)
WHERE type(r) IN ['RIFERIMENTO', 'CITATO_DA']
RETURN a, r, b
LIMIT 20
```

#### Qdrant (Vector Search)

**Collections:**
- `legal_chunks` - Document chunks with embeddings
- `case_law` - Jurisprudence embeddings

**Payload Schema:**
```json
{
  "chunk_id": "string",
  "article_urn": "string",
  "text": "string",
  "source_type": "norm|jurisprudence|doctrine",
  "expert_affinity": {
    "literal": 0.8,
    "systemic": 0.6,
    "principles": 0.4,
    "precedent": 0.3
  }
}
```

---

## Request Flow

### Query Analysis Flow

```
1. User submits query in Platform UI
   └─→ POST /api/v1/analyze (MERL-T)

2. MERL-T receives query
   ├─→ NER extraction (entities, norm references)
   ├─→ Query embedding generation
   └─→ ExpertRouter decision

3. Experts execute (parallel)
   ├─→ LiteralExpert.analyze()
   │     ├─→ semantic_search(Qdrant)
   │     └─→ LLM reasoning
   ├─→ SystemicExpert.analyze()
   │     ├─→ graph_search(FalkorDB)
   │     ├─→ semantic_search(Qdrant)
   │     └─→ LLM reasoning
   ├─→ PrinciplesExpert.analyze()
   │     └─→ ...
   └─→ PrecedentExpert.analyze()
         └─→ ...

4. GatingNetwork combines responses
   └─→ AdaptiveSynthesizer produces final answer

5. Response returned to UI
   └─→ ExpertResponse[] + SynthesizedResponse

6. User provides feedback (optional)
   └─→ POST /api/v1/feedback
       ├─→ RLCF learning update
       └─→ Authority score update
```

---

## Deployment Architecture

### Docker Compose (Development)

```yaml
services:
  # Databases
  postgres:
    image: postgres:15
    ports: [5432:5432]

  falkordb:
    image: falkordb/falkordb:latest
    ports: [6379:6379]

  qdrant:
    image: qdrant/qdrant:latest
    ports: [6333:6333, 6334:6334]

  redis:
    image: redis:7
    ports: [6380:6379]

  # Applications
  merlt-api:
    build: ./merlt
    ports: [8000:8000]
    depends_on: [postgres, falkordb, qdrant]

  visualex-api:
    build: ./visualex-api
    ports: [5000:5000]

  platform-backend:
    build: ./visualex-platform/backend
    ports: [3001:3001]
    depends_on: [postgres]

  platform-frontend:
    build: ./visualex-platform/frontend
    ports: [5173:5173]
```

---

## Security Considerations

### Authentication
- JWT tokens with refresh mechanism
- Bcrypt password hashing
- Rate limiting per user/IP

### Authorization
- Role-based access control
- Authority-based feedback weighting

### Data Protection
- No PII in knowledge graph
- Feedback anonymization option
- GDPR compliance for user data

---

## Scalability Considerations

### Horizontal Scaling
- Stateless API servers
- Redis session store
- Load balancer (not included)

### Performance Optimization
- Qdrant caching
- FalkorDB query optimization
- Expert parallel execution

### Future Considerations
- Kubernetes deployment
- Multi-region replication
- GPU inference for embeddings
