# VisuaLex Platform

> **Piattaforma web per la ricerca e l'analisi giuridica italiana**

---

## Panoramica

VisuaLex Platform è l'interfaccia utente del sistema ALIS. Fornisce:
- Ricerca di testi normativi (Normattiva, Brocardi, EUR-Lex)
- Visualizzazione strutturata degli articoli
- Gestione dossier e preferiti
- Esportazione documenti
- Integrazione con MERL-T via plugin system

---

## Architettura

```
                    +------------------+
                    |     Frontend     |
                    |  React + Vite    |
                    |   Port: 5173     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+       +-----------v-----------+
    |      Backend      |       |      Python API       |
    |  Express + Prisma |       |   Quart (visualex)    |
    |    Port: 3001     |       |      Port: 5000       |
    +---------+---------+       +-----------+-----------+
              |                             |
              |                             |
    +---------v---------+                   |
    |    PostgreSQL     |                   |
    |    Port: 5432     |                   |
    +-------------------+                   |
                                           |
                              +-------------v-------------+
                              |    External Sources       |
                              | Normattiva, Brocardi, ... |
                              +---------------------------+
```

---

## Stack Tecnologico

### Frontend (`frontend/`)
| Tecnologia | Versione | Scopo |
|------------|----------|-------|
| React | 19 | UI Library |
| TypeScript | 5.x | Type Safety |
| Vite | 7 | Build Tool |
| Tailwind CSS | v4 | Styling |
| Zustand | latest | State Management |
| React Router | 7 | Routing |

### Backend (`backend/`)
| Tecnologia | Versione | Scopo |
|------------|----------|-------|
| Node.js | 20+ | Runtime |
| Express | 5 | HTTP Server |
| TypeScript | 5.x | Type Safety |
| Prisma | latest | ORM |
| PostgreSQL | 15+ | Database |
| JWT | - | Authentication |

---

## Struttura Cartelle

```
visualex-platform/
├── frontend/
│   ├── src/
│   │   ├── main.tsx                    # Entry point
│   │   ├── App.tsx                     # Root component
│   │   ├── pages/                      # Page components
│   │   │   ├── SearchPage.tsx          # Main search
│   │   │   ├── LoginPage.tsx           # Auth
│   │   │   ├── ProfilePageWrapper.tsx  # User profile
│   │   │   ├── AdminPage.tsx           # Admin panel
│   │   │   └── SettingsPage.tsx        # Settings
│   │   ├── components/
│   │   │   ├── features/               # Feature components
│   │   │   │   ├── search/             # Search & results
│   │   │   │   ├── dossier/            # Dossier management
│   │   │   │   ├── bulletin/           # Bulletin board
│   │   │   │   └── workspace/          # Workspace
│   │   │   └── ui/                     # Reusable UI
│   │   ├── lib/
│   │   │   └── plugins/                # Plugin system
│   │   │       ├── PluginRegistry.ts   # Plugin loader
│   │   │       ├── PluginSlot.tsx      # Extension points
│   │   │       ├── EventBus.ts         # Event system
│   │   │       └── types.ts            # Type definitions
│   │   ├── services/                   # API clients
│   │   └── store/                      # Zustand stores
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── backend/
│   ├── src/
│   │   ├── index.ts                    # Server entry
│   │   ├── routes/                     # API routes
│   │   │   ├── auth.ts                 # Authentication
│   │   │   ├── articles.ts             # Articles
│   │   │   └── dossier.ts              # Dossiers
│   │   └── services/                   # Business logic
│   ├── prisma/
│   │   └── schema.prisma               # Database schema
│   └── package.json
│
├── docker-compose.yml
├── .env.example
├── start_dev.sh
└── README.md
```

---

## Sistema Plugin

La piattaforma supporta plugin esterni (come MERL-T) tramite un sistema a slot.

### Plugin Slots Disponibili

| Slot Name | Location | Purpose |
|-----------|----------|---------|
| `article-toolbar` | Article header | Toolbar actions |
| `article-sidebar` | Article right panel | Analysis results |
| `article-content-overlay` | Article content | Overlays on text |
| `profile-tabs` | Profile page | Additional tabs |
| `admin-dashboard` | Admin page | Admin panels |
| `bulletin-board` | Bulletin page | Knowledge graph |
| `dossier-actions` | Dossier page | Export actions |
| `graph-view` | Workspace | Graph visualization |

### Eventi Emessi

| Event | Payload | When |
|-------|---------|------|
| `article:viewed` | `{ urn, data }` | Article loaded |
| `article:text-selected` | `{ text, position }` | Text selected |
| `citation:detected` | `{ citations[] }` | Citations found |
| `dossier:updated` | `{ dossierId }` | Dossier changed |
| `search:performed` | `{ query, results }` | Search done |

### Uso Base

```tsx
// In a component
import { PluginSlot } from '@/lib/plugins/PluginSlot';
import { eventBus } from '@/lib/plugins/EventBus';

// Render a slot
<PluginSlot name="article-sidebar" context={{ article }} />

// Emit an event
eventBus.emit('article:viewed', { urn, data });
```

---

## Development

### Prerequisites
- Node.js 20+
- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 15+

### Quick Start

```bash
# Clone repository
git clone <repo-url>
cd visualex-platform

# Copy environment variables
cp .env.example .env
# Edit .env with your credentials

# Start with Docker (recommended)
docker-compose up

# Or run services separately:

# Terminal 1: Frontend
cd frontend && npm install && npm run dev

# Terminal 2: Backend
cd backend && npm install && npx prisma migrate dev && npm run dev

# Terminal 3: Python API (from visualex-api repo)
cd ../visualex-api && ./start_dev.sh
```

### Local Start Script

```bash
./start_dev.sh
```

Avvia frontend, backend, postgres via Docker Compose.

### Access Points
| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:3001 |
| Python API | http://localhost:5000 |
| PostgreSQL | localhost:5432 |

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register user |
| POST | `/api/auth/login` | Login |
| POST | `/api/auth/logout` | Logout |
| GET | `/api/auth/me` | Current user |

### Articles
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/articles/search` | Search articles |
| GET | `/api/articles/:urn` | Get article by URN |
| GET | `/api/articles/:urn/annotations` | Get Brocardi annotations |

### Dossiers
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dossiers` | List user dossiers |
| POST | `/api/dossiers` | Create dossier |
| PUT | `/api/dossiers/:id` | Update dossier |
| DELETE | `/api/dossiers/:id` | Delete dossier |

---

## File Chiave

| File | Scopo | Complessità |
|------|-------|-------------|
| `frontend/src/components/features/search/ArticleTabContent.tsx` | Visualizzazione articoli, plugin slots | Alta (~1200 LOC) |
| `frontend/src/lib/plugins/PluginRegistry.ts` | Caricamento dinamico plugin | Media |
| `frontend/src/lib/plugins/EventBus.ts` | Comunicazione inter-componente | Bassa |
| `frontend/src/store/useAppStore.ts` | State management globale | Media |
| `backend/prisma/schema.prisma` | Schema database | Media |

---

## Testing

```bash
# Frontend tests
cd frontend && npm test

# Backend tests
cd backend && npm test

# E2E tests (requires running services)
npm run test:e2e
```

---

## Environment Variables

### Frontend (`.env`)
```bash
VITE_API_URL=http://localhost:3001
VITE_PYTHON_API_URL=http://localhost:5000
VITE_ENABLE_MERLT=true
```

### Backend (`.env`)
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/visualex
JWT_SECRET=your-secret-key
PYTHON_API_URL=http://localhost:5000
```

---

## Dependencies

| Package | Repository | Purpose |
|---------|------------|---------|
| `visualex` | visualex-api | Legal text scraping |
| `@visualex/merlt-plugin` | visualex-merlt | MERL-T integration (optional) |

---

## License

Proprietary - All rights reserved

---

## Riferimenti

- [README Principale ALIS](../README.md)
- [Architettura Sistema](../ARCHITETTURA.md)
- [Glossario](../GLOSSARIO.md)
- [Guida Navigazione](../GUIDA_NAVIGAZIONE.md)
- [Plugin Architecture Docs](docs/PLUGIN_ARCHITECTURE.md)

---

*Ultimo aggiornamento: Gennaio 2026*
