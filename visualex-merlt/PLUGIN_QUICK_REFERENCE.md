# MERLT Plugin - Quick Reference

Fast lookup guide for MERLT plugin implementation details.

## Plugin Manifest

```typescript
{
  id: 'merlt',
  name: 'MERLT Research',
  version: '1.0.0',
  description: 'Legal knowledge extraction and validation for research',
  requiredFeatures: ['merlt'],
  subscribedEvents: ['article:viewed', 'article:highlighted', 'article:text-selected', 'citation:detected', 'search:performed'],
  contributedSlots: ['article-sidebar', 'article-toolbar', 'article-content-overlay', 'bulletin-board', 'dossier-actions', 'graph-view', 'profile-tabs', 'admin-dashboard']
}
```

## Plugin Slots (14 Available)

### MERLT Plugin Implements (8 slots)

| Slot | Component | Props | Purpose |
|------|-----------|-------|---------|
| `article-sidebar` | MerltSidebarPanel | `{ urn, articleId }` | Main analysis panel with entities, validation queue |
| `article-toolbar` | MerltToolbar | `{ urn, articleId }` | Toolbar with enrichment trigger, status indicators |
| `article-content-overlay` | MerltContentOverlay | `{ urn, articleId, contentRef }` | Floating UI for citations, entity tagging |
| `bulletin-board` | BulletinBoardSlot | `{ userId }` | Knowledge graph visualization, issue management |
| `dossier-actions` | DossierActionsSlot | `{ dossierId, userId, dossier }` | Export dossier for training, manage items |
| `graph-view` | GraphViewSlot | `{ rootUrn, depth?, userId? }` | Graph visualization in workspace |
| `profile-tabs` | ProfilePage | `{ userId }` | User profile tab with authority scores, history |
| `admin-dashboard` | AcademicDashboard | `{ userId }` | Admin panel with pipeline monitoring, stats |

### Available for Other Plugins (6 slots)

| Slot | Props | Purpose |
|------|-------|---------|
| `article-footer` | `{ urn, articleId }` | Below article content |
| `search-filters` | `{ currentFilters }` | Additional search filters |
| `user-menu` | `{ userId }` | User menu items |
| `settings-panel` | `{ userId }` | Settings page sections |
| `graph-explorer` | `{ urn?, depth? }` | Knowledge graph exploration |
| `global-overlay` | `{}` | Full-screen overlays |

## Events (25 Total)

### Article Events (5)

| Event | Payload | Handler |
|-------|---------|---------|
| `article:viewed` | `{ urn, articleId, userId }` | `trackArticleView` |
| `article:scrolled` | `{ urn, visibleSections }` | - |
| `article:highlighted` | `{ urn, text, startOffset, endOffset }` | `trackHighlight` |
| `article:text-selected` | `{ urn, text, startOffset, endOffset }` | - |
| `article-footer` | - | - |

### Search Events (2)

| Event | Payload | Handler |
|-------|---------|---------|
| `search:performed` | `{ query, filters, resultCount }` | `trackSearch` |
| `search:result-clicked` | `{ urn, position, query }` | - |

### User Events (2)

| Event | Payload | Handler |
|-------|---------|---------|
| `user:logged-in` | `{ userId, features }` | - |
| `user:logged-out` | `{ userId }` | - |

### Bookmark Events (2)

| Event | Payload | Handler |
|-------|---------|---------|
| `bookmark:created` | `{ urn, userId }` | - |
| `bookmark:deleted` | `{ urn, userId }` | - |

### Enrichment Pipeline Events (3)

| Event | Payload | Handler |
|-------|---------|---------|
| `enrichment:requested` | `{ urn, userId }` | - |
| `enrichment:started` | `{ urn, articleKey }` | - |
| `enrichment:completed` | `{ urn, entitiesCount, relationsCount }` | - |

### Validation Events (2)

| Event | Payload | Handler |
|-------|---------|---------|
| `entity:validated` | `{ entityId, vote, userId }` | - |
| `relation:validated` | `{ relationId, vote, userId }` | - |

### Citation Events (1)

| Event | Payload | Handler |
|-------|---------|---------|
| `citation:detected` | `{ urn, text, parsed }` | - |

### Graph Events (2)

| Event | Payload | Handler |
|-------|---------|---------|
| `graph:node-clicked` | `{ nodeId, nodeType }` | - |
| `graph:edge-clicked` | `{ edgeId, edgeType }` | - |

### Issue Events (3)

| Event | Payload | Handler |
|-------|---------|---------|
| `issue:viewed` | `{ issueId }` | - |
| `issue:voted` | `{ issueId, vote, userId }` | - |
| `issue:reported` | `{ nodeId, issueType }` | - |

### Dossier Events (1)

| Event | Payload | Handler |
|-------|---------|---------|
| `dossier:training-exported` | `{ dossierId, format }` | - |

## MERLT Components

```
src/components/
├── MerltSidebarPanel        // article-sidebar slot
├── MerltToolbar             // article-toolbar slot
├── MerltContentOverlay      // article-content-overlay slot
├── slots/
│   ├── BulletinBoardSlot    // bulletin-board slot
│   ├── DossierActionsSlot   // dossier-actions slot
│   └── GraphViewSlot        // graph-view slot
├── merlt/
│   ├── profile/
│   │   └── ProfilePage      // profile-tabs slot
│   └── dashboard/
│       └── AcademicDashboard // admin-dashboard slot
└── services/
    ├── merltInit.ts         // Initialize services
    ├── tracking.ts          // Event tracking
    └── ...
```

## Event Handler Implementation

```typescript
getEventHandlers() {
  return {
    'article:viewed': (data) => trackArticleView(data.urn, data.articleId, data.userId),
    'article:highlighted': (data) => trackHighlight(data.urn, data.text, data.startOffset, data.endOffset),
    'search:performed': (data) => trackSearch(data.query, data.filters, data.resultCount),
  };
}
```

**Active Handlers**: 3 (article:viewed, article:highlighted, search:performed)
**Subscribed but inactive**: 2 (article:text-selected, citation:detected)

## Feature Flags

| Flag | Purpose | Sub-features |
|------|---------|--------------|
| `merlt` | Enable MERLT plugin | - |
| `merlt_contribution` | Allow entity contribution | - |
| `merlt_validation` | Allow validation voting | - |

```typescript
// Check in plugin context
const canValidate = context.user?.features.includes('merlt_validation');
const canContribute = context.user?.features.includes('merlt_contribution');
```

## User Enablement Flow

```
1. Admin enables 'merlt' feature for user
   PUT /api/admin/users/:userId/features/merlt { enabled: true }

2. User gives consent
   POST /api/users/me/features/merlt/consent { consent: true }

3. At login, user receives features list
   GET /api/users/me/features → { features: ["merlt"] }

4. PluginProvider loads MERLT plugin dynamically
   import('@visualex/merlt-plugin')

5. Plugin initializes with feature flags
   merltPlugin.initialize(context)

6. Components render in slots
   PluginSlot renders MerltSidebarPanel, etc.
```

## Plugin Initialization

```typescript
async initialize(context: PluginContext): Promise<() => void> {
  // Called when plugin loads
  // context.user.features contains feature flags
  // context.getAuthToken() returns JWT token

  // Setup MERLT backend
  await initializeMerltServices({
    apiBaseUrl: context.apiBaseUrl,
    getAuthToken: context.getAuthToken,
    userId: context.user?.id,
  });

  // Return cleanup function
  return () => shutdownMerltServices();
}
```

## Deployment Checklist

- [ ] Plugin built: `npm run build:plugin`
- [ ] Output verified: `./verify-build.sh`
- [ ] Plugin distributed (npm/PyPI)
- [ ] visualex-platform updated
- [ ] Feature flag 'merlt' created in backend
- [ ] Admin test user configured with 'merlt' feature
- [ ] EventBus emits events correctly
- [ ] Plugin slots render components
- [ ] Event handlers receive data

## Troubleshooting Quick Links

### Plugin not loading
- Check user has 'merlt' feature: `GET /api/users/me/features`
- Check browser console for import errors
- Verify plugin path in dynamic import

### Events not received
- Check manifest.subscribedEvents includes event name
- Verify EventBus.emit() called with exact event name
- Debug with: `EventBus.getHistory('article:viewed')`

### Components not showing
- Verify slot name matches PluginSlotName type
- Check props passed match SlotProps[slotName]
- Higher priority = rendered first (100 > 50)

### Auth errors
- getAuthToken() should return valid JWT
- apiBaseUrl should reach MERLT backend
- Check CORS headers

## File Locations

| Item | Path |
|------|------|
| Plugin entry | `visualex-merlt/frontend/src/plugin/index.ts` |
| Plugin types | `visualex-platform/frontend/src/lib/plugins/types.ts` |
| EventBus | `visualex-platform/frontend/src/lib/plugins/EventBus.ts` |
| PluginSlot | `visualex-platform/frontend/src/lib/plugins/PluginSlot.tsx` |
| Architecture docs | `visualex-merlt/docs/PLUGIN_ARCHITECTURE.md` |

## Quick Stats

- **Slots Implemented**: 8 (57% of 14 available)
- **Events Subscribed**: 5 (20% of 25 available)
- **Components**: 8
- **Feature Flags**: 3
- **Services**: ~5+ (init, tracking, etc.)
- **Documentation**: ~600 lines of detailed specs

## See Also

- **Full Architecture**: `docs/PLUGIN_ARCHITECTURE.md`
- **Build System**: `frontend/PLUGIN.md`
- **Integration Guide**: `frontend/USAGE.md`
- **Index**: `frontend/DOCS-INDEX.md`

---

**Last Updated**: January 19, 2026
