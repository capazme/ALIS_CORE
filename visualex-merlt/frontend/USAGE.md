# Using MERLT Plugin in visualex-platform

Guida per integrare il plugin MERLT in visualex-platform.

## Installation

### Development (Local)

```bash
# Da visualex-platform
npm install file:../visualex-merlt/frontend
```

### Production (npm registry)

```bash
npm install @visualex/merlt-plugin
```

## Loading the Plugin

### 1. Dynamic Import

```typescript
// src/plugins/merlt-loader.ts
import type { Plugin } from '@/lib/plugins';

export const loadMerltPlugin = async (): Promise<Plugin | null> => {
  try {
    // Dynamic import - bundle split, lazy loaded
    const module = await import('@visualex/merlt-plugin');
    return module.default;
  } catch (error) {
    console.error('[Platform] Failed to load MERLT plugin:', error);
    return null;
  }
};
```

### 2. Plugin Registry

```typescript
// src/plugins/registry.ts
import { loadMerltPlugin } from './merlt-loader';

export const PLUGIN_LOADERS = {
  merlt: loadMerltPlugin,
  // other plugins...
};

export const loadPlugin = async (pluginId: string) => {
  const loader = PLUGIN_LOADERS[pluginId];
  if (!loader) {
    throw new Error(`Unknown plugin: ${pluginId}`);
  }
  return await loader();
};
```

### 3. Plugin Manager

```typescript
// src/plugins/manager.ts
import { loadPlugin } from './registry';
import type { Plugin, PluginContext } from '@/lib/plugins';

export class PluginManager {
  private plugins = new Map<string, Plugin>();
  private cleanupFns = new Map<string, () => void>();

  async loadPlugin(pluginId: string, context: PluginContext) {
    const plugin = await loadPlugin(pluginId);
    if (!plugin) return;

    // Check feature flags
    if (!this.hasRequiredFeatures(plugin.manifest.requiredFeatures, context)) {
      console.warn(`[Platform] User lacks required features for ${pluginId}`);
      return;
    }

    // Initialize plugin
    const cleanup = await plugin.initialize(context);

    // Store plugin and cleanup
    this.plugins.set(pluginId, plugin);
    this.cleanupFns.set(pluginId, cleanup);

    console.log(`[Platform] Plugin ${pluginId} loaded`);
  }

  unloadPlugin(pluginId: string) {
    const cleanup = this.cleanupFns.get(pluginId);
    if (cleanup) cleanup();

    this.plugins.delete(pluginId);
    this.cleanupFns.delete(pluginId);

    console.log(`[Platform] Plugin ${pluginId} unloaded`);
  }

  getPlugin(pluginId: string): Plugin | undefined {
    return this.plugins.get(pluginId);
  }

  private hasRequiredFeatures(required: string[], context: PluginContext): boolean {
    if (!required || required.length === 0) return true;
    return required.every(feature => context.user?.features?.includes(feature));
  }
}
```

## Usage in Components

### Rendering Plugin Slots

```typescript
// src/components/ArticlePage.tsx
import { usePluginSlots } from '@/hooks/usePluginSlots';

export const ArticlePage = ({ urn }: { urn: string }) => {
  const sidebarComponents = usePluginSlots('article-sidebar');
  const toolbarComponents = usePluginSlots('article-toolbar');

  return (
    <div className="article-page">
      {/* Toolbar slot */}
      <div className="article-toolbar">
        {toolbarComponents.map((SlotComponent, index) => (
          <SlotComponent key={index} urn={urn} />
        ))}
      </div>

      <div className="article-layout">
        {/* Main content */}
        <div className="article-content">
          <ArticleRenderer urn={urn} />
        </div>

        {/* Sidebar slot */}
        <aside className="article-sidebar">
          {sidebarComponents.map((SlotComponent, index) => (
            <SlotComponent key={index} urn={urn} />
          ))}
        </aside>
      </div>
    </div>
  );
};
```

### usePluginSlots Hook

```typescript
// src/hooks/usePluginSlots.ts
import { useMemo } from 'react';
import { usePluginManager } from '@/context/PluginContext';
import type { SlotComponent } from '@/lib/plugins';

export const usePluginSlots = (slotName: string) => {
  const manager = usePluginManager();

  return useMemo(() => {
    const components: React.ComponentType<any>[] = [];

    // Collect components from all plugins
    for (const plugin of manager.getPlugins()) {
      const slots = plugin.getSlotComponents?.() || [];
      const matchingSlots = slots.filter(s => s.slot === slotName);

      // Sort by priority (higher = rendered first)
      matchingSlots.sort((a, b) => (b.priority || 0) - (a.priority || 0));

      components.push(...matchingSlots.map(s => s.component));
    }

    return components;
  }, [manager, slotName]);
};
```

### Emitting Events

```typescript
// src/hooks/useArticle.ts
import { usePluginManager } from '@/context/PluginContext';

export const useArticle = (urn: string) => {
  const manager = usePluginManager();

  useEffect(() => {
    // Emit event to plugins
    manager.emitEvent('article:viewed', {
      urn,
      articleId: extractArticleId(urn),
      userId: user?.id,
    });
  }, [urn, manager]);

  // ...
};
```

## Plugin Context

```typescript
// src/context/PluginContext.tsx
import { createContext, useContext, useEffect, useState } from 'react';
import { PluginManager } from '@/plugins/manager';
import { useAuth } from '@/hooks/useAuth';
import type { PluginContext } from '@/lib/plugins';

const PluginManagerContext = createContext<PluginManager | null>(null);

export const PluginProvider = ({ children }: { children: React.ReactNode }) => {
  const [manager] = useState(() => new PluginManager());
  const { user, getToken } = useAuth();

  useEffect(() => {
    if (!user) return;

    // Load plugins for user
    const context: PluginContext = {
      apiBaseUrl: import.meta.env.VITE_API_URL,
      getAuthToken: getToken,
      user: {
        id: user.id,
        features: user.features || [],
      },
    };

    // Load MERLT plugin if user has feature
    if (user.features?.includes('merlt')) {
      manager.loadPlugin('merlt', context);
    }

    // Cleanup on unmount
    return () => {
      manager.unloadPlugin('merlt');
    };
  }, [user, manager]);

  return (
    <PluginManagerContext.Provider value={manager}>
      {children}
    </PluginManagerContext.Provider>
  );
};

export const usePluginManager = () => {
  const manager = useContext(PluginManagerContext);
  if (!manager) {
    throw new Error('usePluginManager must be used within PluginProvider');
  }
  return manager;
};
```

## App Setup

```typescript
// src/App.tsx
import { PluginProvider } from '@/context/PluginContext';

export const App = () => {
  return (
    <AuthProvider>
      <PluginProvider>
        <RouterProvider router={router} />
      </PluginProvider>
    </AuthProvider>
  );
};
```

## TypeScript Types

```typescript
// src/lib/plugins.ts
export interface PluginManifest {
  id: string;
  name: string;
  version: string;
  description: string;
  requiredFeatures?: string[];
  subscribedEvents?: string[];
  contributedSlots?: string[];
}

export interface PluginContext {
  apiBaseUrl: string;
  getAuthToken: () => Promise<string>;
  user?: {
    id: string;
    features?: string[];
  };
}

export interface SlotComponent {
  slot: string;
  component: React.ComponentType<any>;
  priority?: number;
}

export interface PluginEventHandler<T = any> {
  (data: T): void | Promise<void>;
}

export interface Plugin {
  manifest: PluginManifest;
  initialize: (context: PluginContext) => Promise<() => void>;
  getSlotComponents?: () => SlotComponent[];
  getEventHandlers?: () => Record<string, PluginEventHandler>;
}
```

## Feature Flags

Nel backend, configura feature flags per controllare chi può usare il plugin:

```python
# backend/features.py
PLUGIN_FEATURES = {
    'merlt': {
        'name': 'MERLT Research Access',
        'description': 'Enable MERLT legal knowledge extraction',
        'default': False,
        'enabled_for': ['academic_users', 'researchers'],
    }
}
```

## Environment Variables

```env
# visualex-platform/.env
VITE_API_URL=https://api.visualex.com
VITE_ENABLE_PLUGINS=true
VITE_MERLT_PLUGIN_ENABLED=true
```

## Build Integration

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      // Don't bundle plugin code - load dynamically
      external: ['@visualex/merlt-plugin'],
    },
  },
});
```

## Troubleshooting

### Plugin not loading

1. Check user has `merlt` feature flag
2. Verify plugin installed in node_modules
3. Check browser console for errors
4. Verify API_URL is correct

### Components not rendering

1. Check slot names match: `'article-sidebar'`
2. Verify plugin initialized successfully
3. Check React DevTools for component tree

### Type errors

1. Ensure `@visualex/platform` types are available
2. Check plugin exports types correctly
3. Verify TypeScript version compatibility

## Testing

```typescript
// tests/plugins/merlt.test.ts
import { loadMerltPlugin } from '@/plugins/merlt-loader';

describe('MERLT Plugin', () => {
  it('loads successfully', async () => {
    const plugin = await loadMerltPlugin();
    expect(plugin).toBeDefined();
    expect(plugin?.manifest.id).toBe('merlt');
  });

  it('provides sidebar component', async () => {
    const plugin = await loadMerltPlugin();
    const slots = plugin?.getSlotComponents?.() || [];
    const sidebar = slots.find(s => s.slot === 'article-sidebar');
    expect(sidebar).toBeDefined();
  });
});
```
