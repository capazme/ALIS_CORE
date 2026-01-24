/**
 * Plugin System Types
 *
 * Defines the contract for plugins that can extend visualex-platform.
 * MERLT is the first plugin, but the system is designed to support any plugin.
 */

import type { ComponentType, RefObject } from 'react';

/**
 * Events that plugins can subscribe to
 */
export interface PluginEvents {
  // Article events
  'article:viewed': { urn: string; articleId: string; userId?: string };
  'article:scrolled': { urn: string; visibleSections: string[] };
  'article:highlighted': { urn: string; text: string; startOffset: number; endOffset: number };
  'article:text-selected': { urn: string; text: string; startOffset: number; endOffset: number };

  // Search events
  'search:performed': { query: string; filters: Record<string, unknown>; resultCount: number };
  'search:result-clicked': { urn: string; position: number; query: string };

  // User events
  'user:logged-in': { userId: string; features: string[] };
  'user:logged-out': { userId: string };

  // Bookmark events
  'bookmark:created': { urn: string; userId: string };
  'bookmark:deleted': { urn: string; userId: string };

  // MERLT-specific events (emitted by plugins, consumed by platform or other plugins)
  'enrichment:requested': { urn: string; userId: string };
  'enrichment:started': { urn: string; articleKey: string };
  'enrichment:completed': { urn: string; entitiesCount: number; relationsCount: number };
  'entity:validated': { entityId: string; vote: string; userId: string };
  'relation:validated': { relationId: string; vote: string; userId: string };
  'citation:detected': { urn: string; text: string; parsed: Record<string, unknown> };

  // Graph events
  'graph:node-clicked': { nodeId: string; nodeType: string };
  'graph:edge-clicked': { edgeId: string; edgeType: string };

  // Issue events
  'issue:viewed': { issueId: string };
  'issue:voted': { issueId: string; vote: string; userId: string };
  'issue:reported': { nodeId: string; issueType: string };

  // Dossier events
  'dossier:training-exported': { dossierId: string; format: string };
}

export type PluginEventName = keyof PluginEvents;
export type PluginEventHandler<T extends PluginEventName> = (data: PluginEvents[T]) => void;

/**
 * Slot names where plugins can inject UI components
 */
export type PluginSlotName =
  | 'article-sidebar'       // Right sidebar in article view
  | 'article-toolbar'       // Toolbar above article content
  | 'article-footer'        // Below article content
  | 'article-content-overlay' // Floating overlay on article content (e.g., citation correction)
  | 'search-filters'        // Additional search filters
  | 'user-menu'             // User menu items
  | 'settings-panel'        // Settings page sections
  | 'profile-tabs'          // Additional tabs in user profile
  | 'admin-dashboard'       // Admin dashboard panels
  | 'bulletin-board'        // Bulletin board / community features
  | 'graph-explorer'        // Knowledge graph explorer
  | 'graph-view'            // Graph view in workspace tabs
  | 'dossier-actions'       // Dossier page actions (export/import)
  | 'global-overlay';       // Full-screen overlays

/**
 * Props passed to slot components
 */
export interface SlotProps {
  'article-sidebar': { urn: string; articleId: string };
  'article-toolbar': { urn: string; articleId: string };
  'article-footer': { urn: string; articleId: string };
  'article-content-overlay': { urn: string; articleId: string; contentRef: RefObject<HTMLElement> };
  'search-filters': { currentFilters: Record<string, unknown> };
  'user-menu': { userId: string };
  'settings-panel': { userId: string };
  'profile-tabs': { userId: string };
  'admin-dashboard': { userId: string };
  'bulletin-board': { userId: string };
  'graph-explorer': { urn?: string; depth?: number };
  'graph-view': { rootUrn: string; depth?: number; userId?: string };
  'dossier-actions': {
    dossierId: string;
    userId: string;
    dossier: {
      title: string;
      description?: string;
      tags?: string[];
      items: Array<{
        type: 'norma' | 'note';
        status?: 'unread' | 'reading' | 'important' | 'done';
        data: {
          urn?: string;
          tipo_atto?: string;
          numero_atto?: string;
          numero_articolo?: string;
          data?: string;
        };
      }>;
    };
  };
  'global-overlay': Record<string, never>;
}

/**
 * A component that a plugin contributes to a slot
 */
export interface SlotComponent<T extends PluginSlotName = PluginSlotName> {
  slot: T;
  component: ComponentType<SlotProps[T]>;
  priority?: number; // Higher priority = rendered first
}

/**
 * Plugin manifest - declares what the plugin provides
 */
export interface PluginManifest {
  id: string;
  name: string;
  version: string;
  description: string;

  // Required feature flags for this plugin
  requiredFeatures: string[];

  // Events this plugin subscribes to
  subscribedEvents: PluginEventName[];

  // Slots this plugin contributes to
  contributedSlots: PluginSlotName[];
}

/**
 * Plugin interface that all plugins must implement
 */
export interface Plugin {
  manifest: PluginManifest;

  /**
   * Called when plugin is loaded. Return cleanup function.
   */
  initialize(context: PluginContext): Promise<() => void>;

  /**
   * Returns components to render in slots
   */
  getSlotComponents(): SlotComponent[];

  /**
   * Returns event handlers
   */
  getEventHandlers(): Partial<{
    [K in PluginEventName]: PluginEventHandler<K>;
  }>;
}

/**
 * Context provided to plugins during initialization
 */
export interface PluginContext {
  /**
   * Current user info
   */
  user: {
    id: string;
    features: string[];
  } | null;

  /**
   * API base URL for plugin-specific backends
   */
  apiBaseUrl: string;

  /**
   * Emit an event that other plugins can listen to
   */
  emit: <T extends PluginEventName>(event: T, data: PluginEvents[T]) => void;

  /**
   * Get authentication token for API calls
   */
  getAuthToken: () => Promise<string | null>;
}

/**
 * Plugin loader function type (for dynamic imports)
 */
export type PluginLoader = () => Promise<{ default: Plugin }>;

/**
 * Plugin registry configuration
 */
export interface PluginConfig {
  id: string;
  enabled: boolean;
  loader: PluginLoader;
}
