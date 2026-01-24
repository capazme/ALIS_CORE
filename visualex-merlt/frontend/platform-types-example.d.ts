/**
 * EXAMPLE: Types that visualex-platform MUST provide
 *
 * This file shows the types that the MERLT plugin expects from @visualex/platform.
 * These types must be exported from @visualex/platform/lib/plugins
 *
 * File location in platform: src/lib/plugins.ts
 */

declare module '@visualex/platform/lib/plugins' {
  /**
   * Plugin manifest - metadata about the plugin
   */
  export interface PluginManifest {
    id: string;
    name: string;
    version: string;
    description: string;

    // Feature flags required to use this plugin
    requiredFeatures?: string[];

    // Events this plugin listens to
    subscribedEvents?: string[];

    // UI slots this plugin provides components for
    contributedSlots?: string[];
  }

  /**
   * Context provided to plugin during initialization
   */
  export interface PluginContext {
    // Base URL for API calls
    apiBaseUrl: string;

    // Function to get current auth token
    getAuthToken: () => Promise<string>;

    // Current user info
    user?: {
      id: string;
      email?: string;
      features?: string[]; // Feature flags
    };
  }

  /**
   * UI component that fills a slot
   */
  export interface SlotComponent {
    // Slot identifier (e.g., 'article-sidebar')
    slot: string;

    // React component to render
    component: React.ComponentType<any>;

    // Priority for ordering (higher = rendered first)
    priority?: number;
  }

  /**
   * Event handler function
   */
  export interface PluginEventHandler<T = any> {
    (data: T): void | Promise<void>;
  }

  /**
   * Main plugin interface
   */
  export interface Plugin {
    // Plugin metadata
    manifest: PluginManifest;

    // Initialize plugin - returns cleanup function
    initialize: (context: PluginContext) => Promise<() => void>;

    // Get UI components for slots
    getSlotComponents?: () => SlotComponent[];

    // Get event handlers
    getEventHandlers?: () => Partial<{
      'article:viewed': PluginEventHandler<ArticleViewedEvent>;
      'article:highlighted': PluginEventHandler<ArticleHighlightedEvent>;
      'search:performed': PluginEventHandler<SearchPerformedEvent>;
      // Add more events as needed
    }>;
  }

  /**
   * Event data types
   */
  export interface ArticleViewedEvent {
    urn: string;
    articleId: string;
    userId: string;
  }

  export interface ArticleHighlightedEvent {
    urn: string;
    text: string;
    startOffset: number;
    endOffset: number;
  }

  export interface SearchPerformedEvent {
    query: string;
    filters: Record<string, any>;
    resultCount: number;
  }
}
