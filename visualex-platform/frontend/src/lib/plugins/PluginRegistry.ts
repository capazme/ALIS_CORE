/**
 * Plugin Registry
 *
 * Manages plugin lifecycle: loading, initialization, and cleanup.
 * Only loads plugins when user has the required feature flags.
 */

import type {
  Plugin,
  PluginConfig,
  PluginContext,
  PluginEventName,
  PluginEventHandler,
  SlotComponent,
  PluginSlotName,
} from './types';
import { EventBus } from './EventBus';

interface LoadedPlugin {
  plugin: Plugin;
  cleanup: () => void;
  eventUnsubscribers: Array<() => void>;
}

class PluginRegistryImpl {
  private configs: Map<string, PluginConfig> = new Map();
  private loadedPlugins: Map<string, LoadedPlugin> = new Map();
  private currentUser: { id: string; features: string[] } | null = null;
  private apiBaseUrl = '';
  private getAuthToken: () => Promise<string | null> = async () => null;

  /**
   * Register a plugin configuration (does not load it yet)
   */
  register(config: PluginConfig): void {
    this.configs.set(config.id, config);
  }

  /**
   * Configure the registry with user context
   */
  configure(options: {
    user: { id: string; features: string[] } | null;
    apiBaseUrl: string;
    getAuthToken: () => Promise<string | null>;
  }): void {
    this.currentUser = options.user;
    this.apiBaseUrl = options.apiBaseUrl;
    this.getAuthToken = options.getAuthToken;
  }

  /**
   * Load all plugins that the current user has access to
   */
  async loadPlugins(): Promise<void> {
    if (!this.currentUser) {
      // No user logged in, unload all plugins
      await this.unloadAll();
      return;
    }

    const userFeatures = new Set(this.currentUser.features);

    for (const [id, config] of this.configs) {
      if (!config.enabled) {
        // Plugin is disabled globally
        if (this.loadedPlugins.has(id)) {
          await this.unloadPlugin(id);
        }
        continue;
      }

      try {
        // Load the plugin module
        const module = await config.loader();
        const plugin = module.default;

        // Check if user has required features
        const hasAccess = plugin.manifest.requiredFeatures.every((f) => userFeatures.has(f));

        if (hasAccess && !this.loadedPlugins.has(id)) {
          await this.initializePlugin(id, plugin);
        } else if (!hasAccess && this.loadedPlugins.has(id)) {
          await this.unloadPlugin(id);
        }
      } catch (error) {
        console.error(`[PluginRegistry] Failed to load plugin "${id}":`, error);
      }
    }
  }

  /**
   * Initialize a single plugin
   */
  private async initializePlugin(id: string, plugin: Plugin): Promise<void> {
    const context: PluginContext = {
      user: this.currentUser,
      apiBaseUrl: this.apiBaseUrl,
      emit: (event, data) => EventBus.emit(event, data),
      getAuthToken: this.getAuthToken,
    };

    // Initialize plugin
    const cleanup = await plugin.initialize(context);

    // Subscribe to events
    const eventHandlers = plugin.getEventHandlers();
    const eventUnsubscribers: Array<() => void> = [];

    for (const [eventName, handler] of Object.entries(eventHandlers)) {
      if (handler) {
        const unsubscribe = EventBus.on(
          eventName as PluginEventName,
          handler as PluginEventHandler<PluginEventName>
        );
        eventUnsubscribers.push(unsubscribe);
      }
    }

    this.loadedPlugins.set(id, {
      plugin,
      cleanup,
      eventUnsubscribers,
    });

    console.log(`[PluginRegistry] Loaded plugin "${plugin.manifest.name}" v${plugin.manifest.version}`);
  }

  /**
   * Unload a single plugin
   */
  private async unloadPlugin(id: string): Promise<void> {
    const loaded = this.loadedPlugins.get(id);
    if (!loaded) return;

    // Unsubscribe from events
    loaded.eventUnsubscribers.forEach((unsub) => unsub());

    // Call cleanup
    try {
      loaded.cleanup();
    } catch (error) {
      console.error(`[PluginRegistry] Error during cleanup of plugin "${id}":`, error);
    }

    this.loadedPlugins.delete(id);
    console.log(`[PluginRegistry] Unloaded plugin "${id}"`);
  }

  /**
   * Unload all plugins
   */
  async unloadAll(): Promise<void> {
    for (const id of this.loadedPlugins.keys()) {
      await this.unloadPlugin(id);
    }
  }

  /**
   * Get all components for a specific slot from loaded plugins
   */
  getSlotComponents<T extends PluginSlotName>(slotName: T): SlotComponent<T>[] {
    const components: SlotComponent<T>[] = [];

    for (const { plugin } of this.loadedPlugins.values()) {
      const slotComponents = plugin.getSlotComponents();
      const matching = slotComponents.filter((sc) => sc.slot === slotName) as SlotComponent<T>[];
      components.push(...matching);
    }

    // Sort by priority (higher first)
    return components.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
  }

  /**
   * Check if a plugin is currently loaded
   */
  isLoaded(pluginId: string): boolean {
    return this.loadedPlugins.has(pluginId);
  }

  /**
   * Get list of loaded plugin IDs
   */
  getLoadedPlugins(): string[] {
    return Array.from(this.loadedPlugins.keys());
  }

  /**
   * Get plugin manifest if loaded
   */
  getPluginManifest(pluginId: string) {
    return this.loadedPlugins.get(pluginId)?.plugin.manifest ?? null;
  }
}

// Singleton instance
export const PluginRegistry = new PluginRegistryImpl();

// Export type for testing
export type { PluginRegistryImpl };
