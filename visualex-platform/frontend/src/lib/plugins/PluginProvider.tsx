/**
 * PluginProvider Component
 *
 * Manages plugin lifecycle based on user authentication state.
 * Loads/unloads plugins when user logs in/out or features change.
 */

import { createContext, useContext, useEffect, useState, useCallback, useMemo } from 'react';
import type { PluginConfig, PluginManifest } from './types';
import { PluginRegistry } from './PluginRegistry';
import { EventBus } from './EventBus';

interface PluginContextValue {
  /**
   * List of currently loaded plugin IDs
   */
  loadedPlugins: string[];

  /**
   * Check if a specific plugin is loaded
   */
  isPluginLoaded: (pluginId: string) => boolean;

  /**
   * Get manifest for a loaded plugin
   */
  getPluginManifest: (pluginId: string) => PluginManifest | null;

  /**
   * Emit an event (shortcut to EventBus.emit)
   */
  emit: typeof EventBus.emit;

  /**
   * Subscribe to an event (shortcut to EventBus.on)
   */
  on: typeof EventBus.on;
}

const PluginContext = createContext<PluginContextValue | null>(null);

export interface PluginProviderProps {
  children: React.ReactNode;
  /**
   * Plugin configurations to register
   */
  plugins: PluginConfig[];
  /**
   * Current user (null if not logged in)
   */
  user: { id: string; features: string[] } | null;
  /**
   * Base URL for plugin API calls
   */
  apiBaseUrl: string;
  /**
   * Function to get current auth token
   */
  getAuthToken: () => Promise<string | null>;
}

export function PluginProvider({
  children,
  plugins,
  user,
  apiBaseUrl,
  getAuthToken,
}: PluginProviderProps): React.ReactElement {
  const [loadedPlugins, setLoadedPlugins] = useState<string[]>([]);
  const [isInitialized, setIsInitialized] = useState(false);

  // Register plugins on mount
  useEffect(() => {
    plugins.forEach((config) => {
      PluginRegistry.register(config);
    });
  }, [plugins]);

  // Configure registry when user/config changes
  useEffect(() => {
    PluginRegistry.configure({
      user,
      apiBaseUrl,
      getAuthToken,
    });
  }, [user, apiBaseUrl, getAuthToken]);

  // Load plugins when user changes
  useEffect(() => {
    let mounted = true;

    const loadPlugins = async () => {
      await PluginRegistry.loadPlugins();
      if (mounted) {
        setLoadedPlugins(PluginRegistry.getLoadedPlugins());
        setIsInitialized(true);
      }
    };

    loadPlugins();

    // Emit user events
    if (user) {
      EventBus.emit('user:logged-in', { userId: user.id, features: user.features });
    } else if (isInitialized) {
      // Only emit logged-out if we were previously initialized
      EventBus.emit('user:logged-out', { userId: '' });
    }

    return () => {
      mounted = false;
    };
  }, [user, isInitialized]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      PluginRegistry.unloadAll();
    };
  }, []);

  const isPluginLoaded = useCallback((pluginId: string) => {
    return PluginRegistry.isLoaded(pluginId);
  }, []);

  const getPluginManifest = useCallback((pluginId: string) => {
    return PluginRegistry.getPluginManifest(pluginId);
  }, []);

  const contextValue = useMemo<PluginContextValue>(
    () => ({
      loadedPlugins,
      isPluginLoaded,
      getPluginManifest,
      emit: EventBus.emit.bind(EventBus),
      on: EventBus.on.bind(EventBus),
    }),
    [loadedPlugins, isPluginLoaded, getPluginManifest]
  );

  return <PluginContext.Provider value={contextValue}>{children}</PluginContext.Provider>;
}

/**
 * Hook to access plugin context
 */
export function usePlugins(): PluginContextValue {
  const context = useContext(PluginContext);
  if (!context) {
    throw new Error('usePlugins must be used within a PluginProvider');
  }
  return context;
}

/**
 * Hook to check if a specific plugin feature is available
 */
export function usePluginFeature(pluginId: string): boolean {
  const { isPluginLoaded } = usePlugins();
  return isPluginLoaded(pluginId);
}
