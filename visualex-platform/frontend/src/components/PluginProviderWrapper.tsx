/**
 * PluginProviderWrapper - Wrapper that connects PluginProvider to AuthContext
 *
 * This component bridges the authentication state with the plugin system.
 */

import type { ReactNode } from 'react';
import { PluginProvider } from '../lib/plugins';
import { useAuthContext } from '../contexts/AuthContext';
import type { PluginConfig } from '../lib/plugins/types';

interface PluginProviderWrapperProps {
  children: ReactNode;
  plugins: PluginConfig[];
}

export function PluginProviderWrapper({ children, plugins }: PluginProviderWrapperProps) {
  const { user, getAuthToken } = useAuthContext();

  // Map user to plugin context format
  const pluginUser = user
    ? {
        id: user.id,
        features: [
          ...(user.is_merlt_enabled ? ['merlt'] : []),
          ...(user.is_admin ? ['admin'] : []),
        ],
      }
    : null;

  // Get API base URL from environment
  const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';

  return (
    <PluginProvider
      plugins={plugins}
      user={pluginUser}
      apiBaseUrl={apiBaseUrl}
      getAuthToken={getAuthToken}
    >
      {children}
    </PluginProvider>
  );
}
