/**
 * AuthContext - Centralized authentication state provider
 *
 * Provides user authentication state to the entire application,
 * including the PluginProvider.
 */

import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';
import { useAuth as useAuthHook } from '../hooks/useAuth';
import { getAccessToken } from '../services/authService';
import type { UserResponse } from '../types/api';

interface AuthContextValue {
  user: UserResponse | null;
  loading: boolean;
  error: string | null;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isMerltEnabled: boolean;
  register: (email: string, username: string, password: string) => Promise<UserResponse>;
  login: (email: string, password: string) => Promise<UserResponse>;
  logout: () => void;
  changePassword: (currentPassword: string, newPassword: string) => Promise<UserResponse>;
  clearError: () => void;
  getAuthToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const auth = useAuthHook();

  // Wrap getAccessToken in async function for PluginProvider
  const getAuthToken = async (): Promise<string | null> => {
    return getAccessToken();
  };

  return (
    <AuthContext.Provider
      value={{
        ...auth,
        getAuthToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

/**
 * Hook to access auth context
 */
export function useAuthContext(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuthContext must be used within an AuthProvider');
  }
  return context;
}
