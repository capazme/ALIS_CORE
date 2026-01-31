/**
 * Tests for useAuth hook
 *
 * Tests the actual implementation:
 * - Initial authentication state from localStorage
 * - Login flow (calls authService.login, then getCurrentUser)
 * - Logout flow (clears tokens, redirects)
 * - Register flow with auto-login
 * - Error handling
 */
import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock authService - must be hoisted
vi.mock('@/services/authService', () => ({
  login: vi.fn(),
  logout: vi.fn(),
  register: vi.fn(),
  getCurrentUser: vi.fn(),
  isAuthenticated: vi.fn(),
  changePassword: vi.fn(),
}));

// Mock appStore
vi.mock('@/store/useAppStore', () => ({
  appStore: {
    getState: () => ({
      isDataLoaded: true,
      isLoadingData: false,
      fetchUserData: vi.fn(),
      clearUserData: vi.fn(),
    }),
  },
}));

// Import mocked module
import * as authService from '@/services/authService';
import { useAuth } from '@/hooks/useAuth';

// Create a simple localStorage mock data store
const localStorageStore: Record<string, string> = {};

describe('useAuth', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Clear localStorage store
    Object.keys(localStorageStore).forEach((key) => delete localStorageStore[key]);

    // Reset mocks to default behavior
    vi.mocked(authService.isAuthenticated).mockReturnValue(false);
    vi.mocked(authService.getCurrentUser).mockResolvedValue({
      id: '',
      email: '',
      username: '',
      is_admin: false,
      is_merlt_enabled: false,
      profile_type: 'assisted_research',
      authority_score: 0,
      last_login: '',
      created_at: '',
      updated_at: '',
    });
  });

  it('starts with unauthenticated state when no token', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(false);

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it('loads user when token exists in localStorage', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(true);
    vi.mocked(authService.getCurrentUser).mockResolvedValue({
      id: '1',
      email: 'test@example.com',
      username: 'testuser',
      is_admin: false,
      is_merlt_enabled: false,
      profile_type: 'assisted_research',
      authority_score: 0,
      last_login: '',
      created_at: '',
      updated_at: '',
    });

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.email).toBe('test@example.com');
    expect(authService.getCurrentUser).toHaveBeenCalled();
  });

  it('handles successful login', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(false);
    vi.mocked(authService.login).mockResolvedValue({
      access_token: 'test-access-token',
      refresh_token: 'test-refresh-token',
    });
    vi.mocked(authService.getCurrentUser).mockResolvedValue({
      id: '1',
      email: 'test@example.com',
      username: 'testuser',
      is_admin: false,
      is_merlt_enabled: false,
      profile_type: 'assisted_research',
      authority_score: 0,
      last_login: '',
      created_at: '',
      updated_at: '',
    });

    const { result } = renderHook(() => useAuth());

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Perform login
    await act(async () => {
      await result.current.login('test@example.com', 'password123');
    });

    expect(authService.login).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'password123',
    });
    expect(authService.getCurrentUser).toHaveBeenCalled();
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.email).toBe('test@example.com');
  });

  it('handles login error', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(false);
    vi.mocked(authService.login).mockRejectedValue(new Error('Invalid credentials'));

    const { result } = renderHook(() => useAuth());

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Perform login (should throw)
    await act(async () => {
      try {
        await result.current.login('wrong@example.com', 'wrongpassword');
      } catch {
        // Expected error
      }
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.error).toBe('Invalid credentials');
  });

  it('handles logout', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(false);
    vi.mocked(authService.login).mockResolvedValue({
      access_token: 'test-token',
      refresh_token: 'test-refresh',
    });
    vi.mocked(authService.getCurrentUser).mockResolvedValue({
      id: '1',
      email: 'test@example.com',
      username: 'testuser',
      is_admin: false,
      is_merlt_enabled: false,
      profile_type: 'assisted_research',
      authority_score: 0,
      last_login: '',
      created_at: '',
      updated_at: '',
    });

    const { result } = renderHook(() => useAuth());

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Login first
    await act(async () => {
      await result.current.login('test@example.com', 'password');
    });

    expect(result.current.isAuthenticated).toBe(true);

    // Then logout
    act(() => {
      result.current.logout();
    });

    expect(authService.logout).toHaveBeenCalled();
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it('handles register with auto-login', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(false);
    vi.mocked(authService.register).mockResolvedValue({ message: 'Registration successful' });
    vi.mocked(authService.login).mockResolvedValue({
      access_token: 'new-token',
      refresh_token: 'new-refresh',
    });
    vi.mocked(authService.getCurrentUser).mockResolvedValue({
      id: '2',
      email: 'newuser@example.com',
      username: 'newuser',
      is_admin: false,
      is_merlt_enabled: false,
      profile_type: 'assisted_research',
      authority_score: 0,
      last_login: '',
      created_at: '',
      updated_at: '',
    });

    const { result } = renderHook(() => useAuth());

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Register
    await act(async () => {
      await result.current.register('newuser@example.com', 'newuser', 'password123');
    });

    expect(authService.register).toHaveBeenCalledWith({
      email: 'newuser@example.com',
      username: 'newuser',
      password: 'password123',
    });
    expect(authService.login).toHaveBeenCalled();
    expect(result.current.isAuthenticated).toBe(true);
  });

  it('handles failed user load gracefully', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(true);
    vi.mocked(authService.getCurrentUser).mockRejectedValue(new Error('Token expired'));

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
    expect(result.current.error).toBe('Token expired');
  });

  it('clears error with clearError', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(false);
    vi.mocked(authService.login).mockRejectedValue(new Error('Login failed'));

    const { result } = renderHook(() => useAuth());

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Cause an error
    await act(async () => {
      try {
        await result.current.login('test@example.com', 'password');
      } catch {
        // Expected
      }
    });

    expect(result.current.error).toBe('Login failed');

    // Clear the error
    act(() => {
      result.current.clearError();
    });

    expect(result.current.error).toBeNull();
  });

  it('returns isAdmin flag from user', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(true);
    vi.mocked(authService.getCurrentUser).mockResolvedValue({
      id: '1',
      email: 'admin@example.com',
      username: 'admin',
      is_admin: true,
      is_merlt_enabled: false,
      profile_type: 'assisted_research',
      authority_score: 0,
      last_login: '',
      created_at: '',
      updated_at: '',
    });

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.isAdmin).toBe(true);
  });

  it('returns isMerltEnabled flag from user', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(true);
    vi.mocked(authService.getCurrentUser).mockResolvedValue({
      id: '1',
      email: 'merlt@example.com',
      username: 'merltuser',
      is_admin: false,
      is_merlt_enabled: true,
      profile_type: 'assisted_research',
      authority_score: 0,
      last_login: '',
      created_at: '',
      updated_at: '',
    });

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.isMerltEnabled).toBe(true);
  });
});
