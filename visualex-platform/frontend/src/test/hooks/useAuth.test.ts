/**
 * Tests for useAuth hook
 */
import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock API service
const mockLogin = vi.fn();
const mockLogout = vi.fn();
const mockRegister = vi.fn();

vi.mock('@/services/api', () => ({
  authApi: {
    login: (...args: unknown[]) => mockLogin(...args),
    logout: (...args: unknown[]) => mockLogout(...args),
    register: (...args: unknown[]) => mockRegister(...args),
  },
}));

describe('useAuth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it('starts with unauthenticated state', async () => {
    const { useAuth } = await import('@/hooks/useAuth');
    const { result } = renderHook(() => useAuth());

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it('handles successful login', async () => {
    mockLogin.mockResolvedValue({
      user: { id: '1', email: 'test@example.com', name: 'Test User' },
      token: 'mock-token',
    });

    const { useAuth } = await import('@/hooks/useAuth');
    const { result } = renderHook(() => useAuth());

    await act(async () => {
      await result.current.login('test@example.com', 'password');
    });

    await waitFor(() => {
      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.user?.email).toBe('test@example.com');
    });
  });

  it('handles login error', async () => {
    mockLogin.mockRejectedValue(new Error('Invalid credentials'));

    const { useAuth } = await import('@/hooks/useAuth');
    const { result } = renderHook(() => useAuth());

    await act(async () => {
      try {
        await result.current.login('wrong@example.com', 'wrong');
      } catch (e) {
        // Expected error
      }
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.error).toBeTruthy();
  });

  it('handles logout', async () => {
    mockLogout.mockResolvedValue(undefined);
    mockLogin.mockResolvedValue({
      user: { id: '1', email: 'test@example.com' },
      token: 'mock-token',
    });

    const { useAuth } = await import('@/hooks/useAuth');
    const { result } = renderHook(() => useAuth());

    // First login
    await act(async () => {
      await result.current.login('test@example.com', 'password');
    });

    expect(result.current.isAuthenticated).toBe(true);

    // Then logout
    await act(async () => {
      await result.current.logout();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it('persists auth state in localStorage', async () => {
    mockLogin.mockResolvedValue({
      user: { id: '1', email: 'test@example.com' },
      token: 'mock-token',
    });

    const { useAuth } = await import('@/hooks/useAuth');
    const { result } = renderHook(() => useAuth());

    await act(async () => {
      await result.current.login('test@example.com', 'password');
    });

    // Check localStorage
    expect(localStorage.getItem('token')).toBe('mock-token');
  });

  it('restores auth state from localStorage', async () => {
    localStorage.setItem('token', 'existing-token');
    localStorage.setItem('user', JSON.stringify({ id: '1', email: 'stored@example.com' }));

    const { useAuth } = await import('@/hooks/useAuth');
    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.user?.email).toBe('stored@example.com');
    });
  });
});
