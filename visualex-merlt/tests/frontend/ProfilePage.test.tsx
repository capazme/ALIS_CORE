/**
 * Tests for ProfilePage component
 */
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the profile service
vi.mock('../../frontend/src/services/merltService', () => ({
  fetchUserProfile: vi.fn().mockResolvedValue({
    user: {
      id: 'user-1',
      email: 'test@example.com',
      name: 'Test User',
    },
    authority: {
      overall: 0.75,
      domains: {
        civil: 0.8,
        criminal: 0.6,
        administrative: 0.7,
      },
    },
    contributions: {
      total: 150,
      validated: 120,
      rejected: 10,
      pending: 20,
    },
  }),
}));

describe('ProfilePage', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    vi.clearAllMocks();
  });

  const renderWithProviders = (component: React.ReactElement) => {
    return render(
      <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
    );
  };

  it('renders user profile information', async () => {
    const { ProfilePage } = await import(
      '../../frontend/src/components/merlt/profile/ProfilePage'
    );

    renderWithProviders(<ProfilePage userId="user-1" />);

    await waitFor(() => {
      expect(screen.getByText('Test User')).toBeInTheDocument();
    });
  });

  it('displays authority score', async () => {
    const { ProfilePage } = await import(
      '../../frontend/src/components/merlt/profile/ProfilePage'
    );

    renderWithProviders(<ProfilePage userId="user-1" />);

    await waitFor(() => {
      expect(screen.getByText(/75%|0.75/)).toBeInTheDocument();
    });
  });

  it('shows contribution statistics', async () => {
    const { ProfilePage } = await import(
      '../../frontend/src/components/merlt/profile/ProfilePage'
    );

    renderWithProviders(<ProfilePage userId="user-1" />);

    await waitFor(() => {
      expect(screen.getByText(/150/)).toBeInTheDocument(); // Total contributions
    });
  });

  it('displays domain authority breakdown', async () => {
    const { ProfilePage } = await import(
      '../../frontend/src/components/merlt/profile/ProfilePage'
    );

    renderWithProviders(<ProfilePage userId="user-1" />);

    await waitFor(() => {
      expect(screen.getByText(/civil|civile/i)).toBeInTheDocument();
      expect(screen.getByText(/criminal|penale/i)).toBeInTheDocument();
    });
  });
});
