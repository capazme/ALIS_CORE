/**
 * Tests for AuthorityScoreDisplay component
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AuthorityScoreDisplay } from './AuthorityScoreDisplay';
import * as authorityService from '../../../services/authorityService';
import type { AuthorityResponse } from '../../../types/api';

// Mock the authority service
vi.mock('../../../services/authorityService');

const mockAuthorityResponse: AuthorityResponse = {
  authority_score: 0.15,
  feedback_count: 5,
  updated_at: '2026-01-31T12:00:00Z',
  components: {
    baseline: {
      score: 0.2,
      weighted: 0.06,
      name: 'Credenziali Base',
      description: 'Punteggio basato sul tuo profilo e livello di esperienza.',
      weight: 0.3,
      icon: '🎓',
    },
    track_record: {
      score: 0.1,
      weighted: 0.05,
      name: 'Storico Feedback',
      description: 'Misura l\'accuratezza storica dei tuoi feedback.',
      weight: 0.5,
      icon: '📊',
    },
    recent_performance: {
      score: 0.2,
      weighted: 0.04,
      name: 'Performance Recente',
      description: 'Qualità dei tuoi ultimi feedback.',
      weight: 0.2,
      icon: '⚡',
    },
  },
};

const mockNewUserResponse: AuthorityResponse = {
  ...mockAuthorityResponse,
  feedback_count: 0,
  message: 'Contribuisci feedback per aumentare la tua autorità',
};

describe('AuthorityScoreDisplay', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(authorityService.getAuthority).mockResolvedValue(mockAuthorityResponse);
  });

  afterEach(() => {
    cleanup();
  });

  it('renders loading state initially', () => {
    render(<AuthorityScoreDisplay />);

    // Should show skeleton loading
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('displays main authority score after loading', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('Punteggio Autorità')).toBeInTheDocument();
    });

    // Main score should be displayed (15% = 0.15 * 100)
    expect(screen.getByText('15')).toBeInTheDocument();
    expect(screen.getByText('/100')).toBeInTheDocument();
  });

  it('displays feedback count', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('5 feedback')).toBeInTheDocument();
    });
  });

  it('displays all three component cards', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('Credenziali Base')).toBeInTheDocument();
    });

    expect(screen.getByText('Storico Feedback')).toBeInTheDocument();
    expect(screen.getByText('Performance Recente')).toBeInTheDocument();
  });

  it('displays component scores as percentages', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      // Baseline score: 0.2 * 100 = 20%
      const percentages = screen.getAllByText('20%');
      expect(percentages.length).toBeGreaterThan(0);
    });
  });

  it('displays component weights', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('(30% peso)')).toBeInTheDocument();
    });

    expect(screen.getByText('(50% peso)')).toBeInTheDocument();
    expect(screen.getByText('(20% peso)')).toBeInTheDocument();
  });

  it('shows message for new users with no feedback', async () => {
    vi.mocked(authorityService.getAuthority).mockResolvedValue(mockNewUserResponse);

    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('Contribuisci feedback per aumentare la tua autorità')).toBeInTheDocument();
    });
  });

  it('does not show message for users with feedback', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('5 feedback')).toBeInTheDocument();
    });

    expect(screen.queryByText('Contribuisci feedback per aumentare la tua autorità')).not.toBeInTheDocument();
  });

  it('has clickable component cards', async () => {
    const user = userEvent.setup();

    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('Credenziali Base')).toBeInTheDocument();
    });

    // Check that component cards are buttons
    const baselineCard = screen.getByText('Credenziali Base').closest('button');
    expect(baselineCard).toBeInTheDocument();
    expect(baselineCard).toHaveAttribute('type', 'button');
  });

  it('displays formula explanation', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText(/Formula: A = 0.3 × Baseline/)).toBeInTheDocument();
    });
  });

  it('shows error message when API fails', async () => {
    vi.mocked(authorityService.getAuthority).mockRejectedValue(
      new Error('Network error')
    );

    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });

  it('displays component icons', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('🎓')).toBeInTheDocument();
      expect(screen.getByText('📊')).toBeInTheDocument();
      expect(screen.getByText('⚡')).toBeInTheDocument();
    });
  });

  it('displays Composizione del punteggio heading', async () => {
    render(<AuthorityScoreDisplay />);

    await waitFor(() => {
      expect(screen.getByText('Composizione del punteggio')).toBeInTheDocument();
    });
  });
});
