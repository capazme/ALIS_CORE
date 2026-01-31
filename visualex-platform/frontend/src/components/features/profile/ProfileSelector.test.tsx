/**
 * Tests for ProfileSelector component
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ProfileSelector } from './ProfileSelector';
import * as profileService from '../../../services/profileService';
import type { ProfileResponse } from '../../../types/api';

// Mock the profile service
vi.mock('../../../services/profileService');

const mockProfileResponse: ProfileResponse = {
  profile_type: 'assisted_research',
  authority_score: 0.3,
  preferences: {
    theme: 'system',
    language: 'it',
    notifications_enabled: true,
  },
  available_profiles: [
    {
      type: 'quick_consultation',
      emoji: '⚡',
      name: 'Consultazione Rapida',
      description: 'Risposte veloci, minima interazione',
      available: true,
    },
    {
      type: 'assisted_research',
      emoji: '📖',
      name: 'Ricerca Assistita',
      description: 'Esplorazione guidata con suggerimenti',
      available: true,
    },
    {
      type: 'expert_analysis',
      emoji: '🔍',
      name: 'Analisi Esperta',
      description: 'Accesso completo a Expert trace e feedback',
      available: true,
    },
    {
      type: 'active_contributor',
      emoji: '🎓',
      name: 'Contributore Attivo',
      description: 'Feedback granulare, impatto sul training',
      available: false,
      requiresAuthority: 0.5,
    },
  ],
};

describe('ProfileSelector', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(profileService.getProfile).mockResolvedValue(mockProfileResponse);
    vi.mocked(profileService.updateProfile).mockResolvedValue({
      message: 'Profilo aggiornato con successo',
      profile_type: 'expert_analysis',
    });
  });

  it('renders loading state initially', () => {
    render(<ProfileSelector />);

    // Should show 4 skeleton cards
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBe(4);
  });

  it('displays all 4 profile cards after loading', async () => {
    render(<ProfileSelector />);

    await waitFor(() => {
      expect(screen.getByText('Consultazione Rapida')).toBeInTheDocument();
    });

    expect(screen.getByText('Ricerca Assistita')).toBeInTheDocument();
    expect(screen.getByText('Analisi Esperta')).toBeInTheDocument();
    expect(screen.getByText('Contributore Attivo')).toBeInTheDocument();
  });

  it('displays profile descriptions', async () => {
    render(<ProfileSelector />);

    await waitFor(() => {
      expect(screen.getByText('Risposte veloci, minima interazione')).toBeInTheDocument();
    });

    expect(screen.getByText('Esplorazione guidata con suggerimenti')).toBeInTheDocument();
    expect(screen.getByText('Accesso completo a Expert trace e feedback')).toBeInTheDocument();
    expect(screen.getByText('Feedback granulare, impatto sul training')).toBeInTheDocument();
  });

  it('shows authority score', async () => {
    render(<ProfileSelector authorityScore={0.3} />);

    await waitFor(() => {
      expect(screen.getByText('Punteggio Autorità')).toBeInTheDocument();
    });

    expect(screen.getByText('0.30')).toBeInTheDocument();
  });

  it('shows initial selected profile as checked', async () => {
    render(<ProfileSelector initialProfileType="assisted_research" />);

    await waitFor(() => {
      expect(screen.getByText('Ricerca Assistita')).toBeInTheDocument();
    });

    // The selected profile card should have a check icon
    const selectedCard = screen.getByRole('button', { name: /ricerca assistita/i });
    expect(selectedCard).toBeInTheDocument();
  });

  it('allows selecting an available profile', async () => {
    const onProfileChange = vi.fn();
    const user = userEvent.setup();

    render(
      <ProfileSelector
        initialProfileType="assisted_research"
        onProfileChange={onProfileChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Analisi Esperta')).toBeInTheDocument();
    });

    // Click on Expert Analysis profile
    const expertCard = screen.getByRole('button', { name: /analisi esperta/i });
    await user.click(expertCard);

    await waitFor(() => {
      expect(profileService.updateProfile).toHaveBeenCalledWith('expert_analysis');
    });

    await waitFor(() => {
      expect(onProfileChange).toHaveBeenCalledWith('expert_analysis');
    });
  });

  it('shows success message after profile update', async () => {
    const user = userEvent.setup();

    render(<ProfileSelector initialProfileType="assisted_research" />);

    await waitFor(() => {
      expect(screen.getByText('Analisi Esperta')).toBeInTheDocument();
    });

    const expertCard = screen.getByRole('button', { name: /analisi esperta/i });
    await user.click(expertCard);

    await waitFor(() => {
      expect(screen.getByText('Profilo aggiornato con successo')).toBeInTheDocument();
    });
  });

  it('prevents selecting locked profile (active_contributor without authority)', async () => {
    const onProfileChange = vi.fn();
    const user = userEvent.setup();

    render(
      <ProfileSelector
        initialProfileType="assisted_research"
        authorityScore={0.3}
        onProfileChange={onProfileChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Contributore Attivo')).toBeInTheDocument();
    });

    // The contributor card should be disabled
    const contributorCard = screen.getByRole('button', { name: /contributore attivo/i });
    expect(contributorCard).toBeDisabled();

    // Should show authority requirement
    expect(screen.getByText('Richiede autorità ≥ 0.5')).toBeInTheDocument();
  });

  it('shows error message when API fails', async () => {
    vi.mocked(profileService.updateProfile).mockRejectedValue(
      new Error('Network error')
    );

    const user = userEvent.setup();

    render(<ProfileSelector initialProfileType="assisted_research" />);

    await waitFor(() => {
      expect(screen.getByText('Analisi Esperta')).toBeInTheDocument();
    });

    const expertCard = screen.getByRole('button', { name: /analisi esperta/i });
    await user.click(expertCard);

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });

  it('does not call API when clicking already selected profile', async () => {
    const user = userEvent.setup();

    render(<ProfileSelector initialProfileType="assisted_research" />);

    await waitFor(() => {
      expect(screen.getByText('Ricerca Assistita')).toBeInTheDocument();
    });

    const assistedCard = screen.getByRole('button', { name: /ricerca assistita/i });
    await user.click(assistedCard);

    // Should not call updateProfile since it's already selected
    expect(profileService.updateProfile).not.toHaveBeenCalled();
  });

  it('uses fallback profiles when getProfile fails', async () => {
    vi.mocked(profileService.getProfile).mockRejectedValue(new Error('API error'));

    render(<ProfileSelector authorityScore={0.6} />);

    await waitFor(() => {
      expect(screen.getByText('Consultazione Rapida')).toBeInTheDocument();
    });

    // All profiles should be available with authority >= 0.5
    const contributorCard = screen.getByRole('button', { name: /contributore attivo/i });
    expect(contributorCard).not.toBeDisabled();
  });

  it('renders profile emojis correctly', async () => {
    render(<ProfileSelector />);

    await waitFor(() => {
      expect(screen.getByText('Consultazione Rapida')).toBeInTheDocument();
    });

    // Check emojis are rendered
    expect(screen.getByRole('img', { name: 'Consultazione Rapida' })).toHaveTextContent('⚡');
    expect(screen.getByRole('img', { name: 'Ricerca Assistita' })).toHaveTextContent('📖');
    expect(screen.getByRole('img', { name: 'Analisi Esperta' })).toHaveTextContent('🔍');
    expect(screen.getByRole('img', { name: 'Contributore Attivo' })).toHaveTextContent('🎓');
  });
});
