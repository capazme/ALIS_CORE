/**
 * Tests for ConsentSelector component
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConsentSelector } from './ConsentSelector';
import * as consentService from '../../../services/consentService';
import type { ConsentResponse, ConsentHistoryResponse } from '../../../types/api';

// Mock the consent service
vi.mock('../../../services/consentService');

const mockConsentResponse: ConsentResponse = {
  consent_level: 'basic',
  granted_at: '2026-01-31T12:00:00Z',
  available_levels: [
    {
      level: 'basic',
      emoji: '🔒',
      name: 'Base',
      description: 'Nessun dato raccolto oltre la sessione. Solo uso del sistema.',
      dataCollected: ['Nessuno'],
    },
    {
      level: 'learning',
      emoji: '📊',
      name: 'Apprendimento',
      description: 'Query anonimizzate + feedback usati per il training RLCF.',
      dataCollected: ['Query anonimizzate', 'Feedback', 'Interazioni di ricerca'],
    },
    {
      level: 'research',
      emoji: '🔬',
      name: 'Ricerca',
      description: 'Dati aggregati disponibili per analisi accademica.',
      dataCollected: ['Query anonimizzate', 'Feedback', 'Pattern di utilizzo', 'Dati aggregati per ricerca'],
    },
  ],
};

const mockHistoryResponse: ConsentHistoryResponse = {
  history: [
    {
      id: '1',
      previous_level: null,
      new_level: 'basic',
      changed_at: '2026-01-31T12:00:00Z',
    },
  ],
};

describe('ConsentSelector', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(consentService.getConsent).mockResolvedValue(mockConsentResponse);
    vi.mocked(consentService.updateConsent).mockResolvedValue({
      message: 'Consenso aggiornato con successo',
      consent_level: 'learning',
      granted_at: '2026-01-31T13:00:00Z',
      is_downgrade: false,
    });
    vi.mocked(consentService.getConsentHistory).mockResolvedValue(mockHistoryResponse);
  });

  afterEach(() => {
    cleanup();
  });

  it('renders loading state initially', () => {
    render(<ConsentSelector />);

    // Should show 3 skeleton cards
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBe(3);
  });

  it('displays all 3 consent level cards after loading', async () => {
    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Base')).toBeInTheDocument();
    });

    expect(screen.getByText('Apprendimento')).toBeInTheDocument();
    expect(screen.getByText('Ricerca')).toBeInTheDocument();
  });

  it('displays consent level descriptions', async () => {
    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText(/Nessun dato raccolto/)).toBeInTheDocument();
    });

    expect(screen.getByText(/Query anonimizzate \+ feedback/)).toBeInTheDocument();
    expect(screen.getByText(/Dati aggregati disponibili/)).toBeInTheDocument();
  });

  it('shows data collected tags for each level', async () => {
    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Nessuno')).toBeInTheDocument();
    });

    expect(screen.getAllByText('Query anonimizzate').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Feedback').length).toBeGreaterThan(0);
  });

  it('shows current consent level as selected', async () => {
    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Base')).toBeInTheDocument();
    });

    // The basic card should have a check icon (selected)
    const basicCard = screen.getByRole('button', { name: /base/i });
    expect(basicCard).toBeInTheDocument();
  });

  it('shows last update timestamp', async () => {
    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText(/Ultimo aggiornamento:/)).toBeInTheDocument();
    });
  });

  it('allows upgrading consent level without confirmation', async () => {
    const onConsentChange = vi.fn();
    const user = userEvent.setup();

    render(<ConsentSelector onConsentChange={onConsentChange} />);

    await waitFor(() => {
      expect(screen.getByText('Apprendimento')).toBeInTheDocument();
    });

    // Click on Learning (upgrade from basic)
    const learningCard = screen.getByRole('button', { name: /apprendimento/i });
    await user.click(learningCard);

    await waitFor(() => {
      expect(consentService.updateConsent).toHaveBeenCalledWith('learning');
    });

    await waitFor(() => {
      expect(onConsentChange).toHaveBeenCalledWith('learning');
    });
  });

  it('shows success message after consent update', async () => {
    const user = userEvent.setup();

    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Apprendimento')).toBeInTheDocument();
    });

    const learningCard = screen.getByRole('button', { name: /apprendimento/i });
    await user.click(learningCard);

    await waitFor(() => {
      expect(screen.getByText('Consenso aggiornato con successo')).toBeInTheDocument();
    });
  });

  it('shows confirmation dialog when downgrading consent', async () => {
    // Start with research level
    vi.mocked(consentService.getConsent).mockResolvedValue({
      ...mockConsentResponse,
      consent_level: 'research',
    });

    const user = userEvent.setup();

    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Base')).toBeInTheDocument();
    });

    // Click on Basic (downgrade from research)
    const basicCard = screen.getByRole('button', { name: /base/i });
    await user.click(basicCard);

    // Should show confirmation dialog
    await waitFor(() => {
      expect(screen.getByText(/Confermi la riduzione del consenso/)).toBeInTheDocument();
    });
  });

  it('cancels downgrade when clicking Annulla', async () => {
    vi.mocked(consentService.getConsent).mockResolvedValue({
      ...mockConsentResponse,
      consent_level: 'research',
    });

    const user = userEvent.setup();

    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Base')).toBeInTheDocument();
    });

    // Click on Basic (downgrade)
    const basicCard = screen.getByRole('button', { name: /base/i });
    await user.click(basicCard);

    // Click cancel
    const cancelButton = screen.getByRole('button', { name: /annulla/i });
    await user.click(cancelButton);

    // Confirmation should disappear
    await waitFor(() => {
      expect(screen.queryByText(/Confermi la riduzione/)).not.toBeInTheDocument();
    });

    // API should not have been called
    expect(consentService.updateConsent).not.toHaveBeenCalled();
  });

  it('proceeds with downgrade when confirmed', async () => {
    vi.mocked(consentService.getConsent).mockResolvedValue({
      ...mockConsentResponse,
      consent_level: 'research',
    });
    vi.mocked(consentService.updateConsent).mockResolvedValue({
      message: 'Consenso aggiornato con successo',
      warning: 'Hai ridotto il tuo livello di consenso.',
      consent_level: 'basic',
      granted_at: '2026-01-31T13:00:00Z',
      is_downgrade: true,
    });

    const user = userEvent.setup();

    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Base')).toBeInTheDocument();
    });

    // Click on Basic (downgrade)
    const basicCard = screen.getByRole('button', { name: /base/i });
    await user.click(basicCard);

    // Click confirm
    const confirmButton = screen.getByRole('button', { name: /conferma/i });
    await user.click(confirmButton);

    await waitFor(() => {
      expect(consentService.updateConsent).toHaveBeenCalledWith('basic');
    });
  });

  it('shows error message when API fails', async () => {
    vi.mocked(consentService.updateConsent).mockRejectedValue(
      new Error('Network error')
    );

    const user = userEvent.setup();

    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Apprendimento')).toBeInTheDocument();
    });

    const learningCard = screen.getByRole('button', { name: /apprendimento/i });
    await user.click(learningCard);

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });

  it('does not call API when clicking already selected consent', async () => {
    const user = userEvent.setup();

    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Base')).toBeInTheDocument();
    });

    // Click on already selected Basic
    const basicCard = screen.getByRole('button', { name: /base/i });
    await user.click(basicCard);

    // Should not call updateConsent
    expect(consentService.updateConsent).not.toHaveBeenCalled();
  });

  it('toggles history section when clicking', async () => {
    const user = userEvent.setup();

    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Base')).toBeInTheDocument();
    });

    // Click on history toggle
    const historyToggle = screen.getByText('Cronologia modifiche');
    await user.click(historyToggle);

    // History should load
    await waitFor(() => {
      expect(consentService.getConsentHistory).toHaveBeenCalled();
    });
  });

  it('uses fallback levels when getConsent fails', async () => {
    vi.mocked(consentService.getConsent).mockRejectedValue(new Error('API error'));

    render(<ConsentSelector />);

    await waitFor(() => {
      expect(screen.getByText('Base')).toBeInTheDocument();
    });

    expect(screen.getByText('Apprendimento')).toBeInTheDocument();
    expect(screen.getByText('Ricerca')).toBeInTheDocument();
  });
});
