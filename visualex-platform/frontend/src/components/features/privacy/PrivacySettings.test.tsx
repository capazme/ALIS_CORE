/**
 * Tests for PrivacySettings component
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PrivacySettings } from './PrivacySettings';
import * as privacyService from '../../../services/privacyService';
import type { PrivacyStatusResponse, DataExport } from '../../../types/api';

// Mock the privacy service
vi.mock('../../../services/privacyService');

const mockStatusActive: PrivacyStatusResponse = {
  deletion_pending: false,
  deletion_requested_at: null,
  deletion_reason: null,
  days_remaining: null,
  account_active: true,
  consent_level: 'learning',
};

const mockStatusPendingDeletion: PrivacyStatusResponse = {
  deletion_pending: true,
  deletion_requested_at: '2026-01-31T12:00:00Z',
  deletion_reason: 'Test reason',
  days_remaining: 25,
  account_active: false,
  consent_level: 'learning',
};

const mockDataExport: DataExport = {
  exported_at: '2026-01-31T12:00:00Z',
  gdpr_reference: 'Art. 20 GDPR',
  user: {
    id: 'test-id',
    email: 'test@example.com',
    username: 'testuser',
    profile_type: 'assisted_research',
    is_verified: true,
    created_at: '2026-01-01T00:00:00Z',
    last_login_at: '2026-01-31T10:00:00Z',
    login_count: 10,
  },
  preferences: {
    theme: 'dark',
    language: 'it',
    notifications_enabled: true,
  },
  consent: {
    current_level: 'learning',
    granted_at: '2026-01-15T00:00:00Z',
    history: [],
  },
  authority: {
    score: 0.15,
    baseline: 0.2,
    track_record: 0.1,
    recent_performance: 0.15,
    feedback_count: 5,
    updated_at: '2026-01-30T00:00:00Z',
  },
};

describe('PrivacySettings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(privacyService.getPrivacyStatus).mockResolvedValue(mockStatusActive);
    vi.mocked(privacyService.exportData).mockResolvedValue(mockDataExport);
    vi.mocked(privacyService.requestDeletion).mockResolvedValue({
      message: 'Richiesta ricevuta',
      deletion_requested_at: '2026-01-31T12:00:00Z',
      grace_period_days: 30,
      warning: 'Il tuo account sarà eliminato',
    });
    vi.mocked(privacyService.cancelDeletion).mockResolvedValue({
      message: 'Cancellazione annullata',
      account_status: 'active',
    });

    // Mock URL.createObjectURL and related
    global.URL.createObjectURL = vi.fn(() => 'blob:test');
    global.URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders loading state initially', async () => {
    render(<PrivacySettings />);

    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThan(0);

    // Wait for async operations to complete to avoid act() warnings
    await waitFor(() => {
      expect(screen.getByText('Esporta i tuoi dati')).toBeInTheDocument();
    });
  });

  it('displays export section after loading', async () => {
    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Esporta i tuoi dati')).toBeInTheDocument();
    });
  });

  it('displays delete section after loading', async () => {
    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Elimina account')).toBeInTheDocument();
    });
  });

  it('shows GDPR Art. 20 reference in export section', async () => {
    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText(/GDPR Art. 20/)).toBeInTheDocument();
    });
  });

  it('shows GDPR Art. 17 reference in delete section', async () => {
    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText(/GDPR Art. 17/)).toBeInTheDocument();
    });
  });

  it('has export button', async () => {
    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Scarica i miei dati')).toBeInTheDocument();
    });
  });

  it('has delete button', async () => {
    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Elimina il mio account')).toBeInTheDocument();
    });
  });

  it('opens delete confirmation modal on button click', async () => {
    const user = userEvent.setup();

    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Elimina il mio account')).toBeInTheDocument();
    });

    const deleteButton = screen.getByText('Elimina il mio account');
    await user.click(deleteButton);

    await waitFor(() => {
      expect(screen.getByText('Conferma eliminazione')).toBeInTheDocument();
    });
  });

  it('shows warning in delete modal', async () => {
    const user = userEvent.setup();

    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Elimina il mio account')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Elimina il mio account'));

    await waitFor(() => {
      expect(screen.getByText(/questa azione è irreversibile/)).toBeInTheDocument();
    });
  });

  it('shows pending deletion warning when deletion is pending', async () => {
    vi.mocked(privacyService.getPrivacyStatus).mockResolvedValue(mockStatusPendingDeletion);

    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Cancellazione account in corso')).toBeInTheDocument();
    });

    expect(screen.getByText(/25 giorni/)).toBeInTheDocument();
  });

  it('shows cancel deletion button when deletion is pending', async () => {
    vi.mocked(privacyService.getPrivacyStatus).mockResolvedValue(mockStatusPendingDeletion);

    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Annulla cancellazione')).toBeInTheDocument();
    });
  });

  it('disables delete button when deletion is pending', async () => {
    vi.mocked(privacyService.getPrivacyStatus).mockResolvedValue(mockStatusPendingDeletion);

    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Cancellazione già richiesta')).toBeInTheDocument();
    });

    const deleteButton = screen.getByText('Cancellazione già richiesta');
    expect(deleteButton).toBeDisabled();
  });

  it('calls exportData when export button clicked', async () => {
    const user = userEvent.setup();

    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Scarica i miei dati')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Scarica i miei dati'));

    await waitFor(() => {
      expect(privacyService.exportData).toHaveBeenCalled();
    });
  });

  it('shows success message after export', async () => {
    const user = userEvent.setup();

    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Scarica i miei dati')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Scarica i miei dati'));

    await waitFor(() => {
      expect(screen.getByText('Dati esportati con successo')).toBeInTheDocument();
    });
  });

  it('closes modal when cancel button clicked', async () => {
    const user = userEvent.setup();

    render(<PrivacySettings />);

    await waitFor(() => {
      expect(screen.getByText('Elimina il mio account')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Elimina il mio account'));

    await waitFor(() => {
      expect(screen.getByText('Conferma eliminazione')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Annulla' }));

    await waitFor(() => {
      expect(screen.queryByText('Conferma eliminazione')).not.toBeInTheDocument();
    });
  });
});
