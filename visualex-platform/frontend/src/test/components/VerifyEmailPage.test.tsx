/**
 * Tests for VerifyEmailPage component
 */
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { VerifyEmailPage } from '@/pages/VerifyEmailPage';

// Mock auth service
const mockVerifyEmail = vi.fn();

vi.mock('@/services/authService', () => ({
  verifyEmail: (...args: unknown[]) => mockVerifyEmail(...args),
}));

// Mock useSearchParams
const mockSearchParams = new URLSearchParams();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useSearchParams: () => [mockSearchParams],
  };
});

function renderVerifyEmailPage() {
  return render(
    <BrowserRouter>
      <VerifyEmailPage />
    </BrowserRouter>
  );
}

describe('VerifyEmailPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearchParams.delete('token');
  });

  describe('without token', () => {
    it('shows error when no token provided', async () => {
      renderVerifyEmailPage();

      await waitFor(() => {
        expect(screen.getByText('Errore di Verifica')).toBeInTheDocument();
      });

      expect(screen.getByText('Token di verifica mancante.')).toBeInTheDocument();
    });

    it('shows link to return to login', async () => {
      renderVerifyEmailPage();

      await waitFor(() => {
        expect(screen.getByRole('link', { name: /Torna al Login/i })).toBeInTheDocument();
      });
    });
  });

  describe('with valid token', () => {
    beforeEach(() => {
      mockSearchParams.set('token', 'valid-verification-token');
    });

    it('shows loading state initially', () => {
      mockVerifyEmail.mockImplementation(() => new Promise(() => {})); // Never resolves
      renderVerifyEmailPage();

      expect(screen.getByText('Verifica in corso...')).toBeInTheDocument();
    });

    it('shows success message on successful verification', async () => {
      mockVerifyEmail.mockResolvedValue({
        message: 'Email verified successfully',
        verified: true,
      });

      renderVerifyEmailPage();

      await waitFor(() => {
        expect(screen.getByText('Email Verificata!')).toBeInTheDocument();
      });

      expect(screen.getByText(/Il tuo account è stato attivato/)).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /Vai al Login/i })).toBeInTheDocument();
    });
  });

  describe('with expired token', () => {
    beforeEach(() => {
      mockSearchParams.set('token', 'expired-token');
      mockVerifyEmail.mockRejectedValue({
        message: 'Verification token has expired',
      });
    });

    it('shows expired message', async () => {
      renderVerifyEmailPage();

      await waitFor(() => {
        expect(screen.getByText('Link Scaduto')).toBeInTheDocument();
      });

      expect(screen.getByText(/Il link di verifica è scaduto/)).toBeInTheDocument();
    });

    it('suggests requesting new verification link', async () => {
      renderVerifyEmailPage();

      await waitFor(() => {
        expect(screen.getByText(/richiedere un nuovo link di verifica/)).toBeInTheDocument();
      });
    });
  });

  describe('with invalid token', () => {
    beforeEach(() => {
      mockSearchParams.set('token', 'invalid-token');
      mockVerifyEmail.mockRejectedValue({
        message: 'Invalid verification token',
      });
    });

    it('shows error message', async () => {
      renderVerifyEmailPage();

      await waitFor(() => {
        expect(screen.getByText('Errore di Verifica')).toBeInTheDocument();
      });

      expect(screen.getByText('Invalid verification token')).toBeInTheDocument();
    });
  });
});
