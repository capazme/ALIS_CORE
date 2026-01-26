/**
 * Tests for RegisterForm component
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { RegisterForm } from '@/components/auth/RegisterForm';

// Mock services
const mockValidateInvitation = vi.fn();
const mockRegister = vi.fn();

vi.mock('@/services/invitationService', () => ({
  validateInvitation: (...args: unknown[]) => mockValidateInvitation(...args),
}));

vi.mock('@/services/authService', () => ({
  register: (...args: unknown[]) => mockRegister(...args),
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

function renderRegisterForm() {
  return render(
    <BrowserRouter>
      <RegisterForm />
    </BrowserRouter>
  );
}

describe('RegisterForm', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearchParams.delete('token');
  });

  describe('without invitation token', () => {
    it('shows invalid invitation message when no token provided', async () => {
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByText('Invito Non Valido')).toBeInTheDocument();
      });

      expect(screen.getByText(/Nessun token di invito fornito/)).toBeInTheDocument();
    });

    it('shows link to return to login', async () => {
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByRole('link', { name: /Torna al Login/i })).toBeInTheDocument();
      });
    });
  });

  describe('with valid invitation token', () => {
    beforeEach(() => {
      mockSearchParams.set('token', 'valid-token-uuid');
      mockValidateInvitation.mockResolvedValue({
        valid: true,
        expires_at: '2026-02-01T00:00:00Z',
        inviter: { username: 'john_doe' },
      });
    });

    it('shows loading state initially', () => {
      renderRegisterForm();

      expect(screen.getByText(/Verifica dell'invito in corso/)).toBeInTheDocument();
    });

    it('shows registration form after validation', async () => {
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByText('Crea il tuo account')).toBeInTheDocument();
      });

      expect(screen.getByText(/Invitato da/)).toBeInTheDocument();
      expect(screen.getByText('john_doe')).toBeInTheDocument();
    });

    it('shows all required form fields', async () => {
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByPlaceholderText('nome@email.com')).toBeInTheDocument();
      });

      expect(screen.getByPlaceholderText('mario_rossi')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Mario Rossi')).toBeInTheDocument();
      expect(screen.getAllByPlaceholderText('••••••••')).toHaveLength(2);
      expect(screen.getByRole('button', { name: /Membro/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Ricercatore/i })).toBeInTheDocument();
    });

    it('pre-fills email when invitation has specific email', async () => {
      mockValidateInvitation.mockResolvedValue({
        valid: true,
        email: 'invited@example.com',
        expires_at: '2026-02-01T00:00:00Z',
        inviter: { username: 'john_doe' },
      });

      renderRegisterForm();

      await waitFor(() => {
        const emailInput = screen.getByPlaceholderText('nome@email.com') as HTMLInputElement;
        expect(emailInput.value).toBe('invited@example.com');
        expect(emailInput.disabled).toBe(true);
      });
    });
  });

  describe('password validation', () => {
    beforeEach(() => {
      mockSearchParams.set('token', 'valid-token-uuid');
      mockValidateInvitation.mockResolvedValue({
        valid: true,
        expires_at: '2026-02-01T00:00:00Z',
      });
    });

    it('shows password requirements when typing', async () => {
      const user = userEvent.setup();
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByPlaceholderText('nome@email.com')).toBeInTheDocument();
      });

      const passwordInputs = screen.getAllByPlaceholderText('••••••••');
      await user.type(passwordInputs[0], 'test');

      expect(screen.getByText('Almeno 8 caratteri')).toBeInTheDocument();
      expect(screen.getByText('Una lettera maiuscola')).toBeInTheDocument();
      expect(screen.getByText('Una lettera minuscola')).toBeInTheDocument();
      expect(screen.getByText('Un numero')).toBeInTheDocument();
    });

    it('validates password strength indicators', async () => {
      const user = userEvent.setup();
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByPlaceholderText('nome@email.com')).toBeInTheDocument();
      });

      const passwordInputs = screen.getAllByPlaceholderText('••••••••');

      // Test weak password
      await user.type(passwordInputs[0], 'test');
      expect(screen.getByText('Almeno 8 caratteri')).toHaveClass('text-slate-500');

      // Test strong password
      await user.clear(passwordInputs[0]);
      await user.type(passwordInputs[0], 'Test1234');

      await waitFor(() => {
        expect(screen.getByText('Almeno 8 caratteri')).toHaveClass('text-green-600');
        expect(screen.getByText('Una lettera maiuscola')).toHaveClass('text-green-600');
        expect(screen.getByText('Una lettera minuscola')).toHaveClass('text-green-600');
        expect(screen.getByText('Un numero')).toHaveClass('text-green-600');
      });
    });
  });

  describe('form submission', () => {
    beforeEach(() => {
      mockSearchParams.set('token', 'valid-token-uuid');
      mockValidateInvitation.mockResolvedValue({
        valid: true,
        expires_at: '2026-02-01T00:00:00Z',
      });
    });

    it('shows validation errors for empty fields', async () => {
      const user = userEvent.setup();
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Crea Account/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /Crea Account/i }));

      expect(screen.getByText(/Email, username e password sono obbligatori/i)).toBeInTheDocument();
    });

    it('shows error for mismatched passwords', async () => {
      const user = userEvent.setup();
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByPlaceholderText('nome@email.com')).toBeInTheDocument();
      });

      await user.type(screen.getByPlaceholderText('nome@email.com'), 'test@example.com');
      await user.type(screen.getByPlaceholderText('mario_rossi'), 'testuser');

      const passwordInputs = screen.getAllByPlaceholderText('••••••••');
      await user.type(passwordInputs[0], 'Test1234');
      await user.type(passwordInputs[1], 'Different1');

      await user.click(screen.getByRole('button', { name: /Crea Account/i }));

      expect(screen.getByText(/Le password non coincidono/i)).toBeInTheDocument();
    });

    it('submits form successfully and shows email verification message', async () => {
      mockRegister.mockResolvedValue({
        message: 'Registration successful. Please check your email.',
        verification_pending: true,
      });

      const user = userEvent.setup();
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByPlaceholderText('nome@email.com')).toBeInTheDocument();
      });

      await user.type(screen.getByPlaceholderText('nome@email.com'), 'test@example.com');
      await user.type(screen.getByPlaceholderText('mario_rossi'), 'testuser');

      const passwordInputs = screen.getAllByPlaceholderText('••••••••');
      await user.type(passwordInputs[0], 'Test1234');
      await user.type(passwordInputs[1], 'Test1234');

      await user.click(screen.getByRole('button', { name: /Crea Account/i }));

      await waitFor(() => {
        expect(screen.getByText('Verifica la tua Email')).toBeInTheDocument();
      });

      expect(screen.getByText('test@example.com')).toBeInTheDocument();
      expect(mockRegister).toHaveBeenCalledWith({
        invitation_token: 'valid-token-uuid',
        email: 'test@example.com',
        username: 'testuser',
        password: 'Test1234',
        name: undefined,
        role: 'member',
      });
    });

    it('handles registration error', async () => {
      mockRegister.mockRejectedValue({
        message: 'This email is already registered',
      });

      const user = userEvent.setup();
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByPlaceholderText('nome@email.com')).toBeInTheDocument();
      });

      await user.type(screen.getByPlaceholderText('nome@email.com'), 'existing@example.com');
      await user.type(screen.getByPlaceholderText('mario_rossi'), 'testuser');

      const passwordInputs = screen.getAllByPlaceholderText('••••••••');
      await user.type(passwordInputs[0], 'Test1234');
      await user.type(passwordInputs[1], 'Test1234');

      await user.click(screen.getByRole('button', { name: /Crea Account/i }));

      await waitFor(() => {
        expect(screen.getByText('This email is already registered')).toBeInTheDocument();
      });
    });
  });

  describe('role selection', () => {
    beforeEach(() => {
      mockSearchParams.set('token', 'valid-token-uuid');
      mockValidateInvitation.mockResolvedValue({
        valid: true,
        expires_at: '2026-02-01T00:00:00Z',
      });
    });

    it('defaults to member role', async () => {
      renderRegisterForm();

      await waitFor(() => {
        const memberButton = screen.getByRole('button', { name: /Membro/i });
        expect(memberButton).toHaveClass('border-blue-500');
      });
    });

    it('allows switching to researcher role', async () => {
      const user = userEvent.setup();
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Ricercatore/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /Ricercatore/i }));

      expect(screen.getByRole('button', { name: /Ricercatore/i })).toHaveClass('border-blue-500');
      expect(screen.getByRole('button', { name: /Membro/i })).not.toHaveClass('border-blue-500');
    });
  });

  describe('expired invitation', () => {
    beforeEach(() => {
      mockSearchParams.set('token', 'expired-token');
      mockValidateInvitation.mockRejectedValue({
        message: 'This invitation has expired',
      });
    });

    it('shows expired invitation message', async () => {
      renderRegisterForm();

      await waitFor(() => {
        expect(screen.getByText('Invito Scaduto')).toBeInTheDocument();
      });
    });
  });
});
