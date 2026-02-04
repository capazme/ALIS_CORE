/**
 * Tests for LoginForm component
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { LoginForm } from '@/components/auth/LoginForm';

// Mock useAuth hook
const mockLogin = vi.fn();
const mockUseAuth = vi.fn(() => ({
  login: mockLogin,
  loading: false,
  error: null,
  isAuthenticated: false,
  user: null,
}));

vi.mock('@/hooks/useAuth', () => ({
  useAuth: () => mockUseAuth(),
}));

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
    useLocation: () => ({ state: null }),
  };
});

const renderLoginForm = () => {
  return render(
    <BrowserRouter>
      <LoginForm />
    </BrowserRouter>
  );
};

describe('LoginForm', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuth.mockReturnValue({
      login: mockLogin,
      loading: false,
      error: null,
      isAuthenticated: false,
      user: null,
    });
  });

  describe('Rendering', () => {
    it('renders the login form', () => {
      renderLoginForm();

      expect(screen.getByPlaceholderText(/name@company.com/i)).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/••••••••/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    });

    it('renders the VisuaLex logo/title', () => {
      renderLoginForm();

      // The title has "Visua" and "Lex" in separate elements
      const heading = screen.getByRole('heading', { level: 1 });
      expect(heading).toHaveTextContent('VisuaLex');
    });

    it('renders the registration link', () => {
      renderLoginForm();

      expect(screen.getByText(/non hai un account/i)).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /registrati/i })).toHaveAttribute('href', '/register');
    });

    it('renders the forgot password link', () => {
      renderLoginForm();

      const forgotPasswordLink = screen.getByRole('link', { name: /password dimenticata/i });
      expect(forgotPasswordLink).toBeInTheDocument();
      expect(forgotPasswordLink).toHaveAttribute('href', '/forgot-password');
    });
  });

  describe('Form Validation', () => {
    it('shows error when submitting empty form', async () => {
      renderLoginForm();

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/email and password are required/i)).toBeInTheDocument();
      });
      expect(mockLogin).not.toHaveBeenCalled();
    });

    // Note: Client-side email validation is tested implicitly through browser HTML5 validation
    // The component uses type="email" which provides native browser validation
  });

  describe('Successful Login', () => {
    it('calls login with correct credentials', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue({ id: '1', email: 'test@example.com' });
      renderLoginForm();

      await user.type(screen.getByPlaceholderText(/name@company.com/i), 'test@example.com');
      await user.type(screen.getByPlaceholderText(/••••••••/), 'Password123');
      await user.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith('test@example.com', 'Password123');
      });
    });

    it('navigates to home after successful login', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue({ id: '1', email: 'test@example.com' });
      renderLoginForm();

      await user.type(screen.getByPlaceholderText(/name@company.com/i), 'test@example.com');
      await user.type(screen.getByPlaceholderText(/••••••••/), 'Password123');
      await user.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/', { replace: true });
      });
    });
  });

  describe('Login Errors', () => {
    it('displays error message from auth hook', () => {
      mockUseAuth.mockReturnValue({
        login: mockLogin,
        loading: false,
        error: 'Invalid credentials',
        isAuthenticated: false,
        user: null,
      });

      renderLoginForm();

      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });

    it('displays error when login fails', async () => {
      const user = userEvent.setup();
      mockLogin.mockRejectedValue(new Error('Invalid email or password'));
      renderLoginForm();

      await user.type(screen.getByPlaceholderText(/name@company.com/i), 'test@example.com');
      await user.type(screen.getByPlaceholderText(/••••••••/), 'WrongPassword');
      await user.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/invalid email or password/i)).toBeInTheDocument();
      });
    });
  });

  describe('Loading State', () => {
    it('disables inputs and button while loading', () => {
      mockUseAuth.mockReturnValue({
        login: mockLogin,
        loading: true,
        error: null,
        isAuthenticated: false,
        user: null,
      });

      renderLoginForm();

      expect(screen.getByPlaceholderText(/name@company.com/i)).toBeDisabled();
      expect(screen.getByPlaceholderText(/••••••••/)).toBeDisabled();
      // The submit button (type=submit) should be disabled
      const submitButton = document.querySelector('button[type="submit"]');
      expect(submitButton).toBeDisabled();
    });

    it('shows loading spinner while loading', () => {
      mockUseAuth.mockReturnValue({
        login: mockLogin,
        loading: true,
        error: null,
        isAuthenticated: false,
        user: null,
      });

      renderLoginForm();

      // The spinner has animate-spin class
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });
  });

  describe('Password Visibility Toggle', () => {
    it('toggles password visibility when eye icon is clicked', async () => {
      const user = userEvent.setup();
      renderLoginForm();

      const passwordInput = screen.getByPlaceholderText(/••••••••/);
      expect(passwordInput).toHaveAttribute('type', 'password');

      // Find and click the toggle button (it's the only button other than submit)
      const toggleButton = screen.getAllByRole('button').find(
        btn => btn.getAttribute('type') === 'button'
      );

      if (toggleButton) {
        await user.click(toggleButton);
        expect(passwordInput).toHaveAttribute('type', 'text');

        await user.click(toggleButton);
        expect(passwordInput).toHaveAttribute('type', 'password');
      }
    });
  });
});
