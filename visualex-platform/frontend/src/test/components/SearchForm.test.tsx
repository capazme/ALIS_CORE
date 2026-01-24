/**
 * Tests for SearchForm component
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';

// Mock the search hook
vi.mock('@/hooks/useSearch', () => ({
  useSearch: () => ({
    search: vi.fn(),
    isLoading: false,
    results: [],
    error: null,
  }),
}));

// Mock the API service
vi.mock('@/services/api', () => ({
  fetchNormaData: vi.fn().mockResolvedValue([]),
}));

describe('SearchForm', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders all form fields', async () => {
    // Lazy import to allow mocks to be set up
    const { SearchForm } = await import('@/components/features/search/SearchForm');

    render(<SearchForm />);

    // Check for act type selector
    expect(screen.getByLabelText(/tipo atto/i)).toBeInTheDocument();

    // Check for article input
    expect(screen.getByLabelText(/articolo/i)).toBeInTheDocument();

    // Check for search button
    expect(screen.getByRole('button', { name: /cerca/i })).toBeInTheDocument();
  });

  it('submits form with correct values', async () => {
    const { SearchForm } = await import('@/components/features/search/SearchForm');
    const user = userEvent.setup();

    const onSubmit = vi.fn();
    render(<SearchForm onSubmit={onSubmit} />);

    // Fill act type
    const actTypeSelect = screen.getByLabelText(/tipo atto/i);
    await user.selectOptions(actTypeSelect, 'codice civile');

    // Fill article number
    const articleInput = screen.getByLabelText(/articolo/i);
    await user.type(articleInput, '1453');

    // Submit form
    const submitButton = screen.getByRole('button', { name: /cerca/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          actType: 'codice civile',
          article: '1453',
        })
      );
    });
  });

  it('validates required fields', async () => {
    const { SearchForm } = await import('@/components/features/search/SearchForm');
    const user = userEvent.setup();

    render(<SearchForm />);

    // Try to submit without filling fields
    const submitButton = screen.getByRole('button', { name: /cerca/i });
    await user.click(submitButton);

    // Should show validation error
    await waitFor(() => {
      expect(screen.getByText(/campo obbligatorio/i)).toBeInTheDocument();
    });
  });

  it('handles article range input', async () => {
    const { SearchForm } = await import('@/components/features/search/SearchForm');
    const user = userEvent.setup();

    render(<SearchForm />);

    const articleInput = screen.getByLabelText(/articolo/i);
    await user.type(articleInput, '1-10');

    // Should accept range format
    expect(articleInput).toHaveValue('1-10');
  });

  it('shows loading state during search', async () => {
    vi.mock('@/hooks/useSearch', () => ({
      useSearch: () => ({
        search: vi.fn(),
        isLoading: true,
        results: [],
        error: null,
      }),
    }));

    const { SearchForm } = await import('@/components/features/search/SearchForm');

    render(<SearchForm />);

    // Submit button should be disabled during loading
    const submitButton = screen.getByRole('button', { name: /cerca/i });
    expect(submitButton).toBeDisabled();
  });
});
