/**
 * Tests for SearchForm component
 *
 * Tests the actual implementation:
 * - Act type selection with grouped options
 * - Article number input with increment/decrement
 * - Version selection (vigente/originale)
 * - Brocardi toggle
 * - Form submission
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { SearchForm } from '@/components/features/search/SearchForm';

// Mock the store
vi.mock('@/store/useAppStore', () => ({
  useAppStore: () => ({
    customAliases: [],
    trackAliasUsage: vi.fn(),
  }),
}));

// Mock fetch for article tree
global.fetch = vi.fn();

describe('SearchForm', () => {
  const mockOnSearch = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ norma_data: [] }),
    });
  });

  it('renders the search form', () => {
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    // Check for main elements
    expect(screen.getByText('Parametri di Estrazione')).toBeInTheDocument();
    expect(screen.getByText('Fonte Normativa')).toBeInTheDocument();
    expect(screen.getByText('Articolo')).toBeInTheDocument();
  });

  it('renders act type selector with grouped options', () => {
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    const select = screen.getByRole('combobox');
    expect(select).toBeInTheDocument();

    // Check for option groups
    expect(screen.getByText('Seleziona Atto...')).toBeInTheDocument();
  });

  it('renders article input with increment/decrement buttons', () => {
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    // Find article input
    const articleInput = screen.getByRole('textbox', { name: '' });
    expect(articleInput).toBeInTheDocument();

    // Find increment/decrement buttons
    const decrementBtn = screen.getByLabelText('Articolo precedente');
    const incrementBtn = screen.getByLabelText('Articolo successivo');
    expect(decrementBtn).toBeInTheDocument();
    expect(incrementBtn).toBeInTheDocument();
  });

  it('increments article number when clicking plus', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    const articleInput = screen.getByDisplayValue('1');
    const incrementBtn = screen.getByLabelText('Articolo successivo');

    await user.click(incrementBtn);

    expect(articleInput).toHaveValue('2');
  });

  it('decrements article number when clicking minus', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    // First increment to 2
    const incrementBtn = screen.getByLabelText('Articolo successivo');
    await user.click(incrementBtn);

    const articleInput = screen.getByDisplayValue('2');
    const decrementBtn = screen.getByLabelText('Articolo precedente');

    await user.click(decrementBtn);

    expect(articleInput).toHaveValue('1');
  });

  it('does not decrement below 1', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    const articleInput = screen.getByDisplayValue('1');
    const decrementBtn = screen.getByLabelText('Articolo precedente');

    await user.click(decrementBtn);

    expect(articleInput).toHaveValue('1');
  });

  it('renders version selection buttons', () => {
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    expect(screen.getByText('Vigente')).toBeInTheDocument();
    expect(screen.getByText('Originale')).toBeInTheDocument();
  });

  it('toggles version selection', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    const originaleBtn = screen.getByText('Originale');
    await user.click(originaleBtn);

    // Check the button is selected (has different styling)
    expect(originaleBtn.className).toContain('border-primary');
  });

  it('renders Brocardi toggle', () => {
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    expect(screen.getByText('Brocardi & Ratio')).toBeInTheDocument();
  });

  it('submits form with search parameters', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    // Select an act type
    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'codice civile');

    // Submit the form
    const submitButton = screen.getByText('Estrai Contenuto');
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockOnSearch).toHaveBeenCalledWith(
        expect.objectContaining({
          act_type: 'codice civile',
          article: '1',
        })
      );
    });
  });

  it('disables submit button when loading', () => {
    render(<SearchForm onSearch={mockOnSearch} isLoading={true} />);

    const submitButton = screen.getByRole('button', { name: /loading/i });
    expect(submitButton).toBeDisabled();
  });

  it('renders reset button', () => {
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    expect(screen.getByText('Reset')).toBeInTheDocument();
  });

  it('resets form when clicking reset', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    // Select an act type
    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'codice civile');

    // Click reset
    const resetButton = screen.getByText('Reset');
    await user.click(resetButton);

    // Check form is reset
    expect(select).toHaveValue('');
  });

  it('shows advanced options when expanded', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    // Click to expand advanced options
    const advancedButton = screen.getByText('Opzioni Avanzate');
    await user.click(advancedButton);

    await waitFor(() => {
      expect(screen.getByLabelText(/allegato/i)).toBeInTheDocument();
    });
  });

  it('disables number/date inputs for codici', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    // Select a codice (doesn't require number/date)
    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'codice civile');

    // Number and date inputs should be disabled
    const numberInput = screen.getByPlaceholderText('n.');
    const dateInput = screen.getByPlaceholderText('aaaa');

    expect(numberInput).toBeDisabled();
    expect(dateInput).toBeDisabled();
  });

  it('enables number/date inputs for leggi', async () => {
    const user = userEvent.setup();
    render(<SearchForm onSearch={mockOnSearch} isLoading={false} />);

    // Select a legge (requires number/date)
    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'legge');

    // Number and date inputs should be enabled
    const numberInput = screen.getByPlaceholderText('n.');
    const dateInput = screen.getByPlaceholderText('aaaa');

    expect(numberInput).not.toBeDisabled();
    expect(dateInput).not.toBeDisabled();
  });
});
