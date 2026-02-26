import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

// Mock lucide-react — explicit named exports (Proxy mock causes vitest to hang)
vi.mock('lucide-react', () => ({
  Brain: (props: Record<string, unknown>) => (
    <span data-testid="icon-Brain" className={props.className as string}>
      {props.children as React.ReactNode}
    </span>
  ),
}));

// Mock the store
const mockOpen = vi.fn();
const mockUseMerltPanelStore = vi.fn();

vi.mock('../../store/useMerltSidebarStore', () => ({
  useMerltPanelStore: (selector: (state: Record<string, unknown>) => unknown) =>
    mockUseMerltPanelStore(selector),
}));

// Mock article status hook — prevents pulling in enrichment store + SSE dependencies
const mockUseMerltArticleStatus = vi.fn();

vi.mock('../../hooks/useMerltArticleStatus', () => ({
  useMerltArticleStatus: (...args: unknown[]) => mockUseMerltArticleStatus(...args),
}));

import { MerltToolbar } from '../MerltToolbar';

describe('MerltToolbar', () => {
  const defaultProps = {
    urn: 'urn:nir:stato:codice.civile:1942;262~art1218',
    articleId: '1218',
    tipo_atto: 'codice civile',
    numero_atto: undefined,
    data_atto: undefined,
  };

  beforeEach(() => {
    vi.clearAllMocks();

    // Default: panel closed
    mockUseMerltPanelStore.mockImplementation(
      (selector: (state: Record<string, unknown>) => unknown) => {
        const state = { open: mockOpen, isOpen: false };
        return selector(state);
      },
    );

    // Default: not enriching, not processed, no pending
    mockUseMerltArticleStatus.mockReturnValue({
      isEnriching: false,
      hasBeenProcessed: false,
      pendingCount: 0,
    });
  });

  it('[P1] should render the Brain icon button', () => {
    render(<MerltToolbar {...defaultProps} />);

    const button = screen.getByRole('button', { name: 'Apri pannello MERLT' });
    expect(button).toBeInTheDocument();
    expect(screen.getByTestId('icon-Brain')).toBeInTheDocument();
  });

  it('[P1] should call open() when button is clicked', () => {
    render(<MerltToolbar {...defaultProps} />);

    const button = screen.getByRole('button', { name: 'Apri pannello MERLT' });
    fireEvent.click(button);

    expect(mockOpen).toHaveBeenCalledTimes(1);
  });

  it('[P1] should show pulse animation when enriching', () => {
    mockUseMerltArticleStatus.mockReturnValue({
      isEnriching: true,
      hasBeenProcessed: false,
      pendingCount: 0,
    });

    render(<MerltToolbar {...defaultProps} />);

    const icon = screen.getByTestId('icon-Brain');
    expect(icon.className).toContain('animate-pulse');
  });

  it('[P1] should not show pulse animation when not enriching', () => {
    render(<MerltToolbar {...defaultProps} />);

    const icon = screen.getByTestId('icon-Brain');
    expect(icon.className).not.toContain('animate-pulse');
  });

  it('[P1] should show pending count badge when there are pending items', () => {
    mockUseMerltArticleStatus.mockReturnValue({
      isEnriching: false,
      hasBeenProcessed: true,
      pendingCount: 5,
    });

    render(<MerltToolbar {...defaultProps} />);

    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('[P1] should show 9+ when pending count exceeds 9', () => {
    mockUseMerltArticleStatus.mockReturnValue({
      isEnriching: false,
      hasBeenProcessed: true,
      pendingCount: 15,
    });

    render(<MerltToolbar {...defaultProps} />);

    expect(screen.getByText('9+')).toBeInTheDocument();
  });

  it('[P1] should not show badge when pending count is 0', () => {
    render(<MerltToolbar {...defaultProps} />);

    expect(screen.queryByText('0')).not.toBeInTheDocument();
    expect(screen.queryByText('9+')).not.toBeInTheDocument();
  });

  it('[P1] should apply active styles when panel is open', () => {
    mockUseMerltPanelStore.mockImplementation(
      (selector: (state: Record<string, unknown>) => unknown) => {
        const state = { open: mockOpen, isOpen: true };
        return selector(state);
      },
    );

    render(<MerltToolbar {...defaultProps} />);

    const button = screen.getByRole('button', { name: 'Apri pannello MERLT' });
    expect(button.className).toContain('bg-blue-100');
  });

  it('[P1] should apply processed styles when article has been processed', () => {
    mockUseMerltArticleStatus.mockReturnValue({
      isEnriching: false,
      hasBeenProcessed: true,
      pendingCount: 0,
    });

    render(<MerltToolbar {...defaultProps} />);

    const button = screen.getByRole('button', { name: 'Apri pannello MERLT' });
    expect(button.className).toContain('bg-emerald-50');
    expect(button.getAttribute('title')).toBe('Nel Knowledge Graph');
  });

  it('[P1] should show contribute title when not processed', () => {
    render(<MerltToolbar {...defaultProps} />);

    const button = screen.getByRole('button', { name: 'Apri pannello MERLT' });
    expect(button.getAttribute('title')).toBe('Contribuisci al Knowledge Graph');
  });

  it('[P1] should pass article parameters to useMerltArticleStatus', () => {
    render(<MerltToolbar {...defaultProps} />);

    expect(mockUseMerltArticleStatus).toHaveBeenCalledWith(
      expect.objectContaining({
        tipo_atto: 'codice civile',
        articolo: '1218',
        user_id: 'anonymous',
        enabled: true,
      }),
    );
  });

  it('[P1] should disable status hook when tipo_atto or articleId is missing', () => {
    render(<MerltToolbar {...defaultProps} tipo_atto="" articleId="" />);

    expect(mockUseMerltArticleStatus).toHaveBeenCalledWith(
      expect.objectContaining({
        enabled: false,
      }),
    );
  });
});
