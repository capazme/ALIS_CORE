import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => (
      <div {...filterDomProps(props)}>{children}</div>
    ),
  },
}));

vi.mock('lucide-react', () =>
  new Proxy(
    {},
    {
      get: (_target, name) => {
        if (name === '__esModule') return true;
        return ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => (
          <span data-testid={`icon-${String(name)}`} {...filterDomProps(props)}>
            {children}
          </span>
        );
      },
    },
  ),
);

function filterDomProps(props: Record<string, unknown>) {
  const allowed = new Set([
    'className',
    'style',
    'role',
    'id',
    'tabIndex',
    'onClick',
    'children',
    'aria-hidden',
    'aria-label',
    'data-testid',
    'disabled',
    'title',
    'href',
    'target',
    'rel',
  ]);
  const filtered: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) {
    if (allowed.has(k)) filtered[k] = v;
  }
  return filtered;
}

const mockGetArchitectureDiagram = vi.fn();
const mockGetNodeDetails = vi.fn();

vi.mock('../../../../../services/dashboardService', () => ({
  getArchitectureDiagram: (...args: unknown[]) => mockGetArchitectureDiagram(...args),
  getNodeDetails: (...args: unknown[]) => mockGetNodeDetails(...args),
}));

vi.mock('../../../../../services/circuitBreakerService', () => ({
  getCircuitBreakerStatus: vi.fn().mockResolvedValue({
    breakers: {},
    total_count: 0,
    open_count: 0,
  }),
  resetBreaker: vi.fn(),
}));

vi.mock('../ApiKeysSection', () => ({
  ApiKeysSection: () => <div data-testid="api-keys-section">ApiKeysSection</div>,
}));

import { ArchitectureTab } from '../ArchitectureTab';

const mockDiagram = {
  nodes: [
    { id: 'FalkorDB', label: 'FalkorDB', type: 'storage' as const, metrics: { nodes: 27742 }, status: 'online' as const },
    { id: 'Qdrant', label: 'Qdrant', type: 'storage' as const, metrics: { vectors: 5926 }, status: 'online' as const },
    { id: 'Literal', label: 'Letterale', type: 'expert' as const, metrics: { queries: 150 }, status: 'online' as const },
  ],
  edges: [
    { source: 'FalkorDB', target: 'Literal', label: 'graph data', animated: false },
  ],
};

describe('ArchitectureTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders with mock data', async () => {
    mockGetArchitectureDiagram.mockResolvedValue(mockDiagram);
    render(<ArchitectureTab />);

    await waitFor(() => {
      expect(screen.getByText('Legenda')).toBeInTheDocument();
    });

    expect(screen.getByText(/Clicca su un nodo/)).toBeInTheDocument();
    expect(screen.getByTestId('api-keys-section')).toBeInTheDocument();
  });

  it('shows loading state while data fetches', () => {
    mockGetArchitectureDiagram.mockReturnValue(new Promise(() => {}));
    render(<ArchitectureTab />);

    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('handles empty diagram gracefully', async () => {
    mockGetArchitectureDiagram.mockResolvedValue({ nodes: [], edges: [] });
    render(<ArchitectureTab />);

    await waitFor(() => {
      expect(screen.getByText('Legenda')).toBeInTheDocument();
    });

    expect(screen.getByText(/Clicca su un nodo/)).toBeInTheDocument();
  });

  it('handles error state from service', async () => {
    mockGetArchitectureDiagram.mockRejectedValue(new Error('Server error'));
    render(<ArchitectureTab />);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    expect(screen.getByText('Errore nel caricamento del diagramma')).toBeInTheDocument();
    expect(screen.getByText('Riprova')).toBeInTheDocument();
  });
});
