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
  ]);
  const filtered: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) {
    if (allowed.has(k)) filtered[k] = v;
  }
  return filtered;
}

const mockGetDashboardOverview = vi.fn();

vi.mock('../../../../../services/dashboardService', () => ({
  getDashboardOverview: (...args: unknown[]) => mockGetDashboardOverview(...args),
  getStatusColor: (status: string) => {
    if (status === 'online') return 'text-green-500';
    if (status === 'offline') return 'text-red-500';
    return 'text-slate-500';
  },
  getSeverityColor: () => 'bg-blue-100 text-blue-800',
  formatUptime: (s: number) => `${s}s`,
}));

import { OverviewTab } from '../OverviewTab';

const mockOverviewData = {
  knowledge_graph: {
    total_nodes: 27742,
    total_edges: 43935,
    articles_count: 1200,
    entities_count: 5000,
    embeddings_count: 5926,
    bridge_mappings: 27117,
  },
  rlcf: {
    total_feedback: 150,
    buffer_size: 80,
    training_epochs: 10,
    avg_authority: 0.72,
    active_users: 5,
  },
  experts: {
    total_queries: 500,
    avg_latency_ms: 1200,
    avg_confidence: 0.85,
    agreement_rate: 0.78,
  },
  health: {
    overall_status: 'online' as const,
    services: [
      { name: 'FalkorDB', status: 'online' as const, latency_ms: 12, details: {}, last_check: '2026-01-01T00:00:00Z' },
      { name: 'Qdrant', status: 'online' as const, latency_ms: 8, details: {}, last_check: '2026-01-01T00:00:00Z' },
    ],
    uptime_seconds: 3600,
    last_check: '2026-01-01T00:00:00Z',
  },
  recent_activity: {
    entries: [
      { id: '1', type: 'pipeline_start', message: 'Pipeline started', details: {}, timestamp: '2026-01-01T10:00:00Z', severity: 'info' as const },
    ],
    total_count: 1,
    has_more: false,
  },
  last_updated: '2026-01-01T00:00:00Z',
};

describe('OverviewTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders with mock data', async () => {
    mockGetDashboardOverview.mockResolvedValue(mockOverviewData);
    render(<OverviewTab />);

    await waitFor(() => {
      expect(screen.getByText('Knowledge Graph')).toBeInTheDocument();
    });

    expect(screen.getByText('System Health')).toBeInTheDocument();
    expect(screen.getByText('Recent Activity')).toBeInTheDocument();
  });

  it('shows loading state while data fetches', () => {
    mockGetDashboardOverview.mockReturnValue(new Promise(() => {}));
    render(<OverviewTab />);

    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('handles empty activity data gracefully', async () => {
    const dataWithEmptyActivity = {
      ...mockOverviewData,
      recent_activity: { entries: [], total_count: 0, has_more: false },
    };
    mockGetDashboardOverview.mockResolvedValue(dataWithEmptyActivity);
    render(<OverviewTab />);

    await waitFor(() => {
      expect(screen.getByText('Knowledge Graph')).toBeInTheDocument();
    });

    expect(screen.getByText('Recent Activity')).toBeInTheDocument();
  });

  it('handles error state from service', async () => {
    mockGetDashboardOverview.mockRejectedValue(new Error('Network error'));
    render(<OverviewTab />);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    expect(screen.getByText('Errore nel caricamento dei dati')).toBeInTheDocument();
    expect(screen.getByText('Riprova')).toBeInTheDocument();
  });
});
