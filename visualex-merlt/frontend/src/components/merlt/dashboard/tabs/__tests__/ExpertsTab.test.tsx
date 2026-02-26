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
    'aria-expanded',
    'data-testid',
  ]);
  const filtered: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) {
    if (allowed.has(k)) filtered[k] = v;
  }
  return filtered;
}

const mockGet = vi.fn();
const mockPost = vi.fn();

vi.mock('../../../../../services/api', () => ({
  get: (...args: unknown[]) => mockGet(...args),
  post: (...args: unknown[]) => mockPost(...args),
}));

import { ExpertsTab } from '../ExpertsTab';

const mockPerformance = {
  experts: [
    {
      name: 'literal',
      display_name: 'Letterale',
      accuracy: 87.5,
      accuracy_ci: [82.0, 93.0],
      latency_ms: 1200,
      latency_p95: 2400,
      usage_percentage: 30.0,
      feedback_score: 0.75,
      feedback_count: 20,
      queries_handled: 150,
    },
    {
      name: 'systemic',
      display_name: 'Sistematico',
      accuracy: 82.0,
      accuracy_ci: [77.0, 87.0],
      latency_ms: 1500,
      latency_p95: 3000,
      usage_percentage: 25.0,
      feedback_score: 0.68,
      feedback_count: 15,
      queries_handled: 125,
    },
  ],
  period_days: 7,
  total_queries: 500,
  last_updated: '2026-01-01T00:00:00Z',
};

const mockQueryStats = {
  total_queries: 500,
  by_type: [
    { type: 'definitional', count: 200, percentage: 40.0, avg_latency_ms: 800, avg_confidence: 0.85 },
    { type: 'interpretive', count: 150, percentage: 30.0, avg_latency_ms: 1200, avg_confidence: 0.75 },
  ],
  avg_latency_ms: 1000,
  avg_confidence: 0.80,
  period_days: 7,
};

const mockRecentQueries = {
  queries: [
    {
      trace_id: 'trace-1',
      query: 'Cos\u00e8 la risoluzione del contratto?',
      timestamp: '2026-01-01T10:00:00Z',
      experts_used: ['literal', 'systemic'],
      confidence: 0.88,
      latency_ms: 1100,
      mode: 'convergent',
      feedback_received: true,
    },
  ],
  total_count: 1,
  has_more: false,
};

const mockAggregation = {
  method: 'weighted_average',
  total_responses: 500,
  agreement_rate: 78.5,
  divergence_count: 107,
  divergence_rate: 21.5,
  avg_confidence: 0.82,
  confidence_ci: [0.78, 0.86],
  avg_experts_per_query: 3.2,
};

describe('ExpertsTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders with mock data', async () => {
    mockGet
      .mockResolvedValueOnce(mockPerformance)
      .mockResolvedValueOnce(mockQueryStats)
      .mockResolvedValueOnce(mockRecentQueries)
      .mockResolvedValueOnce(mockAggregation);

    render(<ExpertsTab />);

    await waitFor(() => {
      expect(screen.getByText('Expert Performance (Last 7 days)')).toBeInTheDocument();
    });

    expect(screen.getByText('Letterale')).toBeInTheDocument();
    expect(screen.getByText('Sistematico')).toBeInTheDocument();
    expect(screen.getByText('Classificazione Query')).toBeInTheDocument();
    expect(screen.getByText('Response Aggregation')).toBeInTheDocument();
  });

  it('shows loading state while data fetches', () => {
    mockGet.mockReturnValue(new Promise(() => {}));
    render(<ExpertsTab />);

    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('handles empty experts list gracefully', async () => {
    mockGet
      .mockResolvedValueOnce({ experts: [], period_days: 7, total_queries: 0, last_updated: '' })
      .mockResolvedValueOnce({ total_queries: 0, by_type: [], avg_latency_ms: 0, avg_confidence: 0, period_days: 7 })
      .mockResolvedValueOnce({ queries: [], total_count: 0, has_more: false })
      .mockResolvedValueOnce(mockAggregation);

    render(<ExpertsTab />);

    await waitFor(() => {
      expect(screen.getByText('Response Aggregation')).toBeInTheDocument();
    });

    expect(screen.getByText('Nessuna query recente disponibile')).toBeInTheDocument();
  });

  it('handles error state from service', async () => {
    mockGet.mockRejectedValue(new Error('API error'));
    render(<ExpertsTab />);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    expect(screen.getByText('Errore nel caricamento delle metriche expert')).toBeInTheDocument();
    expect(screen.getByText('Riprova')).toBeInTheDocument();
  });
});
