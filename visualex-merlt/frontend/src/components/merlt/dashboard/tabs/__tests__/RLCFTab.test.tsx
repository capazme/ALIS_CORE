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
    'aria-valuenow',
    'aria-valuemin',
    'aria-valuemax',
    'data-testid',
    'disabled',
    'title',
  ]);
  const filtered: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) {
    if (allowed.has(k)) filtered[k] = v;
  }
  return filtered;
}

const mockGetTrainingStatus = vi.fn();
const mockGetBufferStatus = vi.fn();
const mockGetPolicyWeights = vi.fn();
const mockConnectTrainingStream = vi.fn(() => vi.fn());

vi.mock('../../../../../services/rlcfService', () => ({
  getTrainingStatus: (...args: unknown[]) => mockGetTrainingStatus(...args),
  startTraining: vi.fn(),
  stopTraining: vi.fn(),
  getBufferStatus: (...args: unknown[]) => mockGetBufferStatus(...args),
  getPolicyWeights: (...args: unknown[]) => mockGetPolicyWeights(...args),
  connectTrainingStream: (...args: unknown[]) => mockConnectTrainingStream(...args),
  formatETA: (s: number | undefined) => (s ? `${s}s` : '-'),
  formatLoss: (l: number | null | undefined) => (l != null ? l.toFixed(4) : '-'),
  getTrainingStatusBadge: () => ({ text: 'Non avviato', color: 'bg-slate-100 text-slate-800' }),
}));

vi.mock('../../../../../services/quarantineService', () => ({
  getFlaggedFeedback: vi.fn().mockResolvedValue({ items: [], total: 0 }),
  getQuarantinedFeedback: vi.fn().mockResolvedValue({ items: [], total: 0 }),
  approveFeedback: vi.fn(),
  quarantineFeedback: vi.fn(),
  autoDetectOutliers: vi.fn().mockResolvedValue({ flagged_count: 0 }),
}));

vi.mock('../PolicyEvolutionChart', () => ({
  PolicyEvolutionChart: () => <div data-testid="policy-evolution-chart">PolicyEvolutionChart</div>,
}));

vi.mock('../DevilsAdvocatePanel', () => ({
  DevilsAdvocatePanel: () => <div data-testid="devils-advocate-panel">DevilsAdvocatePanel</div>,
}));

import { RLCFTab } from '../RLCFTab';

const mockTrainingStatus = {
  is_running: false,
  is_paused: false,
  current_epoch: 0,
  total_epochs: 50,
  current_loss: null,
  best_loss: null,
  learning_rate: 0.001,
  started_at: null,
  eta_seconds: null,
  last_updated: null,
  training_sessions_today: 0,
};

const mockBufferStatus = {
  size: 80,
  capacity: 500,
  fill_percentage: 16.0,
  positive_count: 50,
  negative_count: 10,
  neutral_count: 20,
  training_ready: false,
  last_feedback_at: '2026-01-01T10:00:00Z',
};

const mockPolicyWeightsData = {
  gating: { literal: 0.3, systemic: 0.25, principles: 0.25, precedent: 0.2 },
  traversal: { max_depth: 3, decay_factor: 0.8 },
  timestamp: '2026-01-01T00:00:00Z',
};

describe('RLCFTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders with mock data', async () => {
    mockGetTrainingStatus.mockResolvedValue(mockTrainingStatus);
    mockGetBufferStatus.mockResolvedValue(mockBufferStatus);
    mockGetPolicyWeights.mockResolvedValue(mockPolicyWeightsData);

    render(<RLCFTab />);

    await waitFor(() => {
      expect(screen.getByText('Training Status')).toBeInTheDocument();
    });

    expect(screen.getByText('Policy Weights')).toBeInTheDocument();
    expect(screen.getByText('Feedback Buffer')).toBeInTheDocument();
    expect(screen.getByTestId('policy-evolution-chart')).toBeInTheDocument();
    expect(screen.getByTestId('devils-advocate-panel')).toBeInTheDocument();
  });

  it('shows loading state while data fetches', () => {
    mockGetTrainingStatus.mockReturnValue(new Promise(() => {}));
    mockGetBufferStatus.mockReturnValue(new Promise(() => {}));
    mockGetPolicyWeights.mockReturnValue(new Promise(() => {}));

    render(<RLCFTab />);

    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('handles empty buffer gracefully', async () => {
    mockGetTrainingStatus.mockResolvedValue(mockTrainingStatus);
    mockGetBufferStatus.mockResolvedValue({
      size: 0,
      capacity: 500,
      fill_percentage: 0,
      positive_count: 0,
      negative_count: 0,
      neutral_count: 0,
      training_ready: false,
      last_feedback_at: null,
    });
    mockGetPolicyWeights.mockResolvedValue(mockPolicyWeightsData);

    render(<RLCFTab />);

    await waitFor(() => {
      expect(screen.getByText('Feedback Buffer')).toBeInTheDocument();
    });

    expect(screen.getByText('Accumulating')).toBeInTheDocument();
  });

  it('handles error state from service', async () => {
    mockGetTrainingStatus.mockRejectedValue(new Error('Network error'));
    mockGetBufferStatus.mockRejectedValue(new Error('Network error'));
    mockGetPolicyWeights.mockRejectedValue(new Error('Network error'));

    render(<RLCFTab />);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    expect(screen.getByText('Errore nel caricamento dei dati RLCF')).toBeInTheDocument();
    expect(screen.getByText('Riprova')).toBeInTheDocument();
  });
});
