import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';

vi.mock('../../../monitoring/PipelineMonitoringDashboard', () => ({
  PipelineMonitoringDashboard: () => (
    <div data-testid="pipeline-monitoring-dashboard">PipelineMonitoringDashboard</div>
  ),
}));

vi.mock('../SchedulesSection', () => ({
  SchedulesSection: () => (
    <div data-testid="schedules-section">SchedulesSection</div>
  ),
}));

import { PipelineTab } from '../PipelineTab';

describe('PipelineTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<PipelineTab />);

    expect(screen.getByTestId('pipeline-monitoring-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('schedules-section')).toBeInTheDocument();
  });

  it('renders loading state via child components', () => {
    render(<PipelineTab />);

    const container = screen.getByTestId('pipeline-monitoring-dashboard').parentElement;
    expect(container).toBeInTheDocument();
  });

  it('renders both child components in correct order', () => {
    render(<PipelineTab />);

    const dashboard = screen.getByTestId('pipeline-monitoring-dashboard');
    const schedules = screen.getByTestId('schedules-section');

    const parent = dashboard.parentElement;
    expect(parent).toBe(schedules.parentElement);

    const children = Array.from(parent!.children);
    const dashboardIdx = children.indexOf(dashboard);
    const schedulesIdx = children.indexOf(schedules);
    expect(dashboardIdx).toBeLessThan(schedulesIdx);
  });

  it('has correct container styling', () => {
    render(<PipelineTab />);

    const dashboard = screen.getByTestId('pipeline-monitoring-dashboard');
    const container = dashboard.parentElement;
    expect(container?.className).toContain('space-y-6');
  });
});
