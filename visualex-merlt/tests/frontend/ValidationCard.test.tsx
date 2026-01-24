/**
 * Tests for ValidationCard component
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the services
vi.mock('../../frontend/src/services/merltService', () => ({
  submitValidation: vi.fn().mockResolvedValue({ success: true }),
}));

describe('ValidationCard', () => {
  const mockValidationItem = {
    id: 'val-1',
    type: 'entity',
    content: {
      name: 'Test Entity',
      entityType: 'CONCEPT',
      confidence: 0.85,
    },
    article: {
      urn: 'urn:nir:stato:codice.civile~art1453',
      title: 'Art. 1453 c.c.',
    },
    createdAt: '2024-01-15T10:00:00Z',
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders validation item details', async () => {
    const { ValidationCard } = await import(
      '../../frontend/src/components/merlt/ValidationCard'
    );

    render(<ValidationCard item={mockValidationItem} />);

    expect(screen.getByText('Test Entity')).toBeInTheDocument();
    expect(screen.getByText(/CONCEPT/i)).toBeInTheDocument();
  });

  it('shows approve and reject buttons', async () => {
    const { ValidationCard } = await import(
      '../../frontend/src/components/merlt/ValidationCard'
    );

    render(<ValidationCard item={mockValidationItem} />);

    expect(screen.getByRole('button', { name: /approva/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /rifiuta/i })).toBeInTheDocument();
  });

  it('calls onApprove when approve button clicked', async () => {
    const { ValidationCard } = await import(
      '../../frontend/src/components/merlt/ValidationCard'
    );

    const onApprove = vi.fn();
    render(<ValidationCard item={mockValidationItem} onApprove={onApprove} />);

    fireEvent.click(screen.getByRole('button', { name: /approva/i }));

    expect(onApprove).toHaveBeenCalledWith(mockValidationItem.id);
  });

  it('calls onReject when reject button clicked', async () => {
    const { ValidationCard } = await import(
      '../../frontend/src/components/merlt/ValidationCard'
    );

    const onReject = vi.fn();
    render(<ValidationCard item={mockValidationItem} onReject={onReject} />);

    fireEvent.click(screen.getByRole('button', { name: /rifiuta/i }));

    expect(onReject).toHaveBeenCalledWith(mockValidationItem.id);
  });

  it('displays confidence score', async () => {
    const { ValidationCard } = await import(
      '../../frontend/src/components/merlt/ValidationCard'
    );

    render(<ValidationCard item={mockValidationItem} />);

    expect(screen.getByText(/85%|0.85/)).toBeInTheDocument();
  });
});
