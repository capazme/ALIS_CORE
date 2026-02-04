/**
 * Tests for Button component
 *
 * Tests the actual implementation:
 * - Variants: primary, secondary, ghost, danger, glass, outline
 * - Sizes: sm, md, lg
 * - Loading state
 * - Disabled state
 * - Icon support
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Button } from '@/components/ui/Button';

describe('Button', () => {
  it('renders with default props', () => {
    render(<Button>Click me</Button>);

    expect(screen.getByRole('button')).toBeInTheDocument();
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('calls onClick handler when clicked', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click</Button>);

    fireEvent.click(screen.getByRole('button'));

    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('renders disabled state', () => {
    render(<Button disabled>Disabled</Button>);

    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
  });

  it('applies primary variant styles by default', () => {
    render(<Button>Primary</Button>);

    const button = screen.getByRole('button');
    // Primary variant has bg-primary-600 class
    expect(button.className).toContain('bg-primary-600');
  });

  it('applies secondary variant styles', () => {
    render(<Button variant="secondary">Secondary</Button>);

    const button = screen.getByRole('button');
    // Secondary variant has specific border class
    expect(button.className).toContain('border-slate-200');
  });

  it('applies ghost variant styles', () => {
    render(<Button variant="ghost">Ghost</Button>);

    const button = screen.getByRole('button');
    expect(button.className).toContain('bg-transparent');
  });

  it('applies danger variant styles', () => {
    render(<Button variant="danger">Danger</Button>);

    const button = screen.getByRole('button');
    expect(button.className).toContain('bg-red-600');
  });

  it('renders with small size', () => {
    render(<Button size="sm">Small</Button>);

    const button = screen.getByRole('button');
    expect(button.className).toContain('px-3');
    expect(button.className).toContain('text-xs');
  });

  it('renders with medium size (default)', () => {
    render(<Button>Medium</Button>);

    const button = screen.getByRole('button');
    expect(button.className).toContain('px-4');
    expect(button.className).toContain('text-sm');
  });

  it('renders with large size', () => {
    render(<Button size="lg">Large</Button>);

    const button = screen.getByRole('button');
    expect(button.className).toContain('px-6');
    expect(button.className).toContain('text-base');
  });

  it('shows loading state with spinner', () => {
    render(<Button loading>Submit</Button>);

    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders with icon', () => {
    const TestIcon = () => <span data-testid="test-icon">★</span>;
    render(<Button icon={<TestIcon />}>With Icon</Button>);

    expect(screen.getByTestId('test-icon')).toBeInTheDocument();
    expect(screen.getByText('With Icon')).toBeInTheDocument();
  });

  it('does not call onClick when disabled', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick} disabled>Disabled</Button>);

    fireEvent.click(screen.getByRole('button'));

    expect(handleClick).not.toHaveBeenCalled();
  });

  it('does not call onClick when loading', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick} loading>Loading</Button>);

    fireEvent.click(screen.getByRole('button'));

    expect(handleClick).not.toHaveBeenCalled();
  });

  it('applies custom className', () => {
    render(<Button className="custom-class">Custom</Button>);

    const button = screen.getByRole('button');
    expect(button.className).toContain('custom-class');
  });

  it('forwards ref to button element', () => {
    const ref = { current: null as HTMLButtonElement | null };
    render(<Button ref={ref}>Ref Button</Button>);

    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });
});
