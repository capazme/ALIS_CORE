/**
 * Tests for Button component
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

  it('renders with variant styles', () => {
    const { rerender } = render(<Button variant="primary">Primary</Button>);
    expect(screen.getByRole('button')).toHaveClass(/primary/i);

    rerender(<Button variant="secondary">Secondary</Button>);
    expect(screen.getByRole('button')).toHaveClass(/secondary/i);
  });

  it('renders with different sizes', () => {
    const { rerender } = render(<Button size="sm">Small</Button>);
    expect(screen.getByRole('button')).toHaveClass(/sm/i);

    rerender(<Button size="lg">Large</Button>);
    expect(screen.getByRole('button')).toHaveClass(/lg/i);
  });

  it('renders as a link when href is provided', () => {
    render(<Button href="/path">Link Button</Button>);

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', '/path');
  });

  it('shows loading state', () => {
    render(<Button loading>Loading</Button>);

    expect(screen.getByRole('button')).toBeDisabled();
    // Should show spinner or loading indicator
  });
});
