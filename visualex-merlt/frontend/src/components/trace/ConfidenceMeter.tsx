/**
 * ConfidenceMeter - Visual confidence indicator with color gradient.
 */

import { cn } from '../../lib/utils';

export interface ConfidenceMeterProps {
  value: number; // 0-1
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  className?: string;
}

function getConfidenceColor(value: number): string {
  if (value >= 0.8) return '#10b981'; // emerald
  if (value >= 0.6) return '#f59e0b'; // amber
  if (value >= 0.4) return '#f97316'; // orange
  return '#ef4444'; // red
}

const SIZES = {
  sm: { bar: 'h-1.5', text: 'text-[10px]', container: 'gap-1' },
  md: { bar: 'h-2', text: 'text-xs', container: 'gap-1.5' },
  lg: { bar: 'h-3', text: 'text-sm', container: 'gap-2' },
} as const;

export function ConfidenceMeter({ value, size = 'md', showLabel = true, className }: ConfidenceMeterProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const percentage = Math.round(clamped * 100);
  const color = getConfidenceColor(clamped);

  const s = SIZES[size];

  return (
    <div className={cn("flex items-center", s.container, className)}>
      {/* Progress bar */}
      <div
        className={cn("flex-1 rounded-full overflow-hidden bg-slate-200 dark:bg-slate-700", s.bar)}
        role="progressbar"
        aria-valuenow={percentage}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Confidenza"
      >
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>

      {/* Label */}
      {showLabel && (
        <span className={cn("font-semibold shrink-0", s.text)} style={{ color }}>
          {percentage}%
        </span>
      )}
    </div>
  );
}

/**
 * Circular confidence gauge variant.
 */
export interface ConfidenceGaugeProps {
  value: number;
  size?: number;
  className?: string;
}

export function ConfidenceGauge({ value, size = 64, className }: ConfidenceGaugeProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const percentage = Math.round(clamped * 100);
  const color = getConfidenceColor(clamped);
  const radius = (size - 8) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - clamped);

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)} style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90" role="img" aria-label={`Confidenza: ${percentage}%`}>
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={4}
          className="text-slate-200 dark:text-slate-700"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={4}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700"
        />
      </svg>
      {/* Center text (decorative, SVG carries the aria-label) */}
      <span className="absolute text-xs font-bold" style={{ color }} aria-hidden="true">
        {percentage}%
      </span>
    </div>
  );
}
