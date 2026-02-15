/**
 * Tooltip - Lightweight tooltip on hover with a11y support.
 */

import { useState, useRef, useId } from 'react';
import { cn } from '../../lib/utils';

type TooltipContent = React.ReactNode;

export interface TooltipProps {
  content: TooltipContent;
  children: React.ReactElement;
  /** @deprecated Use `side` instead */
  placement?: 'top' | 'bottom' | 'left' | 'right';
  side?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

const POSITION_CLASSES = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 -translate-y-1/2 ml-2',
} as const;

export function Tooltip({ content, children, side, placement, className }: TooltipProps) {
  const resolvedSide = side ?? placement ?? 'top';
  const [visible, setVisible] = useState(false);
  const timeoutRef = useRef(undefined as ReturnType<typeof setTimeout> | undefined);
  const tooltipId = useId();

  const show = () => {
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setVisible(true), 200);
  };

  const hide = () => {
    clearTimeout(timeoutRef.current);
    setVisible(false);
  };

  return (
    <div
      className="relative inline-block"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {visible && (
        <div
          id={tooltipId}
          role="tooltip"
          className={cn(
            'absolute z-50 whitespace-nowrap rounded-md px-2.5 py-1.5',
            'text-xs font-medium text-white bg-slate-800 shadow-lg',
            'dark:bg-slate-700 dark:text-slate-100',
            'pointer-events-none',
            POSITION_CLASSES[resolvedSide],
            className
          )}
        >
          {content}
        </div>
      )}
    </div>
  );
}
