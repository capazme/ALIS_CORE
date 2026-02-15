/**
 * SourceSplitView - Resizable split view: synthesis (60%) + source detail (40%).
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { cn } from '../../lib/utils';

type NodeContent = React.ReactNode;
type DivMouseEvent = React.MouseEvent<HTMLDivElement>;
type DivKeyboardEvent = React.KeyboardEvent<HTMLDivElement>;

export interface SourceSplitViewProps {
  leftContent: NodeContent;
  rightContent: NodeContent;
  isOpen: boolean;
  className?: string;
}

export function SourceSplitView({ leftContent, rightContent, isOpen, className }: SourceSplitViewProps) {
  const [splitRatio, setSplitRatio] = useState(0.6); // 60% left, 40% right
  const containerRef = useRef(null as HTMLDivElement | null);
  const isDragging = useRef(false);

  const listenersRef = useRef({ move: null, up: null } as { move: ((e: MouseEvent) => void) | null; up: (() => void) | null });

  const cleanupListeners = useCallback(() => {
    if (listenersRef.current.move) {
      document.removeEventListener('mousemove', listenersRef.current.move);
      listenersRef.current.move = null;
    }
    if (listenersRef.current.up) {
      document.removeEventListener('mouseup', listenersRef.current.up);
      listenersRef.current.up = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return cleanupListeners;
  }, [cleanupListeners]);

  const handleMouseDown = useCallback((e: DivMouseEvent) => {
    e.preventDefault();
    isDragging.current = true;

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      setSplitRatio(Math.max(0.3, Math.min(0.8, ratio)));
    };

    const handleMouseUp = () => {
      isDragging.current = false;
      cleanupListeners();
    };

    listenersRef.current = { move: handleMouseMove, up: handleMouseUp };
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [cleanupListeners]);

  if (!isOpen) {
    return <div className={className}>{leftContent}</div>;
  }

  const leftPercent = `${splitRatio * 100}%`;
  const rightPercent = `${(1 - splitRatio) * 100}%`;

  return (
    <div
      ref={containerRef}
      className={cn("flex flex-col md:flex-row h-full overflow-hidden", className)}
      style={{ '--split-left': leftPercent, '--split-right': rightPercent } as React.CSSProperties}
    >
      {/* Left panel (synthesis) - full width on mobile, split ratio on md+ */}
      <div
        className="overflow-y-auto custom-scrollbar w-full md:w-[var(--split-left)]"
      >
        {leftContent}
      </div>

      {/* Resizer handle - hidden on mobile, visible on md+ */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-valuenow={Math.round(splitRatio * 100)}
        aria-valuemin={30}
        aria-valuemax={80}
        aria-label="Ridimensiona pannelli"
        tabIndex={0}
        onMouseDown={handleMouseDown}
        onKeyDown={(e: DivKeyboardEvent) => {
          const step = 0.05;
          if (e.key === 'ArrowLeft') {
            e.preventDefault();
            setSplitRatio((prev: number) => Math.max(0.3, prev - step));
          } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            setSplitRatio((prev: number) => Math.min(0.8, prev + step));
          }
        }}
        className="hidden md:flex w-1.5 shrink-0 items-center justify-center cursor-col-resize bg-slate-200 transition-colors dark:bg-slate-700 hover:bg-blue-400 dark:hover:bg-blue-600 focus-visible:bg-blue-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
      >
        <div className="w-0.5 h-8 rounded-full bg-slate-400 dark:bg-slate-500" />
      </div>

      {/* Mobile separator */}
      <div className="md:hidden border-t border-slate-200 dark:border-slate-700" />

      {/* Right panel (source detail) - full width on mobile, split ratio on md+ */}
      <div
        className="overflow-hidden w-full md:w-[var(--split-right)]"
      >
        {rightContent}
      </div>
    </div>
  );
}
