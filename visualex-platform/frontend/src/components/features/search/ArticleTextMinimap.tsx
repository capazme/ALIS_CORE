/**
 * ArticleTextMinimap - Vertical minimap showing document overview with viewport indicator.
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { cn } from '../../../lib/utils';

export interface ArticleTextMinimapProps {
  contentRef: React.RefObject<HTMLDivElement | null>;
  highlights?: Array<{ text: string; color: string }>;
  className?: string;
}

export function ArticleTextMinimap({ contentRef, highlights = [], className }: ArticleTextMinimapProps) {
  const [viewportTop, setViewportTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(30);
  const [contentHeight, setContentHeight] = useState(0);
  const minimapRef = useRef<HTMLDivElement>(null);
  const minimapHeight = 200;

  const updateViewport = useCallback(() => {
    const el = contentRef.current;
    if (!el) return;

    const scrollParent = el.closest('.overflow-y-auto') || el.parentElement;
    if (!scrollParent) return;

    const totalH = el.scrollHeight;
    const visibleH = scrollParent.clientHeight;
    const scrollT = scrollParent.scrollTop;

    if (totalH === 0) return;

    setContentHeight(totalH);
    const ratio = minimapHeight / totalH;
    setViewportTop(scrollT * ratio);
    setViewportHeight(Math.max(20, visibleH * ratio));
  }, [contentRef]);

  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;

    const scrollParent = el.closest('.overflow-y-auto') || el.parentElement;
    if (!scrollParent) return;

    updateViewport();
    scrollParent.addEventListener('scroll', updateViewport, { passive: true });
    window.addEventListener('resize', updateViewport);

    return () => {
      scrollParent.removeEventListener('scroll', updateViewport);
      window.removeEventListener('resize', updateViewport);
    };
  }, [contentRef, updateViewport]);

  const handleMinimapClick = useCallback((e: React.MouseEvent) => {
    const el = contentRef.current;
    const minimap = minimapRef.current;
    if (!el || !minimap || contentHeight === 0) return;

    const scrollParent = el.closest('.overflow-y-auto') || el.parentElement;
    if (!scrollParent) return;

    const rect = minimap.getBoundingClientRect();
    const clickY = e.clientY - rect.top;
    const ratio = clickY / minimapHeight;
    scrollParent.scrollTo({ top: ratio * contentHeight, behavior: 'smooth' });
  }, [contentRef, contentHeight]);

  if (contentHeight < 600) return null;

  return (
    <div
      ref={minimapRef}
      onClick={handleMinimapClick}
      role="slider"
      aria-label="Navigazione documento"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={contentHeight > 0 ? Math.round((viewportTop / minimapHeight) * 100) : 0}
      tabIndex={0}
      className={cn(
        "relative w-3 cursor-pointer rounded-full bg-slate-100 transition-all duration-150 hover:w-4 dark:bg-slate-800",
        className
      )}
      style={{ height: minimapHeight }}
      title="Naviga nel documento"
    >
      {/* Highlight markers */}
      {highlights.map((h, i) => (
        <div
          key={i}
          className="absolute left-0 right-0 h-1 rounded-full opacity-60"
          style={{
            top: `${(i / Math.max(highlights.length, 1)) * 100}%`,
            backgroundColor: h.color === 'yellow' ? '#FCD34D' :
              h.color === 'green' ? '#6EE7B7' :
              h.color === 'red' ? '#FCA5A5' : '#93C5FD',
          }}
        />
      ))}

      {/* Viewport indicator */}
      <div
        className="absolute left-0 right-0 bg-primary-500/30 dark:bg-primary-400/30 rounded-full border border-primary-500/50"
        style={{
          top: viewportTop,
          height: viewportHeight,
        }}
      />
    </div>
  );
}
