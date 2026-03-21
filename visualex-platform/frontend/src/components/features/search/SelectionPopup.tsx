import { useState, useEffect, useRef, useCallback } from 'react';
import { Highlighter, StickyNote, Copy, Search, X, Brain } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { EventBus } from '../../../lib/plugins/EventBus';

export interface SelectedTextInfo {
  text: string;
  startOffset: number;
  endOffset: number;
}

interface SelectionPopupProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  onHighlight: (text: string, color: 'yellow' | 'green' | 'red' | 'blue') => void;
  onAddNote: (text: string) => void;
  onCopy: (text: string) => void;
  onSearch?: (text: string) => void;
  urn?: string; // For event emission
}

interface PopupState {
  visible: boolean;
  x: number;
  y: number;
  text: string;
  startOffset: number;
  endOffset: number;
}

const HIGHLIGHT_COLORS = [
  { name: 'yellow', bg: 'bg-yellow-200', border: 'border-yellow-400', hover: 'hover:bg-yellow-300' },
  { name: 'green', bg: 'bg-green-200', border: 'border-green-400', hover: 'hover:bg-green-300' },
  { name: 'blue', bg: 'bg-blue-200', border: 'border-blue-400', hover: 'hover:bg-blue-300' },
  { name: 'red', bg: 'bg-red-200', border: 'border-red-400', hover: 'hover:bg-red-300' },
] as const;

export function SelectionPopup({
  containerRef,
  onHighlight,
  onAddNote,
  onCopy,
  onSearch,
  urn
}: SelectionPopupProps) {
  const [popup, setPopup] = useState<PopupState>({ visible: false, x: 0, y: 0, text: '', startOffset: 0, endOffset: 0 });
  const [showColorPicker, setShowColorPicker] = useState(false);
  const popupRef = useRef<HTMLDivElement>(null);
  const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const hidePopup = useCallback(() => {
    setPopup(prev => ({ ...prev, visible: false }));
    setShowColorPicker(false);
  }, []);

  const handleMouseUp = useCallback(() => {
    // Clear any pending hide timeout
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = null;
    }

    // Small delay to let selection finalize
    setTimeout(() => {
      const selection = window.getSelection();
      const selectedText = selection?.toString().trim();

      if (!selectedText || selectedText.length < 2) {
        // Delay hiding to allow clicking on popup
        hideTimeoutRef.current = setTimeout(hidePopup, 200);
        return;
      }

      // Check if selection is within our container
      if (!selection?.rangeCount || !containerRef.current) return;

      const range = selection.getRangeAt(0);
      if (!containerRef.current.contains(range.commonAncestorContainer)) {
        hideTimeoutRef.current = setTimeout(hidePopup, 200);
        return;
      }

      // Get position for popup
      const rect = range.getBoundingClientRect();
      const containerRect = containerRef.current.getBoundingClientRect();

      // Position above the selection, centered
      const x = rect.left + rect.width / 2 - containerRect.left;
      const y = rect.top - containerRect.top - 10;

      // Calculate text offsets within container for NER training
      // Create a range from start of container to start of selection
      const preSelectionRange = document.createRange();
      preSelectionRange.selectNodeContents(containerRef.current);
      preSelectionRange.setEnd(range.startContainer, range.startOffset);
      const startOffset = preSelectionRange.toString().length;
      const endOffset = startOffset + selectedText.length;

      setPopup({
        visible: true,
        x: Math.max(80, Math.min(x, containerRect.width - 80)), // Keep within bounds
        y: Math.max(50, y), // Ensure not too high
        text: selectedText,
        startOffset,
        endOffset
      });

      // Emit text-selected event for plugins
      if (urn) {
        EventBus.emit('article:text-selected', {
          urn,
          text: selectedText,
          startOffset,
          endOffset,
        });
      }
    }, 10);
  }, [containerRef, hidePopup, urn]);

  // Handle mousedown outside popup to hide it
  useEffect(() => {
    const handleMouseDown = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        // Don't hide immediately if clicking within container (might be selecting)
        if (containerRef.current?.contains(e.target as Node)) {
          return;
        }
        hidePopup();
      }
    };

    document.addEventListener('mousedown', handleMouseDown);
    return () => document.removeEventListener('mousedown', handleMouseDown);
  }, [containerRef, hidePopup]);

  // Attach mouseup listener to container
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('mouseup', handleMouseUp);
    return () => container.removeEventListener('mouseup', handleMouseUp);
  }, [containerRef, handleMouseUp]);

  // Handle keyboard shortcuts + Tab focus trap
  useEffect(() => {
    if (!popup.visible) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        hidePopup();
        window.getSelection()?.removeAllRanges();
      } else if (e.key === 'Tab' && popupRef.current) {
        const focusable = Array.from(
          popupRef.current.querySelectorAll<HTMLElement>('button:not([disabled])')
        );
        if (focusable.length === 0) return;
        e.preventDefault();
        const current = document.activeElement;
        const idx = focusable.indexOf(current as HTMLElement);
        if (e.shiftKey) {
          const prev = idx <= 0 ? focusable[focusable.length - 1] : focusable[idx - 1];
          prev.focus();
        } else {
          const next = idx === -1 || idx === focusable.length - 1 ? focusable[0] : focusable[idx + 1];
          next.focus();
        }
      } else if (e.key === 'h' && !e.metaKey && !e.ctrlKey) {
        onHighlight(popup.text, 'yellow');
        hidePopup();
        window.getSelection()?.removeAllRanges();
      } else if (e.key === 'n' && !e.metaKey && !e.ctrlKey) {
        onAddNote(popup.text);
        hidePopup();
        window.getSelection()?.removeAllRanges();
      } else if ((e.key === 'c' && (e.metaKey || e.ctrlKey))) {
        setTimeout(hidePopup, 100);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [popup.visible, popup.text, onHighlight, onAddNote, hidePopup]);

  const handleAction = (action: 'highlight' | 'note' | 'copy' | 'search' | 'merlt') => {
    switch (action) {
      case 'highlight':
        setShowColorPicker(true);
        break;
      case 'note':
        onAddNote(popup.text);
        hidePopup();
        window.getSelection()?.removeAllRanges();
        break;
      case 'copy':
        onCopy(popup.text);
        hidePopup();
        window.getSelection()?.removeAllRanges();
        break;
      case 'search':
        onSearch?.(popup.text);
        hidePopup();
        window.getSelection()?.removeAllRanges();
        break;
      case 'merlt':
        EventBus.emit('merlt:query-prefill', { text: popup.text, urn });
        hidePopup();
        window.getSelection()?.removeAllRanges();
        break;
    }
  };

  const handleHighlightColor = (color: 'yellow' | 'green' | 'red' | 'blue') => {
    onHighlight(popup.text, color);
    hidePopup();
    setShowColorPicker(false);
    window.getSelection()?.removeAllRanges();
  };

  if (!popup.visible) return null;

  return (
    <div
      ref={popupRef}
      role="dialog"
      aria-label="Azioni selezione testo"
      className={cn(
        "absolute z-50 transform -translate-x-1/2 -translate-y-full",
        "animate-in fade-in zoom-in-95 duration-150"
      )}
      style={{
        left: popup.x,
        top: popup.y
      }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {/* Main popup */}
      <div className="bg-slate-900 dark:bg-slate-800 text-white rounded-lg shadow-2xl border border-slate-700 overflow-hidden">
        {showColorPicker ? (
          /* Color picker view */
          <div className="p-2 flex items-center gap-1">
            <button
              onClick={() => setShowColorPicker(false)}
              className="p-1.5 rounded hover:bg-slate-700 text-slate-400"
              aria-label="Torna alle azioni"
            >
              <X size={14} />
            </button>
            <div className="w-px h-5 bg-slate-700 mx-1" />
            {HIGHLIGHT_COLORS.map(({ name, bg, border, hover }) => (
              <button
                key={name}
                onClick={() => handleHighlightColor(name as 'yellow' | 'green' | 'red' | 'blue')}
                className={cn(
                  "w-7 h-7 rounded-full border-2 transition-transform hover:scale-110",
                  bg, border, hover
                )}
                aria-label={`Evidenzia in ${name}`}
              />
            ))}
          </div>
        ) : (
          /* Main actions */
          <div role="toolbar" aria-label="Azioni testo selezionato" className="flex items-center">
            <button
              onClick={() => handleAction('highlight')}
              className="p-2.5 hover:bg-slate-700 transition-colors flex items-center gap-1.5 text-sm focus-visible:ring-2 focus-visible:ring-yellow-400 focus-visible:outline-none"
              aria-label="Evidenzia testo (H)"
              title="Evidenzia (H)"
            >
              <Highlighter size={16} className="text-yellow-400" />
            </button>
            <div className="w-px h-5 bg-slate-700" />
            <button
              onClick={() => handleAction('note')}
              className="p-2.5 hover:bg-slate-700 transition-colors flex items-center gap-1.5 text-sm focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:outline-none"
              aria-label="Aggiungi nota (N)"
              title="Aggiungi nota (N)"
            >
              <StickyNote size={16} className="text-blue-400" />
            </button>
            <div className="w-px h-5 bg-slate-700" />
            <button
              onClick={() => handleAction('copy')}
              className="p-2.5 hover:bg-slate-700 transition-colors flex items-center gap-1.5 text-sm focus-visible:ring-2 focus-visible:ring-green-400 focus-visible:outline-none"
              aria-label="Copia testo selezionato (Cmd+C)"
              title="Copia (Cmd+C)"
            >
              <Copy size={16} className="text-green-400" />
            </button>
            {onSearch && (
              <>
                <div className="w-px h-5 bg-slate-700" />
                <button
                  onClick={() => handleAction('search')}
                  className="p-2.5 hover:bg-slate-700 transition-colors flex items-center gap-1.5 text-sm focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:outline-none"
                  aria-label="Cerca testo selezionato"
                  title="Cerca"
                >
                  <Search size={16} className="text-purple-400" />
                </button>
              </>
            )}
            <div className="w-px h-5 bg-slate-700" />
            <button
              onClick={() => handleAction('merlt')}
              className="p-2.5 hover:bg-slate-700 transition-colors flex items-center gap-1.5 text-sm focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:outline-none"
              aria-label="Chiedi a MERL-T"
              title="Chiedi a MERL-T"
            >
              <Brain size={16} className="text-indigo-400" />
            </button>
          </div>
        )}
      </div>

      {/* Arrow pointing down */}
      <div className="absolute left-1/2 transform -translate-x-1/2 top-full">
        <div className="w-0 h-0 border-l-[8px] border-l-transparent border-r-[8px] border-r-transparent border-t-[8px] border-t-slate-900 dark:border-t-slate-800" />
      </div>
    </div>
  );
}
