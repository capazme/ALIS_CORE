/**
 * SkipQueuePanel
 * ==============
 *
 * Collapsible panel showing items that were skipped during validation.
 * Allows users to return to skipped items or clear the queue.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  SkipForward,
  ChevronDown,
  ChevronRight,
  RotateCcw,
  Trash2,
  Clock,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import type { SkipQueueItem } from '../../hooks/useValidationState';

interface SkipQueuePanelProps {
  skipQueue: SkipQueueItem[];
  onUnSkip: (itemId: string) => void;
  onClearAll: () => void;
  getItemName: (itemId: string) => string;
  className?: string;
}

function formatTimeAgo(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / 60000);

  if (diffMinutes < 1) return 'Adesso';
  if (diffMinutes < 60) return `${diffMinutes}m fa`;
  return `${Math.floor(diffMinutes / 60)}h fa`;
}

export function SkipQueuePanel({
  skipQueue,
  onUnSkip,
  onClearAll,
  getItemName,
  className,
}: SkipQueuePanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (skipQueue.length === 0) {
    return null;
  }

  return (
    <div
      className={cn(
        'bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800/30 rounded-lg overflow-hidden',
        className
      )}
    >
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-amber-100/50 dark:hover:bg-amber-900/20 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset"
        aria-expanded={isExpanded}
        aria-label={`Elementi saltati (${skipQueue.length})`}
      >
        <div className="flex items-center gap-2">
          <SkipForward size={14} className="text-amber-600 dark:text-amber-400" aria-hidden="true" />
          <span className="text-xs font-medium text-amber-700 dark:text-amber-300">
            Saltati ({skipQueue.length})
          </span>
        </div>
        <div className="flex items-center gap-2">
          {!isExpanded && (
            <span className="text-[10px] text-amber-600 dark:text-amber-400">
              Clicca per rivedere
            </span>
          )}
          {isExpanded ? (
            <ChevronDown size={14} className="text-amber-500" />
          ) : (
            <ChevronRight size={14} className="text-amber-500" />
          )}
        </div>
      </button>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 border-t border-amber-200/50 dark:border-amber-800/20 pt-2 space-y-2">
              {/* Queue items */}
              <div className="space-y-1.5 max-h-40 overflow-y-auto">
                {skipQueue.map((item) => (
                  <div
                    key={item.itemId}
                    className="flex items-center gap-2 p-2 bg-white dark:bg-slate-800/50 rounded-md border border-amber-100 dark:border-amber-900/20"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate">
                        {getItemName(item.itemId)}
                      </p>
                      <div className="flex items-center gap-1 text-[10px] text-slate-500">
                        <Clock size={8} />
                        <span>{formatTimeAgo(item.skippedAt)}</span>
                        {item.reason && (
                          <>
                            <span>•</span>
                            <span className="truncate">{item.reason}</span>
                          </>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => onUnSkip(item.itemId)}
                      className={cn(
                        'p-1.5 rounded-md transition-colors',
                        'text-amber-600 hover:text-amber-700 hover:bg-amber-100',
                        'dark:text-amber-400 dark:hover:text-amber-300 dark:hover:bg-amber-900/30',
                        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                      )}
                      title="Torna a questo elemento"
                      aria-label={`Ripristina ${getItemName(item.itemId)}`}
                    >
                      <RotateCcw size={12} aria-hidden="true" />
                    </button>
                  </div>
                ))}
              </div>

              {/* Clear all button */}
              <button
                onClick={onClearAll}
                className={cn(
                  'w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium',
                  'text-amber-700 hover:bg-amber-100 dark:text-amber-300 dark:hover:bg-amber-900/20',
                  'transition-colors',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                )}
              >
                <Trash2 size={10} aria-hidden="true" />
                Rimuovi tutti gli elementi saltati
              </button>

              {/* Info */}
              <p className="text-[10px] text-amber-600/80 dark:text-amber-400/70 text-center">
                Gli elementi saltati torneranno visibili quando li selezioni o li rimuovi dalla coda.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
