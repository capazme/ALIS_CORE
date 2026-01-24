/**
 * VoteHistoryModal
 * ================
 *
 * Timeline view of user's voting history in the current session.
 * Shows committed and uncommitted votes with timestamps.
 */

import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  History,
  ThumbsUp,
  ThumbsDown,
  Edit3,
  Clock,
  Check,
  Circle,
  Undo2,
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import { Modal } from '../../ui/Modal';
import type { VoteHistoryEntry, ValidationItemType } from '../../../hooks/useValidationState';
import type { VoteType } from '../../../types/merlt';

interface VoteHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  history: VoteHistoryEntry[];
  onUndoVote?: (entryId: string) => void;
}

const voteIcons: Record<VoteType, typeof ThumbsUp> = {
  approve: ThumbsUp,
  reject: ThumbsDown,
  edit: Edit3,
};

const voteColors: Record<VoteType, { text: string; bg: string; border: string }> = {
  approve: {
    text: 'text-emerald-600 dark:text-emerald-400',
    bg: 'bg-emerald-100 dark:bg-emerald-900/30',
    border: 'border-emerald-200 dark:border-emerald-800/30',
  },
  reject: {
    text: 'text-red-600 dark:text-red-400',
    bg: 'bg-red-100 dark:bg-red-900/30',
    border: 'border-red-200 dark:border-red-800/30',
  },
  edit: {
    text: 'text-amber-600 dark:text-amber-400',
    bg: 'bg-amber-100 dark:bg-amber-900/30',
    border: 'border-amber-200 dark:border-amber-800/30',
  },
};

const voteLabels: Record<VoteType, string> = {
  approve: 'Approvato',
  reject: 'Rifiutato',
  edit: 'Modificato',
};

const typeLabels: Record<ValidationItemType, string> = {
  entity: 'Entità',
  relation: 'Relazione',
};

function formatTime(date: Date): string {
  return date.toLocaleTimeString('it-IT', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);

  if (diffSeconds < 60) {
    return 'Adesso';
  } else if (diffMinutes < 60) {
    return `${diffMinutes} min fa`;
  } else {
    return formatTime(date);
  }
}

export function VoteHistoryModal({
  isOpen,
  onClose,
  history,
  onUndoVote,
}: VoteHistoryModalProps) {
  const committedVotes = history.filter((h) => h.isCommitted);
  const uncommittedVotes = history.filter((h) => !h.isCommitted);

  // Group by vote type for stats
  const stats = {
    approve: history.filter((h) => h.vote === 'approve').length,
    reject: history.filter((h) => h.vote === 'reject').length,
    edit: history.filter((h) => h.vote === 'edit').length,
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Cronologia Voti" size="md">
      <div className="space-y-6">
        {/* Stats Header */}
        <div className="grid grid-cols-3 gap-3">
          <div
            className={cn(
              'rounded-lg p-3 border',
              voteColors.approve.bg,
              voteColors.approve.border
            )}
          >
            <div className="flex items-center gap-2 mb-1">
              <ThumbsUp size={14} className={voteColors.approve.text} />
              <span
                className={cn('text-xs font-medium', voteColors.approve.text)}
              >
                Approvati
              </span>
            </div>
            <p className={cn('text-2xl font-bold', voteColors.approve.text)}>
              {stats.approve}
            </p>
          </div>

          <div
            className={cn(
              'rounded-lg p-3 border',
              voteColors.reject.bg,
              voteColors.reject.border
            )}
          >
            <div className="flex items-center gap-2 mb-1">
              <ThumbsDown size={14} className={voteColors.reject.text} />
              <span className={cn('text-xs font-medium', voteColors.reject.text)}>
                Rifiutati
              </span>
            </div>
            <p className={cn('text-2xl font-bold', voteColors.reject.text)}>
              {stats.reject}
            </p>
          </div>

          <div
            className={cn(
              'rounded-lg p-3 border',
              voteColors.edit.bg,
              voteColors.edit.border
            )}
          >
            <div className="flex items-center gap-2 mb-1">
              <Edit3 size={14} className={voteColors.edit.text} />
              <span className={cn('text-xs font-medium', voteColors.edit.text)}>
                Modificati
              </span>
            </div>
            <p className={cn('text-2xl font-bold', voteColors.edit.text)}>
              {stats.edit}
            </p>
          </div>
        </div>

        {/* Empty State */}
        {history.length === 0 && (
          <div className="text-center py-8">
            <History
              size={32}
              className="mx-auto text-slate-300 dark:text-slate-600 mb-3"
            />
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Nessun voto registrato in questa sessione
            </p>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
              I tuoi voti appariranno qui mentre validi
            </p>
          </div>
        )}

        {/* Uncommitted Section */}
        {uncommittedVotes.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Circle size={8} className="text-amber-500 fill-current" />
              <h3 className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
                Da Confermare ({uncommittedVotes.length})
              </h3>
            </div>
            <div className="space-y-2">
              {uncommittedVotes.map((entry, index) => (
                <VoteHistoryItem
                  key={entry.id}
                  entry={entry}
                  index={index}
                  onUndo={onUndoVote}
                  showUndo
                />
              ))}
            </div>
          </div>
        )}

        {/* Committed Section */}
        {committedVotes.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Check size={12} className="text-emerald-500" />
              <h3 className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
                Confermati ({committedVotes.length})
              </h3>
            </div>
            <div className="space-y-2 opacity-70">
              {committedVotes.map((entry, index) => (
                <VoteHistoryItem
                  key={entry.id}
                  entry={entry}
                  index={index}
                />
              ))}
            </div>
          </div>
        )}

        {/* Footer tip */}
        {history.length > 0 && (
          <div className="flex items-center gap-2 pt-4 border-t border-slate-100 dark:border-slate-800 text-xs text-slate-500 dark:text-slate-400">
            <Clock size={12} />
            <p>
              La cronologia viene salvata solo per questa sessione. I voti
              confermati sono inviati al server.
            </p>
          </div>
        )}
      </div>
    </Modal>
  );
}

function VoteHistoryItem({
  entry,
  index,
  onUndo,
  showUndo = false,
}: {
  entry: VoteHistoryEntry;
  index: number;
  onUndo?: (id: string) => void;
  showUndo?: boolean;
}) {
  const Icon = voteIcons[entry.vote];
  const colors = voteColors[entry.vote];

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className={cn(
        'flex items-center gap-3 p-2.5 rounded-lg',
        'bg-white dark:bg-slate-800/50',
        'border border-slate-100 dark:border-slate-800'
      )}
    >
      {/* Vote indicator */}
      <div className={cn('w-8 h-8 rounded-lg flex items-center justify-center', colors.bg)}>
        <Icon size={14} className={colors.text} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
          {entry.itemName}
        </p>
        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          <span>{typeLabels[entry.itemType]}</span>
          <span>•</span>
          <span className={colors.text}>{voteLabels[entry.vote]}</span>
          <span>•</span>
          <span>{formatRelativeTime(entry.timestamp)}</span>
        </div>
      </div>

      {/* Status/Actions */}
      {entry.isCommitted ? (
        <div className="flex items-center gap-1 text-emerald-500">
          <Check size={12} />
        </div>
      ) : showUndo && onUndo ? (
        <button
          onClick={() => onUndo(entry.id)}
          className={cn(
            'p-1.5 rounded-md text-slate-400 hover:text-red-500',
            'hover:bg-red-50 dark:hover:bg-red-900/20',
            'transition-colors'
          )}
          title="Annulla voto"
        >
          <Undo2 size={14} />
        </button>
      ) : (
        <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
      )}
    </motion.div>
  );
}
