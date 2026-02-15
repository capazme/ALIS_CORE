/**
 * ValidationKeyboardHelpModal
 * ===========================
 *
 * Shows keyboard shortcuts specific to the validation panel.
 * Triggered by pressing '?' while the panel is open.
 */

import { Modal } from '../ui/Modal';
import { Keyboard, ThumbsUp, ThumbsDown, Edit3, SkipForward, ArrowUp, ArrowDown, HelpCircle, X } from 'lucide-react';
import { cn } from '../../lib/utils';
import { VALIDATION_SHORTCUTS, type KeyboardShortcut } from '../../hooks/useValidationKeyboard';

interface ValidationKeyboardHelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function KeyBadge({ children }: { children: React.ReactNode }) {
  return (
    <kbd
      className={cn(
        'inline-flex items-center justify-center min-w-[24px] h-6 px-1.5',
        'text-xs font-medium',
        'bg-slate-100 dark:bg-slate-800',
        'border border-slate-300 dark:border-slate-600',
        'rounded shadow-sm',
        'text-slate-700 dark:text-slate-300'
      )}
    >
      {children}
    </kbd>
  );
}

const categoryLabels: Record<string, string> = {
  voting: 'Votazione',
  navigation: 'Navigazione',
  other: 'Altro',
};

const categoryIcons: Record<string, typeof ThumbsUp> = {
  voting: ThumbsUp,
  navigation: ArrowUp,
  other: HelpCircle,
};

export function ValidationKeyboardHelpModal({
  isOpen,
  onClose,
}: ValidationKeyboardHelpModalProps) {
  // Group shortcuts by category
  const groupedShortcuts = VALIDATION_SHORTCUTS.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = [];
    }
    acc[shortcut.category].push(shortcut);
    return acc;
  }, {} as Record<string, KeyboardShortcut[]>);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Scorciatoie Validazione" size="sm">
      <div className="space-y-6">
        {/* Header info */}
        <div className="flex items-center gap-3 p-3 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-100 dark:border-primary-900/30">
          <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
            <Keyboard size={20} className="text-primary-600 dark:text-primary-400" aria-hidden="true" />
          </div>
          <div>
            <p className="text-sm font-medium text-primary-900 dark:text-primary-100">
              Valida velocemente con la tastiera
            </p>
            <p className="text-xs text-primary-600 dark:text-primary-400">
              Usa le scorciatoie per navigare e votare
            </p>
          </div>
        </div>

        {/* Shortcut groups */}
        <div className="space-y-4">
          {(['voting', 'navigation', 'other'] as const).map((category) => {
            const shortcuts = groupedShortcuts[category];
            if (!shortcuts || shortcuts.length === 0) return null;

            const Icon = categoryIcons[category];

            return (
              <div key={category}>
                <div className="flex items-center gap-2 mb-2">
                  <Icon size={12} className="text-slate-400" />
                  <h3 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    {categoryLabels[category]}
                  </h3>
                </div>
                <div className="divide-y divide-slate-100 dark:divide-slate-800">
                  {shortcuts.map((shortcut) => (
                    <div
                      key={shortcut.key}
                      className="flex items-center justify-between py-2"
                    >
                      <span className="text-sm text-slate-600 dark:text-slate-400">
                        {shortcut.description}
                      </span>
                      <div className="flex items-center gap-1">
                        <KeyBadge>{shortcut.label}</KeyBadge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Tips */}
        <div className="pt-4 border-t border-slate-100 dark:border-slate-800 space-y-2">
          <p className="text-xs text-slate-500 dark:text-slate-400">
            <strong>Suggerimento:</strong> Usa <KeyBadge>J</KeyBadge> e{' '}
            <KeyBadge>K</KeyBadge> per scorrere rapidamente gli elementi, poi{' '}
            <KeyBadge>A</KeyBadge> o <KeyBadge>R</KeyBadge> per votare.
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Premi <KeyBadge>S</KeyBadge> per saltare elementi di cui non sei sicuro e
            rivederli alla fine.
          </p>
        </div>
      </div>
    </Modal>
  );
}
