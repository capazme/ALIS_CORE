/**
 * ConsentSelector Component
 * =========================
 *
 * Displays the 3 GDPR-compliant consent levels for RLCF data participation.
 * Users can choose how their data contributes to system learning.
 *
 * Consent levels:
 * - 🔒 Base: No data collected beyond session
 * - 📊 Apprendimento: Anonymized queries + feedback for RLCF training
 * - 🔬 Ricerca: Aggregated data available for academic analysis
 */
import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertTriangle, Shield, Clock, ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '../../../lib/utils';
import * as consentService from '../../../services/consentService';
import type { ConsentLevel, ConsentLevelDescription, ConsentHistoryEntry } from '../../../types/api';

export interface ConsentSelectorProps {
  onConsentChange?: (consentLevel: ConsentLevel) => void;
  className?: string;
}

const CONSENT_ICONS: Record<ConsentLevel, string> = {
  basic: '🔒',
  learning: '📊',
  research: '🔬',
};

const CONSENT_COLORS: Record<ConsentLevel, { bg: string; border: string; text: string }> = {
  basic: {
    bg: 'bg-slate-50 dark:bg-slate-900/50',
    border: 'border-slate-300 dark:border-slate-600',
    text: 'text-slate-700 dark:text-slate-300',
  },
  learning: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-300 dark:border-blue-700',
    text: 'text-blue-700 dark:text-blue-300',
  },
  research: {
    bg: 'bg-purple-50 dark:bg-purple-900/20',
    border: 'border-purple-300 dark:border-purple-700',
    text: 'text-purple-700 dark:text-purple-300',
  },
};

// Consent level hierarchy (for downgrade detection)
const CONSENT_HIERARCHY: Record<ConsentLevel, number> = {
  basic: 0,
  learning: 1,
  research: 2,
};

export function ConsentSelector({
  onConsentChange,
  className,
}: ConsentSelectorProps) {
  const [selectedConsent, setSelectedConsent] = useState<ConsentLevel>('basic');
  const [consentLevels, setConsentLevels] = useState<ConsentLevelDescription[]>([]);
  const [grantedAt, setGrantedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<ConsentHistoryEntry[]>([]);
  const [pendingConsent, setPendingConsent] = useState<ConsentLevel | null>(null);

  // Load consent on mount
  useEffect(() => {
    const loadConsent = async () => {
      try {
        const response = await consentService.getConsent();
        setConsentLevels(response.available_levels);
        setSelectedConsent(response.consent_level);
        setGrantedAt(response.granted_at);
        setLoading(false);
      } catch {
        // Use default levels if API fails
        const defaultLevels: ConsentLevelDescription[] = [
          {
            level: 'basic',
            emoji: '🔒',
            name: 'Base',
            description: 'Nessun dato raccolto oltre la sessione. Solo uso del sistema.',
            dataCollected: ['Nessuno'],
          },
          {
            level: 'learning',
            emoji: '📊',
            name: 'Apprendimento',
            description: 'Query anonimizzate + feedback usati per il training RLCF.',
            dataCollected: ['Query anonimizzate', 'Feedback', 'Interazioni di ricerca'],
          },
          {
            level: 'research',
            emoji: '🔬',
            name: 'Ricerca',
            description: 'Dati aggregati disponibili per analisi accademica.',
            dataCollected: ['Query anonimizzate', 'Feedback', 'Pattern di utilizzo', 'Dati aggregati per ricerca'],
          },
        ];
        setConsentLevels(defaultLevels);
        setLoading(false);
      }
    };

    loadConsent();
  }, []);

  // Load history when expanded
  useEffect(() => {
    if (showHistory && history.length === 0) {
      const loadHistory = async () => {
        try {
          const response = await consentService.getConsentHistory();
          setHistory(response.history);
        } catch {
          // Ignore history load errors
        }
      };
      loadHistory();
    }
  }, [showHistory, history.length]);

  // Handle consent selection
  const handleSelectConsent = useCallback(
    (consentLevel: ConsentLevel) => {
      if (consentLevel === selectedConsent) return;

      // Check if this is a downgrade
      const isDowngrade = CONSENT_HIERARCHY[consentLevel] < CONSENT_HIERARCHY[selectedConsent];

      if (isDowngrade) {
        // Show confirmation for downgrade
        setPendingConsent(consentLevel);
        return;
      }

      // Proceed with update
      confirmConsentUpdate(consentLevel);
    },
    [selectedConsent]
  );

  // Confirm consent update (called directly or after downgrade confirmation)
  const confirmConsentUpdate = useCallback(
    async (consentLevel: ConsentLevel) => {
      setSaving(true);
      setError(null);
      setSuccess(false);
      setWarning(null);
      setPendingConsent(null);

      try {
        const response = await consentService.updateConsent(consentLevel);
        setSelectedConsent(response.consent_level);
        setGrantedAt(response.granted_at);
        setSuccess(true);

        if (response.warning) {
          setWarning(response.warning);
        }

        onConsentChange?.(consentLevel);

        // Clear success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Errore nel salvataggio del consenso';
        setError(message);
      } finally {
        setSaving(false);
      }
    },
    [onConsentChange]
  );

  // Cancel pending downgrade
  const cancelDowngrade = useCallback(() => {
    setPendingConsent(null);
  }, []);

  if (loading) {
    return (
      <div className={cn('space-y-4', className)}>
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-28 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 animate-pulse"
          />
        ))}
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Current consent info */}
      {grantedAt && (
        <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
          <Clock size={14} />
          <span>
            Ultimo aggiornamento: {new Date(grantedAt).toLocaleDateString('it-IT', {
              day: 'numeric',
              month: 'long',
              year: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
        </div>
      )}

      {/* Consent Level Cards */}
      <div className="space-y-3">
        {consentLevels.map((level) => {
          const isSelected = selectedConsent === level.level;
          const colors = CONSENT_COLORS[level.level];

          return (
            <motion.button
              key={level.level}
              onClick={() => handleSelectConsent(level.level)}
              disabled={saving}
              className={cn(
                'relative w-full p-4 rounded-xl border-2 text-left transition-all',
                'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500',
                isSelected
                  ? cn(colors.bg, colors.border)
                  : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600',
                saving && 'cursor-wait opacity-70'
              )}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
            >
              {/* Selected indicator */}
              {isSelected && (
                <div className="absolute top-4 right-4">
                  <CheckCircle className="w-5 h-5 text-emerald-500" />
                </div>
              )}

              {/* Content */}
              <div className="flex items-start gap-3 pr-8">
                <span className="text-2xl" role="img" aria-label={level.name}>
                  {CONSENT_ICONS[level.level]}
                </span>
                <div className="flex-1">
                  <h4
                    className={cn(
                      'font-semibold',
                      isSelected ? colors.text : 'text-slate-800 dark:text-slate-100'
                    )}
                  >
                    {level.name}
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    {level.description}
                  </p>

                  {/* Data collected list */}
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {level.dataCollected.map((data, idx) => (
                      <span
                        key={idx}
                        className={cn(
                          'text-xs px-2 py-0.5 rounded-full',
                          isSelected
                            ? 'bg-white/50 dark:bg-black/20 text-slate-600 dark:text-slate-300'
                            : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400'
                        )}
                      >
                        {data}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Downgrade Confirmation Modal */}
      <AnimatePresence>
        {pendingConsent && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/30"
          >
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-medium text-amber-800 dark:text-amber-200">
                  Confermi la riduzione del consenso?
                </h4>
                <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                  I dati precedentemente raccolti (se presenti) rimarranno fino a quando non
                  richiederai la cancellazione nelle impostazioni privacy.
                </p>
                <div className="flex gap-2 mt-3">
                  <button
                    onClick={() => confirmConsentUpdate(pendingConsent)}
                    disabled={saving}
                    className="px-3 py-1.5 text-sm font-medium rounded-md bg-amber-600 text-white hover:bg-amber-700 transition-colors"
                  >
                    {saving ? 'Salvataggio...' : 'Conferma'}
                  </button>
                  <button
                    onClick={cancelDowngrade}
                    disabled={saving}
                    className="px-3 py-1.5 text-sm font-medium rounded-md bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
                  >
                    Annulla
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Warning message (from downgrade) */}
      {warning && !pendingConsent && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/30"
        >
          <p className="text-sm text-amber-700 dark:text-amber-300 flex items-start gap-2">
            <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" />
            {warning}
          </p>
        </motion.div>
      )}

      {/* Error message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30"
        >
          <p className="text-sm text-red-600 dark:text-red-400 flex items-center gap-2">
            <Shield size={16} />
            {error}
          </p>
        </motion.div>
      )}

      {/* Success message */}
      {success && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800/30"
        >
          <p className="text-sm text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
            <CheckCircle size={16} />
            Consenso aggiornato con successo
          </p>
        </motion.div>
      )}

      {/* History toggle */}
      <button
        onClick={() => setShowHistory(!showHistory)}
        className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
      >
        {showHistory ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        Cronologia modifiche
      </button>

      {/* History section */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="space-y-2 pt-2">
              {history.length === 0 ? (
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Nessuna modifica registrata
                </p>
              ) : (
                history.slice(0, 5).map((entry) => (
                  <div
                    key={entry.id}
                    className="flex items-center justify-between text-sm p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50"
                  >
                    <div className="flex items-center gap-2">
                      {entry.previous_level ? (
                        <>
                          <span className="text-slate-500">{CONSENT_ICONS[entry.previous_level]}</span>
                          <span className="text-slate-400">→</span>
                        </>
                      ) : (
                        <span className="text-slate-400 text-xs">Iniziale:</span>
                      )}
                      <span>{CONSENT_ICONS[entry.new_level]}</span>
                      <span className="text-slate-600 dark:text-slate-400">
                        {consentLevels.find((l) => l.level === entry.new_level)?.name}
                      </span>
                    </div>
                    <span className="text-xs text-slate-400">
                      {new Date(entry.changed_at).toLocaleDateString('it-IT')}
                    </span>
                  </div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default ConsentSelector;
