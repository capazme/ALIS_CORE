/**
 * AuthorityScoreDisplay Component
 * ================================
 *
 * Displays the user's authority score with breakdown of components.
 * Authority determines influence on RLCF system learning.
 *
 * Components:
 * - Baseline (30%): From profile/credentials
 * - Track Record (50%): Historical feedback accuracy
 * - Recent Performance (20%): Last N feedback quality
 */
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Info, TrendingUp, AlertCircle } from 'lucide-react';
import { cn } from '../../../lib/utils';
import * as authorityService from '../../../services/authorityService';
import type { AuthorityResponse, AuthorityComponentDescription } from '../../../types/api';

export interface AuthorityScoreDisplayProps {
  className?: string;
}

// Color mapping based on score
const getScoreColor = (score: number): string => {
  if (score >= 0.7) return 'text-emerald-500';
  if (score >= 0.4) return 'text-blue-500';
  if (score >= 0.2) return 'text-amber-500';
  return 'text-slate-400';
};

const getProgressColor = (score: number): string => {
  if (score >= 0.7) return 'bg-emerald-500';
  if (score >= 0.4) return 'bg-blue-500';
  if (score >= 0.2) return 'bg-amber-500';
  return 'bg-slate-400';
};

// Component card
interface ComponentCardProps {
  component: AuthorityComponentDescription;
  label: string;
}

function ComponentCard({ component, label }: ComponentCardProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative">
      <button
        type="button"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={() => setShowTooltip(!showTooltip)}
        className={cn(
          'w-full p-3 rounded-lg border text-left transition-all',
          'bg-white dark:bg-slate-800/50',
          'border-slate-200 dark:border-slate-700',
          'hover:border-slate-300 dark:hover:border-slate-600',
          'focus:outline-none focus:ring-2 focus:ring-blue-500'
        )}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-lg">{component.icon}</span>
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
              {component.name}
            </span>
          </div>
          <Info size={14} className="text-slate-400" />
        </div>

        <div className="flex items-end justify-between">
          <div>
            <span className={cn('text-xl font-bold', getScoreColor(component.score))}>
              {(component.score * 100).toFixed(0)}%
            </span>
            <span className="text-xs text-slate-400 ml-1">
              ({(component.weight * 100).toFixed(0)}% peso)
            </span>
          </div>
          <span className="text-xs text-slate-500">
            +{component.weighted.toFixed(2)}
          </span>
        </div>

        {/* Progress bar */}
        <div className="mt-2 h-1.5 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${component.score * 100}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className={cn('h-full rounded-full', getProgressColor(component.score))}
          />
        </div>
      </button>

      {/* Tooltip */}
      {showTooltip && (
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className={cn(
            'absolute z-10 left-0 right-0 mt-2 p-3 rounded-lg shadow-lg',
            'bg-slate-900 dark:bg-slate-700 text-white',
            'text-sm'
          )}
        >
          <p>{component.description}</p>
          <div className="mt-2 pt-2 border-t border-slate-700 dark:border-slate-600">
            <p className="text-xs text-slate-400">
              Contributo: {component.score.toFixed(2)} × {component.weight} = {component.weighted.toFixed(3)}
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
}

export function AuthorityScoreDisplay({ className }: AuthorityScoreDisplayProps) {
  const [authority, setAuthority] = useState<AuthorityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadAuthority = async () => {
      try {
        const response = await authorityService.getAuthority();
        setAuthority(response);
        setLoading(false);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Errore nel caricamento';
        setError(message);
        setLoading(false);
      }
    };

    loadAuthority();
  }, []);

  if (loading) {
    return (
      <div className={cn('space-y-4', className)}>
        <div className="h-24 rounded-xl bg-slate-100 dark:bg-slate-800 animate-pulse" />
        <div className="grid grid-cols-3 gap-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-28 rounded-lg bg-slate-100 dark:bg-slate-800 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30', className)}>
        <p className="text-sm text-red-600 dark:text-red-400 flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
        </p>
      </div>
    );
  }

  if (!authority) return null;

  return (
    <div className={cn('space-y-4', className)}>
      {/* Main score display */}
      <div className="p-4 rounded-xl bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-900 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h4 className="text-sm font-medium text-slate-600 dark:text-slate-400">
              Punteggio Autorità
            </h4>
            <div className="flex items-baseline gap-2 mt-1">
              <span className={cn('text-4xl font-bold', getScoreColor(authority.authority_score))}>
                {(authority.authority_score * 100).toFixed(0)}
              </span>
              <span className="text-lg text-slate-400">/100</span>
            </div>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-1 text-sm text-slate-500">
              <TrendingUp size={14} />
              <span>{authority.feedback_count} feedback</span>
            </div>
          </div>
        </div>

        {/* Overall progress bar */}
        <div className="h-2.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${authority.authority_score * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className={cn('h-full rounded-full', getProgressColor(authority.authority_score))}
          />
        </div>
      </div>

      {/* New user message */}
      {authority.message && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800/30"
        >
          <p className="text-sm text-blue-700 dark:text-blue-300 flex items-center gap-2">
            <Info size={16} />
            {authority.message}
          </p>
        </motion.div>
      )}

      {/* Component breakdown */}
      <div>
        <h5 className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-3">
          Composizione del punteggio
        </h5>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <ComponentCard
            component={authority.components.baseline}
            label="baseline"
          />
          <ComponentCard
            component={authority.components.track_record}
            label="track_record"
          />
          <ComponentCard
            component={authority.components.recent_performance}
            label="recent_performance"
          />
        </div>
      </div>

      {/* Formula explanation */}
      <div className="pt-3 border-t border-slate-200 dark:border-slate-700">
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Formula: A = 0.3 × Baseline + 0.5 × Storico + 0.2 × Recente
        </p>
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
          Il tuo punteggio autorità determina quanto i tuoi feedback influenzano l'apprendimento del sistema RLCF.
        </p>
      </div>
    </div>
  );
}

export default AuthorityScoreDisplay;
