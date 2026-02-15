/**
 * ExpertProgressIndicator - Container showing overall progress bar + 4 expert blocks.
 */

import { motion } from 'framer-motion';
import { Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { cn } from '../../lib/utils';
import { ExpertStatusBlock } from './ExpertStatusBlock';
import type { ExpertPipelineStatus } from '../../types/pipeline';

export interface ExpertProgressIndicatorProps {
  status: ExpertPipelineStatus;
  className?: string;
}

function PhaseLabel({ phase }: { phase: ExpertPipelineStatus['phase'] }) {
  switch (phase) {
    case 'routing': return <span>Routing query agli esperti...</span>;
    case 'expert_analysis': return <span>Analisi esperti in corso...</span>;
    case 'synthesis': return <span>Sintesi delle risposte...</span>;
    case 'completed': return <span className="text-emerald-600 dark:text-emerald-400">Analisi completata</span>;
    case 'failed': return <span className="text-red-600 dark:text-red-400">Errore durante l'analisi</span>;
    default: return null;
  }
}

export function ExpertProgressIndicator({ status, className }: ExpertProgressIndicatorProps) {
  const isActive = status.phase !== 'completed' && status.phase !== 'failed';
  const isCompleted = status.phase === 'completed';
  const isFailed = status.phase === 'failed';

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("space-y-3", className)}
    >
      {/* Header with phase label */}
      <div className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400" role="status">
        {isActive && <Loader2 size={14} className="animate-spin text-blue-500" aria-hidden="true" />}
        {isCompleted && <CheckCircle2 size={14} className="text-emerald-500" aria-hidden="true" />}
        {isFailed && <AlertCircle size={14} className="text-red-500" aria-hidden="true" />}
        <PhaseLabel phase={status.phase} />
      </div>

      {/* Overall progress bar */}
      <div
        className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden"
        role="progressbar"
        aria-valuenow={Math.round(status.overallProgress)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Progresso analisi: ${Math.round(status.overallProgress)}%`}
      >
        <motion.div
          className={cn(
            "h-full rounded-full",
            isFailed ? "bg-red-500" : isCompleted ? "bg-emerald-500" : "bg-blue-500"
          )}
          initial={{ width: 0 }}
          animate={{ width: `${status.overallProgress}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>

      {/* Expert status blocks */}
      <div className="grid grid-cols-2 gap-2">
        {status.experts.map(expert => (
          <ExpertStatusBlock key={expert.id} expert={expert} />
        ))}
      </div>

      {/* Synthesis status */}
      {status.phase === 'synthesis' && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-purple-200 dark:border-purple-800 bg-purple-50/50 dark:bg-purple-900/20">
          <Loader2 size={12} className="animate-spin text-purple-500" aria-hidden="true" />
          <span className="text-xs font-medium text-purple-700 dark:text-purple-300">
            Sintesi in elaborazione...
          </span>
        </div>
      )}
    </motion.div>
  );
}
