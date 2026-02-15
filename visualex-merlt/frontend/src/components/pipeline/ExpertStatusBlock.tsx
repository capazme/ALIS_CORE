/**
 * ExpertStatusBlock - Single expert status indicator with icon, name, and status badge.
 */

import { BookOpen, TrendingUp, Scale, Gavel, Loader2, Check, X, Clock } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { ExpertStatus, ExpertId } from '../../types/pipeline';
import { EXPERT_CONFIG } from '../../types/pipeline';

export interface ExpertStatusBlockProps {
  expert: ExpertStatus;
}

const ICON_MAP: Record<string, React.ComponentType<{ size?: number; className?: string; style?: React.CSSProperties }>> = {
  BookOpen,
  TrendingUp,
  Scale,
  Gavel,
};

function StatusIndicator({ status }: { status: ExpertStatus['status'] }) {
  switch (status) {
    case 'running':
      return <Loader2 size={12} className="animate-spin text-blue-500" aria-hidden="true" />;
    case 'completed':
      return <Check size={12} className="text-emerald-500" aria-hidden="true" />;
    case 'failed':
      return <X size={12} className="text-red-500" aria-hidden="true" />;
    case 'skipped':
      return <span className="text-[10px] text-slate-400" aria-hidden="true">skip</span>;
    default:
      return <Clock size={12} className="text-slate-400" aria-hidden="true" />;
  }
}

export function ExpertStatusBlock({ expert }: ExpertStatusBlockProps) {
  const config = EXPERT_CONFIG[expert.id];
  const IconComponent = ICON_MAP[config.icon] || BookOpen;

  return (
    <div
      aria-label={`${config.displayName}: ${expert.status}`}
      className={cn(
        "flex items-center gap-2 px-2.5 py-2 rounded-lg border transition-all",
        expert.status === 'running'
          ? "border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/20"
          : expert.status === 'completed'
          ? "border-emerald-200 dark:border-emerald-800 bg-emerald-50/50 dark:bg-emerald-900/20"
          : expert.status === 'failed'
          ? "border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-900/20"
          : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"
      )}
    >
      {/* Expert icon */}
      <div
        className="w-7 h-7 rounded-md flex items-center justify-center shrink-0"
        style={{ backgroundColor: `${config.color}20` }}
      >
        <IconComponent size={14} className="text-current" style={{ color: config.color }} />
      </div>

      {/* Name + status */}
      <div className="flex-1 min-w-0">
        <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 truncate">
          {config.displayName}
        </div>
        {expert.status === 'running' && expert.progress !== undefined && (
          <div className="w-full h-1 bg-slate-200 dark:bg-slate-700 rounded-full mt-1 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${expert.progress}%`,
                backgroundColor: config.color,
              }}
            />
          </div>
        )}
      </div>

      {/* Status indicator */}
      <StatusIndicator status={expert.status} />
    </div>
  );
}
