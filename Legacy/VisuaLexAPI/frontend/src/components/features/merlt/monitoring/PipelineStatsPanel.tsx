/**
 * PipelineStatsPanel
 * ==================
 *
 * Pannello con KPI aggregate per pipeline monitoring.
 * Mostra statistiche totali: runs attive, completate, fallite, items processati.
 */

import { Activity, CheckCircle2, XCircle, Clock, Database, AlertTriangle } from 'lucide-react';
import { cn } from '../../../../lib/utils';
import type { PipelineStats } from '../../../../hooks/usePipelineMonitoring';

interface PipelineStatsPanelProps {
  stats: PipelineStats;
  className?: string;
}

interface StatCardProps {
  label: string;
  value: number | string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'red' | 'amber' | 'purple';
}

function StatCard({ label, value, icon, color }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400',
    green: 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400',
    red: 'bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400',
    amber: 'bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400',
    purple: 'bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400',
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center gap-3">
        <div className={cn('p-2.5 rounded-lg', colorClasses[color])}>
          {icon}
        </div>
        <div>
          <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
        </div>
      </div>
    </div>
  );
}

export function PipelineStatsPanel({ stats, className }: PipelineStatsPanelProps) {
  return (
    <div className={cn('grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4', className)}>
      <StatCard
        label="Run Totali"
        value={stats.totalRuns}
        icon={<Database size={20} />}
        color="purple"
      />

      <StatCard
        label="Attive"
        value={stats.activeRuns}
        icon={<Activity size={20} />}
        color="blue"
      />

      <StatCard
        label="Completate"
        value={stats.completedRuns}
        icon={<CheckCircle2 size={20} />}
        color="green"
      />

      <StatCard
        label="Fallite"
        value={stats.failedRuns}
        icon={<XCircle size={20} />}
        color="red"
      />

      <StatCard
        label="Items Processati"
        value={stats.totalItemsProcessed}
        icon={<Clock size={20} />}
        color="blue"
      />

      <StatCard
        label="Errori Totali"
        value={stats.totalErrors}
        icon={<AlertTriangle size={20} />}
        color="amber"
      />
    </div>
  );
}

export default PipelineStatsPanel;
