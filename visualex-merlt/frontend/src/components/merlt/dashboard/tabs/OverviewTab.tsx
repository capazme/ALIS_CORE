/**
 * OverviewTab
 * ===========
 *
 * Tab Overview della dashboard accademica che mostra:
 * - KPI cards per Knowledge Graph, RLCF, Expert System
 * - System health status
 * - Activity feed recente
 * - Mini architecture diagram preview
 *
 * @example
 * ```tsx
 * <OverviewTab />
 * ```
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  GitBranch,
  FileText,
  Box,
  Cpu,
  Link2,
  Users,
  MessageSquare,
  TrendingUp,
  Activity,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Clock,
  RefreshCw,
} from 'lucide-react';
import { cn } from '../../../../../lib/utils';
import {
  getDashboardOverview,
  getStatusColor,
  getSeverityColor,
  formatUptime,
  type DashboardOverview,
  type ServiceStatus,
  type ActivityEntry,
} from '../../../../../services/dashboardService';

// =============================================================================
// KPI CARD
// =============================================================================

interface KPICardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  subtitle?: string;
  trend?: number;
  color?: 'blue' | 'green' | 'purple' | 'orange' | 'cyan';
}

function KPICard({ title, value, icon, subtitle, trend, color = 'blue' }: KPICardProps) {
  const colorClasses = {
    blue: 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30',
    green: 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30',
    purple: 'text-purple-600 dark:text-purple-400 bg-purple-100 dark:bg-purple-900/30',
    orange: 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30',
    cyan: 'text-cyan-600 dark:text-cyan-400 bg-cyan-100 dark:bg-cyan-900/30',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow"
    >
      <div className="flex items-start justify-between">
        <div className={cn('p-2.5 rounded-lg', colorClasses[color])}>
          {icon}
        </div>
        {trend !== undefined && (
          <span
            className={cn(
              'flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full',
              trend >= 0
                ? 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30'
                : 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30'
            )}
          >
            <TrendingUp size={12} className={cn(trend < 0 && 'rotate-180')} />
            {Math.abs(trend)}%
          </span>
        )}
      </div>

      <div className="mt-4">
        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
          {title}
        </h3>
        <p className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
          {typeof value === 'number' ? value.toLocaleString('it-IT') : value}
        </p>
        {subtitle && (
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
            {subtitle}
          </p>
        )}
      </div>
    </motion.div>
  );
}

// =============================================================================
// SERVICE STATUS
// =============================================================================

interface ServiceStatusCardProps {
  name: string;
  status: ServiceStatus;
  latencyMs?: number;
  lastCheck: string;
}

function ServiceStatusCard({ name, status, latencyMs }: ServiceStatusCardProps) {
  const statusIcon = {
    online: <CheckCircle2 size={16} className="text-green-500" />,
    offline: <XCircle size={16} className="text-red-500" />,
    degraded: <AlertCircle size={16} className="text-yellow-500" />,
    unknown: <Clock size={16} className="text-gray-500" />,
  };

  const statusLabel = {
    online: 'Online',
    offline: 'Offline',
    degraded: 'Degradato',
    unknown: 'Sconosciuto',
  };

  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 dark:border-gray-700 last:border-0">
      <div className="flex items-center gap-3">
        {statusIcon[status]}
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          {name}
        </span>
      </div>
      <div className="flex items-center gap-4">
        {latencyMs !== undefined && (
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {latencyMs}ms
          </span>
        )}
        <span className={cn('text-xs font-medium', getStatusColor(status))}>
          {statusLabel[status]}
        </span>
      </div>
    </div>
  );
}

// =============================================================================
// ACTIVITY ITEM
// =============================================================================

interface ActivityItemProps {
  entry: ActivityEntry;
}

function ActivityItem({ entry }: ActivityItemProps) {
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('it-IT', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="flex items-start gap-3 py-2 border-b border-gray-100 dark:border-gray-700 last:border-0">
      <div className="text-xs text-gray-400 dark:text-gray-500 w-12 flex-shrink-0">
        {formatTime(entry.timestamp)}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-700 dark:text-gray-300 truncate">
          {entry.message}
        </p>
        <span className={cn('inline-block text-xs px-2 py-0.5 rounded mt-1', getSeverityColor(entry.severity))}>
          {entry.type.replace(/_/g, ' ')}
        </span>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function OverviewTab() {
  const [data, setData] = useState<DashboardOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const overview = await getDashboardOverview();
      setData(overview);
    } catch (err) {
      setError('Errore nel caricamento dei dati');
      console.error('Failed to load dashboard overview:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw size={24} className="animate-spin text-blue-500" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-12">
        <AlertCircle size={48} className="mx-auto text-red-400 mb-4" />
        <p className="text-gray-500 dark:text-gray-400">{error}</p>
        <button
          onClick={fetchData}
          className="mt-4 px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Riprova
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Knowledge Graph KPIs */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Knowledge Graph
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <KPICard
            title="Nodi Totali"
            value={data.knowledge_graph.total_nodes}
            icon={<Database size={20} />}
            color="blue"
          />
          <KPICard
            title="Relazioni"
            value={data.knowledge_graph.total_edges}
            icon={<GitBranch size={20} />}
            color="green"
          />
          <KPICard
            title="Articoli"
            value={data.knowledge_graph.articles_count}
            icon={<FileText size={20} />}
            color="purple"
          />
          <KPICard
            title="Entità"
            value={data.knowledge_graph.entities_count}
            icon={<Box size={20} />}
            color="orange"
          />
          <KPICard
            title="Embeddings"
            value={data.knowledge_graph.embeddings_count}
            icon={<Cpu size={20} />}
            color="cyan"
          />
          <KPICard
            title="Bridge Mappings"
            value={data.knowledge_graph.bridge_mappings}
            icon={<Link2 size={20} />}
            color="blue"
          />
        </div>
      </div>

      {/* RLCF + Expert KPIs */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* RLCF KPIs */}
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            RLCF System
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <KPICard
              title="Total Feedback"
              value={data.rlcf.total_feedback}
              icon={<MessageSquare size={20} />}
              color="purple"
            />
            <KPICard
              title="Buffer Size"
              value={data.rlcf.buffer_size}
              icon={<Database size={20} />}
              color="orange"
            />
            <KPICard
              title="Training Epochs"
              value={data.rlcf.training_epochs}
              icon={<Activity size={20} />}
              color="green"
            />
            <KPICard
              title="Avg Authority"
              value={data.rlcf.avg_authority.toFixed(2)}
              icon={<TrendingUp size={20} />}
              color="blue"
            />
            <KPICard
              title="Active Users"
              value={data.rlcf.active_users}
              icon={<Users size={20} />}
              color="cyan"
            />
          </div>
        </div>

        {/* Expert System KPIs */}
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Expert System
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <KPICard
              title="Total Queries"
              value={data.experts.total_queries}
              icon={<MessageSquare size={20} />}
              color="blue"
            />
            <KPICard
              title="Avg Latency"
              value={`${data.experts.avg_latency_ms}ms`}
              icon={<Clock size={20} />}
              color="orange"
            />
            <KPICard
              title="Avg Confidence"
              value={`${(data.experts.avg_confidence * 100).toFixed(1)}%`}
              icon={<TrendingUp size={20} />}
              color="green"
            />
            <KPICard
              title="Agreement Rate"
              value={`${(data.experts.agreement_rate * 100).toFixed(1)}%`}
              icon={<CheckCircle2 size={20} />}
              color="purple"
            />
          </div>
        </div>
      </div>

      {/* System Health + Activity */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* System Health */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              System Health
            </h2>
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  'w-2.5 h-2.5 rounded-full',
                  data.health.overall_status === 'online' && 'bg-green-500',
                  data.health.overall_status === 'degraded' && 'bg-yellow-500',
                  data.health.overall_status === 'offline' && 'bg-red-500'
                )}
              />
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Uptime: {formatUptime(data.health.uptime_seconds)}
              </span>
            </div>
          </div>

          <div className="space-y-1">
            {data.health.services.map((service) => (
              <ServiceStatusCard
                key={service.name}
                name={service.name}
                status={service.status}
                latencyMs={service.latency_ms}
                lastCheck={service.last_check}
              />
            ))}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Recent Activity
          </h2>

          <div className="space-y-1 max-h-64 overflow-y-auto">
            {data.recent_activity.entries.slice(0, 10).map((entry) => (
              <ActivityItem key={entry.id} entry={entry} />
            ))}
          </div>

          {data.recent_activity.has_more && (
            <button className="w-full mt-3 text-center text-sm text-blue-600 dark:text-blue-400 hover:underline">
              Mostra altre {data.recent_activity.total_count - 10} attività
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default OverviewTab;
