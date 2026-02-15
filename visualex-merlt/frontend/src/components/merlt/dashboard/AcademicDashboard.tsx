/**
 * AcademicDashboard
 * =================
 *
 * Dashboard accademica completa per MERL-T che mostra in modo granulare
 * tutti i processi del sistema Legal Knowledge Graph + RLCF.
 *
 * Ottimizzata per:
 * - Tesi di laurea in "Sociologia Computazionale del Diritto"
 * - Rigore accademico con statistiche, p-values, effect sizes, CI
 * - Visualizzazioni architetturali interattive
 * - Monitoraggio real-time di tutti i processi
 * - Export per analisi esterne (CSV, JSON, LaTeX)
 *
 * Tabs:
 * 1. Overview - KPIs aggregati e system status
 * 2. Architecture - Diagramma interattivo react-flow
 * 3. Pipeline - Monitoring ingestion/enrichment
 * 4. RLCF Training - Training status e policy weights
 * 5. Experts - Performance multi-expert system
 * 6. Statistics - Hypothesis testing con rigore accademico
 *
 * @example
 * ```tsx
 * <AcademicDashboard />
 * ```
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  Network,
  Workflow,
  Brain,
  Users,
  BarChart3,
  RefreshCw,
  Download,
  Calendar,
} from 'lucide-react';
import { cn } from '../../../lib/utils';

// Tab components
import { OverviewTab } from './tabs/OverviewTab';
import { ArchitectureTab } from './tabs/ArchitectureTab';
import { PipelineTab } from './tabs/PipelineTab';
import { RLCFTab } from './tabs/RLCFTab';
import { ExpertsTab } from './tabs/ExpertsTab';
import { StatisticsTab } from './tabs/StatisticsTab';

// =============================================================================
// TYPES
// =============================================================================

type TabId = 'overview' | 'architecture' | 'pipeline' | 'rlcf' | 'experts' | 'statistics';

interface Tab {
  id: TabId;
  label: string;
  icon: React.ReactNode;
  description: string;
}

// =============================================================================
// TABS CONFIG
// =============================================================================

const TABS: Tab[] = [
  {
    id: 'overview',
    label: 'Overview',
    icon: <LayoutDashboard size={18} />,
    description: 'KPIs aggregati e system status',
  },
  {
    id: 'architecture',
    label: 'Architecture',
    icon: <Network size={18} />,
    description: 'Diagramma architettura interattivo',
  },
  {
    id: 'pipeline',
    label: 'Pipeline',
    icon: <Workflow size={18} />,
    description: 'Monitoring ingestion/enrichment',
  },
  {
    id: 'rlcf',
    label: 'RLCF Training',
    icon: <Brain size={18} />,
    description: 'Training status e policy weights',
  },
  {
    id: 'experts',
    label: 'Experts',
    icon: <Users size={18} />,
    description: 'Performance multi-expert system',
  },
  {
    id: 'statistics',
    label: 'Statistics',
    icon: <BarChart3 size={18} />,
    description: 'Hypothesis testing accademico',
  },
];

// =============================================================================
// TAB BUTTON
// =============================================================================

interface TabButtonProps {
  tab: Tab;
  isActive: boolean;
  onClick: () => void;
}

function TabButton({ tab, isActive, onClick }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      role="tab"
      aria-selected={isActive}
      className={cn(
        'relative flex items-center gap-2 px-4 py-2.5 text-sm font-medium',
        'rounded-lg transition-all duration-200',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
        isActive
          ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
          : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800'
      )}
      title={tab.description}
    >
      <span aria-hidden="true">{tab.icon}</span>
      <span className="hidden sm:inline">{tab.label}</span>
      <span className="sr-only sm:hidden">{tab.label}</span>

      {/* Active indicator */}
      {isActive && (
        <motion.div
          layoutId="activeTab"
          className="absolute inset-0 bg-blue-500/10 rounded-lg -z-10"
          initial={false}
          transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
        />
      )}
    </button>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface AcademicDashboardProps {
  className?: string;
  initialTab?: TabId;
}

export function AcademicDashboard({
  className,
  initialTab = 'overview',
}: AcademicDashboardProps) {
  const [activeTab, setActiveTab] = useState(initialTab);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Update timestamp when tab changes
  useEffect(() => {
    setLastUpdated(new Date());
  }, [activeTab]);

  const formatLastUpdated = () => {
    return lastUpdated.toLocaleTimeString('it-IT', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab />;
      case 'architecture':
        return <ArchitectureTab />;
      case 'pipeline':
        return <PipelineTab />;
      case 'rlcf':
        return <RLCFTab />;
      case 'experts':
        return <ExpertsTab />;
      case 'statistics':
        return <StatisticsTab />;
      default:
        return <OverviewTab />;
    }
  };

  return (
    <div className={cn('min-h-screen bg-slate-50 dark:bg-slate-900', className)}>
      {/* Header */}
      <div className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Title row */}
          <div className="py-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                MERL-T Academic Dashboard
              </h1>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                Legal Knowledge Graph + RLCF Monitoring
              </p>
            </div>

            <div className="flex items-center gap-3">
              {/* Last updated */}
              <span className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                <Calendar size={14} aria-hidden="true" />
                Ultimo aggiornamento: {formatLastUpdated()}
              </span>

              {/* Refresh button */}
              <button
                onClick={() => setLastUpdated(new Date())}
                className={cn(
                  'p-2 rounded-lg transition-colors',
                  'text-slate-500 hover:text-slate-700 hover:bg-slate-100',
                  'dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-700',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                )}
                aria-label="Aggiorna dati"
              >
                <RefreshCw size={18} aria-hidden="true" />
              </button>

              {/* Export button */}
              <button
                className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium',
                  'bg-blue-600 text-white hover:bg-blue-700 transition-colors',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2'
                )}
                aria-label="Esporta dati"
              >
                <Download size={16} aria-hidden="true" />
                <span className="hidden sm:inline">Export</span>
              </button>
            </div>
          </div>

          {/* Tabs navigation */}
          <div className="flex items-center gap-1 pb-2 overflow-x-auto scrollbar-hide" role="tablist" aria-label="Dashboard tabs">
            {TABS.map((tab) => (
              <TabButton
                key={tab.id}
                tab={tab}
                isActive={activeTab === tab.id}
                onClick={() => setActiveTab(tab.id)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Tab content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            role="tabpanel"
            aria-label={TABS.find(t => t.id === activeTab)?.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {renderTabContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

export default AcademicDashboard;
