/**
 * PipelineTab
 * ===========
 *
 * Tab wrapper per il Pipeline Monitoring esistente + Ingestion Schedules.
 * Riutilizza PipelineMonitoringDashboard per coerenza.
 *
 * @example
 * ```tsx
 * <PipelineTab />
 * ```
 */

import { PipelineMonitoringDashboard } from '../../monitoring/PipelineMonitoringDashboard';
import { SchedulesSection } from './SchedulesSection';

export function PipelineTab() {
  return (
    <div className="space-y-6">
      <PipelineMonitoringDashboard />
      <SchedulesSection />
    </div>
  );
}

export default PipelineTab;
