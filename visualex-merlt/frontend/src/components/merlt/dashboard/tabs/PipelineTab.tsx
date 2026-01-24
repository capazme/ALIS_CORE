/**
 * PipelineTab
 * ===========
 *
 * Tab wrapper per il Pipeline Monitoring esistente.
 * Riutilizza PipelineMonitoringDashboard per coerenza.
 *
 * @example
 * ```tsx
 * <PipelineTab />
 * ```
 */

import { PipelineMonitoringDashboard } from '../../monitoring/PipelineMonitoringDashboard';

export function PipelineTab() {
  return <PipelineMonitoringDashboard />;
}

export default PipelineTab;
