/**
 * Pipeline Types for Expert Pipeline Status UI
 */

export type ExpertId = 'literal_interpreter' | 'systemic_teleological' | 'principles_balancer' | 'precedent_analyst';

export type ExpertRunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface ExpertStatus {
  id: ExpertId;
  status: ExpertRunStatus;
  progress?: number; // 0-100
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

export interface ExpertPipelineStatus {
  queryId: string;
  overallProgress: number; // 0-100
  phase: 'routing' | 'expert_analysis' | 'synthesis' | 'completed' | 'failed';
  experts: ExpertStatus[];
  synthesisStatus?: 'pending' | 'running' | 'completed' | 'failed';
}

export interface ExpertConfig {
  color: string;
  icon: string;
  displayName: string;
}

export const EXPERT_CONFIG: Record<ExpertId, ExpertConfig> = {
  literal_interpreter: {
    color: '#3b82f6',
    icon: 'BookOpen',
    displayName: 'Letterale',
  },
  systemic_teleological: {
    color: '#8b5cf6',
    icon: 'TrendingUp',
    displayName: 'Sistematico',
  },
  principles_balancer: {
    color: '#f59e0b',
    icon: 'Scale',
    displayName: 'Principi',
  },
  precedent_analyst: {
    color: '#10b981',
    icon: 'Gavel',
    displayName: 'Giurisprudenza',
  },
};

export const EXPERT_IDS: ExpertId[] = [
  'literal_interpreter',
  'systemic_teleological',
  'principles_balancer',
  'precedent_analyst',
];
