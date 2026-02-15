/**
 * Statistics Service
 * ==================
 *
 * Service per interagire con l'API Statistics di MERL-T.
 * Fornisce funzioni per hypothesis testing, distribuzioni, correlazioni ed export.
 *
 * API Endpoints:
 * - GET /statistics/overview - Tutte le statistiche
 * - GET /statistics/hypothesis-tests - Test ipotesi H1-H4
 * - GET /statistics/distributions - Distribuzioni
 * - GET /statistics/correlations - Matrice correlazione
 * - POST /statistics/export - Export CSV/JSON/LaTeX
 */

import { get, post } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

export type TestType =
  | 't-test'
  | 'paired-t-test'
  | 'anova'
  | 'mann-whitney'
  | 'wilcoxon'
  | 'correlation'
  | 'chi-square';

export type EffectSizeType =
  | 'cohens_d'
  | 'hedges_g'
  | 'eta_squared'
  | 'partial_eta_squared'
  | 'r'
  | 'r_squared'
  | 'cramers_v';

export type SignificanceLevel = 'ns' | '*' | '**' | '***';

export type EffectInterpretation = 'negligible' | 'small' | 'medium' | 'large';

export type ExportFormat = 'csv' | 'json' | 'latex';

export interface DescriptiveStats {
  mean: number;
  std: number;
  n: number;
  median?: number;
  min_val?: number;
  max_val?: number;
  ci_lower?: number;
  ci_upper?: number;
}

export interface EffectSize {
  value: number;
  type: EffectSizeType;
  interpretation: EffectInterpretation;
  ci_lower?: number;
  ci_upper?: number;
}

export interface HypothesisTestResult {
  hypothesis_id: string;
  description: string;
  pre_stats?: DescriptiveStats;
  post_stats?: DescriptiveStats;
  delta?: number;
  test_type: TestType;
  statistic: number;
  df?: number;
  df2?: number;
  p_value: number;
  effect_size: EffectSize;
  ci_level: number;
  ci_lower?: number;
  ci_upper?: number;
  supported: boolean;
  significance: SignificanceLevel;
  notes?: string;
  computed_at: string;
}

export interface HypothesisTestSummary {
  tests: HypothesisTestResult[];
  supported_count: number;
  total_count: number;
  alpha: number;
}

export interface DistributionData {
  name: string;
  values?: number[];
  bins: number[];
  counts: number[];
  mean: number;
  std: number;
  median: number;
  skewness: number;
  kurtosis: number;
  n: number;
}

export interface NormalityTest {
  test_name: string;
  statistic: number;
  p_value: number;
  is_normal: boolean;
}

export interface DistributionAnalysis {
  distribution: DistributionData;
  normality_test?: NormalityTest;
  percentiles: Record<string, number>;
}

export interface CorrelationPair {
  var1: string;
  var2: string;
  r: number;
  p_value: number;
  n: number;
  significance: SignificanceLevel;
}

export interface CorrelationMatrix {
  variables: string[];
  matrix: number[][];
  p_values: number[][];
  significant_pairs: CorrelationPair[];
}

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy?: number;
  learning_rate: number;
  timestamp: string;
}

export interface PolicyWeights {
  literal: number;
  systemic: number;
  principles: number;
  precedent: number;
  timestamp: string;
}

export interface AuthorityDistribution {
  novizio: number;
  contributore: number;
  esperto: number;
  autorita: number;
  mean_authority: number;
  std_authority: number;
}

export interface StatisticsOverview {
  hypothesis_tests: HypothesisTestSummary;
  distributions: Record<string, DistributionAnalysis>;
  correlations: CorrelationMatrix;
  training_history: TrainingMetrics[];
  policy_weights: PolicyWeights;
  authority_distribution: AuthorityDistribution;
  last_computed: string;
}

export interface ExportRequest {
  format: ExportFormat;
  include_hypothesis_tests?: boolean;
  include_descriptive_stats?: boolean;
  include_raw_data?: boolean;
  include_confidence_intervals?: boolean;
  include_effect_sizes?: boolean;
  date_range_start?: string;
  date_range_end?: string;
}

export interface ExportResponse {
  success: boolean;
  format: ExportFormat;
  download_url?: string;
  filename: string;
  file_size_kb?: number;
  records_count: number;
  message: string;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Recupera tutte le statistiche.
 */
export async function getStatisticsOverview(): Promise<StatisticsOverview> {
  return get(`${PREFIX}/statistics/overview`);
}

/**
 * Recupera risultati test di ipotesi H1-H4.
 */
export async function getHypothesisTests(): Promise<HypothesisTestSummary> {
  return get(`${PREFIX}/statistics/hypothesis-tests`);
}

/**
 * Recupera analisi delle distribuzioni.
 */
export async function getDistributions(): Promise<Record<string, DistributionAnalysis>> {
  return get(`${PREFIX}/statistics/distributions`);
}

/**
 * Recupera matrice di correlazione.
 */
export async function getCorrelations(): Promise<CorrelationMatrix> {
  return get(`${PREFIX}/statistics/correlations`);
}

/**
 * Esporta statistiche in vari formati.
 */
export async function exportStatistics(request: ExportRequest): Promise<ExportResponse> {
  return post(`${PREFIX}/statistics/export`, request);
}

/**
 * Download file esportato.
 */
export async function downloadExport(filename: string): Promise<Blob> {
  const response = await fetch(`/api${PREFIX}/statistics/download/${filename}`);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.statusText}`);
  }
  return response.blob();
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Formatta p-value con asterischi.
 */
export function formatPValue(pValue: number | null | undefined, significance: SignificanceLevel): string {
  if (pValue == null) return 'p = -';
  const formatted = pValue < 0.001 ? '< 0.001' : pValue.toFixed(4);
  return `p = ${formatted} ${significance}`;
}

/**
 * Formatta effect size con interpretazione.
 */
export function formatEffectSize(effectSize: EffectSize | null | undefined): string {
  if (!effectSize || effectSize.value == null) return '-';

  const typeLabels: Record<EffectSizeType, string> = {
    cohens_d: "Cohen's d",
    hedges_g: "Hedges' g",
    eta_squared: 'η²',
    partial_eta_squared: 'partial η²',
    r: 'r',
    r_squared: 'R²',
    cramers_v: "Cramer's V",
  };

  return `${typeLabels[effectSize.type] || effectSize.type} = ${effectSize.value.toFixed(2)} (${effectSize.interpretation || 'N/A'})`;
}

/**
 * Formatta confidence interval.
 */
export function formatCI(lower: number | null | undefined, upper: number | null | undefined, level: number = 0.95): string {
  if (lower == null || upper == null) {
    return '-';
  }
  const percent = Math.round(level * 100);
  return `${percent}% CI [${lower.toFixed(2)}, ${upper.toFixed(2)}]`;
}

/**
 * Ritorna colore per significance level.
 */
export function getSignificanceColor(significance: SignificanceLevel): string {
  switch (significance) {
    case '***':
      return 'text-green-600 font-bold';
    case '**':
      return 'text-green-500 font-semibold';
    case '*':
      return 'text-green-400';
    case 'ns':
      return 'text-slate-400';
  }
}

/**
 * Ritorna colore per effect size interpretation.
 */
export function getEffectSizeColor(interpretation: EffectInterpretation): string {
  switch (interpretation) {
    case 'large':
      return 'text-purple-600';
    case 'medium':
      return 'text-blue-500';
    case 'small':
      return 'text-slate-500';
    case 'negligible':
      return 'text-slate-400';
  }
}

/**
 * Genera label per test type.
 */
export function getTestTypeLabel(testType: TestType): string {
  const labels: Record<TestType, string> = {
    't-test': 't-test indipendente',
    'paired-t-test': 't-test appaiato',
    'anova': 'ANOVA',
    'mann-whitney': 'Mann-Whitney U',
    'wilcoxon': 'Wilcoxon',
    'correlation': 'Correlazione Pearson',
    'chi-square': 'Chi-quadrato',
  };
  return labels[testType];
}
