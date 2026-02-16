/**
 * Devil's Advocate Service
 * =========================
 *
 * Service per interazione con il sistema Devil's Advocate (RLCF Pillar 4).
 *
 * Endpoints backend:
 * - POST /devils-advocate/check
 * - POST /devils-advocate/feedback
 * - GET /devils-advocate/effectiveness
 */

import { get, post } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface DACheckResponse {
  triggered: boolean;
  critical_prompt: string | null;
  message: string;
}

export interface DAFeedbackRequest {
  trace_id: string;
  feedback_text: string;
  assessment: 'valid' | 'weak' | 'interesting';
}

export interface DAFeedbackResponse {
  received: boolean;
  engagement_score: number;
  critical_keywords_found: number;
}

export interface DAEffectivenessResponse {
  total_triggers: number;
  total_feedbacks: number;
  avg_engagement: number;
  avg_keywords: number;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

export async function checkDevilsAdvocate(
  traceId: string,
  disagreementScore: number,
): Promise<DACheckResponse> {
  return post(`${PREFIX}/devils-advocate/check?trace_id=${encodeURIComponent(traceId)}&disagreement_score=${disagreementScore}`);
}

export async function submitDAFeedback(
  data: DAFeedbackRequest,
): Promise<DAFeedbackResponse> {
  return post(`${PREFIX}/devils-advocate/feedback`, data);
}

export async function getDAEffectiveness(): Promise<DAEffectivenessResponse> {
  return get(`${PREFIX}/devils-advocate/effectiveness`);
}
