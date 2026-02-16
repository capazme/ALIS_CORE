/**
 * Quarantine Service
 * ==================
 *
 * Service per gestione quarantine/moderation di feedback via API MERL-T.
 *
 * Endpoints:
 * - POST /feedback/{id}/flag — flag feedback
 * - POST /feedback/{id}/quarantine — quarantina
 * - POST /feedback/{id}/approve — approva
 * - GET /feedback/flagged — lista flagged
 * - GET /feedback/quarantined — lista quarantined
 * - POST /feedback/auto-detect — auto-detect outlier
 */

import { get, post } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface FeedbackItem {
  id: number;
  trace_id: string;
  user_id: string;
  inline_rating: number | null;
  status: 'approved' | 'flagged' | 'quarantined' | 'deleted';
  quarantine_reason: string | null;
  flagged_at: string | null;
  flagged_by: string | null;
  reviewed_at: string | null;
  reviewed_by: string | null;
  user_authority: number | null;
  created_at: string | null;
}

export interface FeedbackListResponse {
  items: FeedbackItem[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface AutoDetectResponse {
  flagged_count: number;
  flagged_by: string;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

export async function flagFeedback(
  feedbackId: number,
  reason: string,
  flaggedBy: string = 'admin'
): Promise<FeedbackItem> {
  return post(`${PREFIX}/feedback/${feedbackId}/flag`, { reason, flagged_by: flaggedBy });
}

export async function quarantineFeedback(
  feedbackId: number,
  reason: string,
  reviewedBy: string = 'admin'
): Promise<FeedbackItem> {
  return post(`${PREFIX}/feedback/${feedbackId}/quarantine`, { reason, reviewed_by: reviewedBy });
}

export async function approveFeedback(
  feedbackId: number,
  reviewedBy: string = 'admin'
): Promise<FeedbackItem> {
  return post(`${PREFIX}/feedback/${feedbackId}/approve`, { reviewed_by: reviewedBy });
}

export async function getFlaggedFeedback(
  limit: number = 50,
  offset: number = 0
): Promise<FeedbackListResponse> {
  return get(`${PREFIX}/feedback/flagged?limit=${limit}&offset=${offset}`);
}

export async function getQuarantinedFeedback(
  limit: number = 50,
  offset: number = 0
): Promise<FeedbackListResponse> {
  return get(`${PREFIX}/feedback/quarantined?limit=${limit}&offset=${offset}`);
}

export async function autoDetectOutliers(): Promise<AutoDetectResponse> {
  return post(`${PREFIX}/feedback/auto-detect`);
}
