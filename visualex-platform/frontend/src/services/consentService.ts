/**
 * Consent service for GDPR-compliant consent management
 */
import { get, put } from './api';
import type {
  ConsentResponse,
  ConsentLevel,
  UpdateConsentResponse,
  ConsentHistoryResponse,
} from '../types/api';

/**
 * Get current user's consent level and available options
 */
export const getConsent = async (): Promise<ConsentResponse> => {
  return get<ConsentResponse>('/consent');
};

/**
 * Update user's consent level
 */
export const updateConsent = async (consentLevel: ConsentLevel): Promise<UpdateConsentResponse> => {
  return put<UpdateConsentResponse>('/consent', { consent_level: consentLevel });
};

/**
 * Get user's consent change history
 */
export const getConsentHistory = async (): Promise<ConsentHistoryResponse> => {
  return get<ConsentHistoryResponse>('/consent/history');
};
