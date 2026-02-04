/**
 * Privacy service for GDPR data management
 * - Art. 20: Data portability (export)
 * - Art. 17: Right to erasure (deletion)
 */
import { get, post } from './api';
import type {
  PrivacyStatusResponse,
  DeleteAccountResponse,
  CancelDeletionResponse,
  DataExport,
} from '../types/api';

/**
 * Get current privacy/deletion status
 */
export const getPrivacyStatus = async (): Promise<PrivacyStatusResponse> => {
  return get<PrivacyStatusResponse>('/privacy/status');
};

/**
 * Export user data (GDPR Art. 20)
 */
export const exportData = async (): Promise<DataExport> => {
  return post<DataExport>('/privacy/export', {});
};

/**
 * Request account deletion (GDPR Art. 17)
 */
export const requestDeletion = async (
  password: string,
  reason?: string
): Promise<DeleteAccountResponse> => {
  return post<DeleteAccountResponse>('/privacy/delete-account', {
    password,
    reason,
  });
};

/**
 * Cancel pending account deletion
 */
export const cancelDeletion = async (): Promise<CancelDeletionResponse> => {
  return post<CancelDeletionResponse>('/privacy/cancel-deletion', {});
};
