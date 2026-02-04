/**
 * Authority service for RLCF influence score management
 */
import { get } from './api';
import type { AuthorityResponse } from '../types/api';

/**
 * Get current user's authority score breakdown
 */
export const getAuthority = async (): Promise<AuthorityResponse> => {
  return get<AuthorityResponse>('/authority');
};
