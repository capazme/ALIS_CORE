/**
 * API Key Service
 * ================
 *
 * Service per gestione API keys via MERL-T backend (FR45).
 *
 * Endpoints:
 * - POST /api-keys/bootstrap — crea primo admin key
 * - POST /api-keys/ — crea key (admin)
 * - GET /api-keys/ — lista keys (admin)
 * - PATCH /api-keys/{key_id} — modifica key (admin)
 * - DELETE /api-keys/{key_id} — revoca key (admin)
 */

import { get, post, patch, del } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface ApiKeyInfo {
  key_id: string;
  role: string;
  rate_limit_tier: string;
  is_active: boolean;
  user_id: string;
  description: string | null;
  created_at: string | null;
  last_used_at: string | null;
  expires_at: string | null;
}

export interface ApiKeyListResponse {
  keys: ApiKeyInfo[];
  count: number;
}

export interface CreateApiKeyRequest {
  user_id: string;
  role?: string;
  rate_limit_tier?: string;
  description?: string;
}

export interface CreateApiKeyResponse {
  key_id: string;
  raw_key: string;
  role: string;
  rate_limit_tier: string;
  user_id: string;
  description: string | null;
  created_at: string | null;
}

export interface UpdateApiKeyRequest {
  is_active?: boolean;
  role?: string;
  rate_limit_tier?: string;
  description?: string;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

export async function listApiKeys(): Promise<ApiKeyListResponse> {
  return get(`${PREFIX}/api-keys/`);
}

export async function createApiKey(
  data: CreateApiKeyRequest,
): Promise<CreateApiKeyResponse> {
  return post(`${PREFIX}/api-keys/`, data);
}

export async function updateApiKey(
  keyId: string,
  data: UpdateApiKeyRequest,
): Promise<ApiKeyInfo> {
  return patch(`${PREFIX}/api-keys/${keyId}`, data);
}

export async function deleteApiKey(keyId: string): Promise<ApiKeyInfo> {
  return del(`${PREFIX}/api-keys/${keyId}`);
}

export async function bootstrapApiKey(): Promise<CreateApiKeyResponse> {
  return post(`${PREFIX}/api-keys/bootstrap`);
}
