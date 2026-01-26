/**
 * Invitation service for validating and managing invitations
 */
import { get, post, del } from './api';
import type {
  InvitationValidateResponse,
  InvitationCreateRequest,
  InvitationResponse,
} from '../types/api';

/**
 * Validate an invitation token (public endpoint)
 */
export const validateInvitation = async (token: string): Promise<InvitationValidateResponse> => {
  return get<InvitationValidateResponse>(`/invitations/${token}/validate`);
};

/**
 * Create a new invitation (authenticated)
 */
export const createInvitation = async (data?: InvitationCreateRequest): Promise<InvitationResponse> => {
  return post<InvitationResponse>('/invitations', data || {});
};

/**
 * List my sent invitations (authenticated)
 */
export const listMyInvitations = async (): Promise<InvitationResponse[]> => {
  return get<InvitationResponse[]>('/invitations');
};

/**
 * Revoke an invitation (authenticated)
 */
export const revokeInvitation = async (id: string): Promise<void> => {
  return del(`/invitations/${id}`);
};
