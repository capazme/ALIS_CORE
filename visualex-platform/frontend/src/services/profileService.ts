/**
 * Profile service for user profile and preferences management
 */
import { get, put } from './api';
import type {
  ProfileResponse,
  ProfileType,
  UpdateProfileRequest,
  UpdatePreferencesRequest,
  UserPreferences,
} from '../types/api';

/**
 * Get current user's profile and preferences
 */
export const getProfile = async (): Promise<ProfileResponse> => {
  return get<ProfileResponse>('/profile');
};

/**
 * Update user's profile type
 */
export const updateProfile = async (profileType: ProfileType): Promise<{ message: string; profile_type: ProfileType }> => {
  const data: UpdateProfileRequest = { profile_type: profileType };
  return put<{ message: string; profile_type: ProfileType }>('/profile', data);
};

/**
 * Update user's preferences
 */
export const updatePreferences = async (preferences: UpdatePreferencesRequest): Promise<{ message: string; preferences: UserPreferences }> => {
  return put<{ message: string; preferences: UserPreferences }>('/profile/preferences', preferences);
};
