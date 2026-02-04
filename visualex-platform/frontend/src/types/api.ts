/**
 * Type definitions for API requests and responses
 */

// ============================================================================
// Authentication Types
// ============================================================================

export interface UserRegisterRequest {
  invitation_token: string;
  email: string;
  username: string;
  password: string;
  name?: string;
  role?: 'member' | 'researcher';
}

export interface UserLoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface RegisterResponse {
  message: string;
  verification_pending: boolean;
}

// ============================================================================
// Invitation Types
// ============================================================================

export interface InvitationValidateResponse {
  valid: boolean;
  email?: string;
  expires_at: string;
  inviter?: {
    username: string;
  };
}

export interface InvitationCreateRequest {
  email?: string;
}

export interface InvitationResponse {
  id: string;
  token: string;
  email?: string;
  expires_at: string;
  used_at?: string;
  created_at: string;
}

export interface VerifyEmailResponse {
  message: string;
  verified: boolean;
}

export interface ResendVerificationRequest {
  email: string;
}

export interface ResendVerificationResponse {
  message: string;
}

// Profile types for ALIS user experience levels
export type ProfileType =
  | 'quick_consultation'   // ⚡ Consultazione Rapida
  | 'assisted_research'    // 📖 Ricerca Assistita
  | 'expert_analysis'      // 🔍 Analisi Esperta
  | 'active_contributor';  // 🎓 Contributore Attivo (requires authority ≥ 0.5)

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: 'it' | 'en';
  notifications_enabled: boolean;
}

export interface ProfileDescription {
  type: ProfileType;
  emoji: string;
  name: string;
  description: string;
  available: boolean;
  requiresAuthority?: number;
}

export interface ProfileResponse {
  profile_type: ProfileType;
  authority_score: number;
  preferences: UserPreferences;
  available_profiles: ProfileDescription[];
}

export interface UpdateProfileRequest {
  profile_type: ProfileType;
}

export interface UpdatePreferencesRequest {
  theme?: 'light' | 'dark' | 'system';
  language?: 'it' | 'en';
  notifications_enabled?: boolean;
}

// ============================================================================
// Consent Types (GDPR Art. 6, Art. 7)
// ============================================================================

export type ConsentLevel = 'basic' | 'learning' | 'research';

export interface ConsentLevelDescription {
  level: ConsentLevel;
  emoji: string;
  name: string;
  description: string;
  dataCollected: string[];
}

export interface ConsentResponse {
  consent_level: ConsentLevel;
  granted_at: string;
  available_levels: ConsentLevelDescription[];
}

export interface UpdateConsentRequest {
  consent_level: ConsentLevel;
}

export interface UpdateConsentResponse {
  message: string;
  warning?: string;
  consent_level: ConsentLevel;
  granted_at: string;
  is_downgrade: boolean;
}

export interface ConsentHistoryEntry {
  id: string;
  previous_level: ConsentLevel | null;
  new_level: ConsentLevel;
  changed_at: string;
}

export interface ConsentHistoryResponse {
  history: ConsentHistoryEntry[];
}

// ============================================================================
// Authority Types (RLCF Influence)
// ============================================================================

export interface AuthorityComponentDescription {
  score: number;
  weighted: number;
  name: string;
  description: string;
  weight: number;
  icon: string;
}

export interface AuthorityComponents {
  baseline: AuthorityComponentDescription;
  track_record: AuthorityComponentDescription;
  recent_performance: AuthorityComponentDescription;
}

export interface AuthorityResponse {
  authority_score: number;
  feedback_count: number;
  updated_at: string;
  components: AuthorityComponents;
  message?: string; // Message for new users
}

// ============================================================================
// Privacy Types (GDPR Art. 17, Art. 20)
// ============================================================================

export interface PrivacyStatusResponse {
  deletion_pending: boolean;
  deletion_requested_at: string | null;
  deletion_reason: string | null;
  days_remaining: number | null;
  account_active: boolean;
  consent_level: ConsentLevel;
}

export interface DeleteAccountRequest {
  password: string;
  reason?: string;
}

export interface DeleteAccountResponse {
  message: string;
  deletion_requested_at: string;
  grace_period_days: number;
  warning: string;
}

export interface CancelDeletionResponse {
  message: string;
  account_status: string;
}

export interface DataExport {
  exported_at: string;
  gdpr_reference: string;
  user: {
    id: string;
    email: string;
    username: string;
    profile_type: string;
    is_verified: boolean;
    created_at: string;
    last_login_at: string | null;
    login_count: number;
  };
  preferences: {
    theme: string;
    language: string;
    notifications_enabled: boolean;
  };
  consent: {
    current_level: string;
    granted_at: string | null;
    history: Array<{
      previous_level: string | null;
      new_level: string;
      changed_at: string;
    }>;
  };
  authority: {
    score: number;
    baseline: number;
    track_record: number;
    recent_performance: number;
    feedback_count: number;
    updated_at: string | null;
  };
}

export interface UserResponse {
  id: string;
  email: string;
  username: string;
  is_active: boolean;
  is_verified: boolean;
  is_admin: boolean;
  is_merlt_enabled: boolean;
  profile_type: ProfileType;
  authority_score: number;
  login_count?: number;
  last_login_at?: string | null;
  created_at: string;
  updated_at?: string;
}

// Admin user management types
export interface AdminUserCreate {
  email: string;
  username: string;
  password: string;
  isAdmin?: boolean;
  isActive?: boolean;
  isMerltEnabled?: boolean;
}

export interface AdminUserUpdate {
  email?: string;
  username?: string;
  isAdmin?: boolean;
  isActive?: boolean;
  isVerified?: boolean;
  isMerltEnabled?: boolean;
}

export interface AdminUserResponse extends UserResponse {
  login_count?: number;
  last_login_at?: string | null;
  stats?: {
    bookmarks: number;
    dossiers: number;
    annotations: number;
    highlights: number;
  };
}

export interface AdminResetPassword {
  newPassword: string;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

// ============================================================================
// Folder Types
// ============================================================================

export interface FolderCreate {
  name: string;
  description?: string;
  color?: string;
  icon?: string;
  parent_id?: string;
  position?: number;
}

export interface FolderUpdate {
  name?: string;
  description?: string;
  color?: string;
  icon?: string;
  parent_id?: string;
  position?: number;
}

export interface FolderMove {
  parent_id?: string;
  position?: number;
}

export interface FolderResponse {
  id: string;
  name: string;
  description?: string;
  color?: string;
  icon?: string;
  parent_id?: string;
  position: number;
  created_at: string;
  updated_at?: string;
  children?: FolderResponse[];
}

export interface FolderTree {
  id: string;
  name: string;
  description?: string;
  color?: string;
  icon?: string;
  parent_id?: string;
  position: number;
  children: FolderTree[];
  bookmark_count?: number;
}

export interface FolderBulkDelete {
  folder_ids: string[];
}

export interface FolderBulkMove {
  folder_ids: string[];
  target_parent_id?: string;
}

// ============================================================================
// Bookmark Types (for future implementation)
// ============================================================================

export interface BookmarkCreate {
  norma_key: string;
  norma_data: Record<string, unknown>; // JSON object with norma data
  folder_id?: string;
  tags?: string[];
  notes?: string;
}

export interface BookmarkUpdate {
  title?: string;
  notes?: string;
  tags?: string[];
  folder_id?: string;
}

export interface BookmarkResponse {
  id: string;
  normaKey: string;
  normaData: Record<string, unknown>;
  title?: string;
  folderId?: string;
  tags: string[];
  notes?: string;
  userId: string;
  createdAt: string;
  updatedAt?: string;
}

// Aliases for camelCase
export type Bookmark = BookmarkResponse;

// ============================================================================
// Annotation Types (for future implementation)
// ============================================================================

export type AnnotationType = 'note' | 'question' | 'important' | 'follow_up' | 'summary';

export interface AnnotationCreate {
  norma_key: string;
  content: string;
  annotation_type?: AnnotationType;
  bookmark_id?: string;
  text_context?: string;
  position?: string;
}

export interface AnnotationUpdate {
  content?: string;
  annotation_type?: AnnotationType;
  text_context?: string;
  position?: number;
}

export interface AnnotationResponse {
  id: string;
  normaKey: string;
  content: string;
  annotationType: AnnotationType;
  bookmarkId?: string;
  textContext?: string;
  position?: number;
  userId: string;
  createdAt: string;
  updatedAt?: string;
}

// Aliases for camelCase
export type Annotation = AnnotationResponse;

// ============================================================================
// Highlight Types (for future implementation)
// ============================================================================

export interface HighlightCreate {
  norma_key: string;
  text: string;
  color: string;
  start_offset?: number;
  end_offset?: number;
  container_id?: string;
  note?: string;
  bookmark_id?: string;
}

export interface HighlightUpdate {
  color?: 'yellow' | 'green' | 'blue' | 'red' | 'purple';
  note?: string;
}

export interface HighlightResponse {
  id: string;
  normaKey: string;
  text: string;
  color: 'yellow' | 'green' | 'blue' | 'red' | 'purple';
  startOffset: number;
  endOffset: number;
  note?: string;
  bookmarkId?: string;
  userId: string;
  createdAt: string;
}

// Aliases for camelCase
export type Highlight = HighlightResponse;

// ============================================================================
// API Error Types
// ============================================================================

export interface APIError {
  status?: number;
  message: string;
  data?: Record<string, unknown>;
}
