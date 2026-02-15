/**
 * useProfile
 * ==========
 *
 * Hook per recuperare e gestire il profilo utente RLCF.
 *
 * Features:
 * - Fetch dati profilo con authority e statistiche
 * - Calcolo tier e progress automatico
 * - Refresh on demand
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { getUserProfile } from '../services/merltService';
import type { ProfileResponse, AuthorityTier, LegalDomain, DomainStats } from '../types/merlt';

// =============================================================================
// TIER UTILITIES
// =============================================================================

/**
 * Thresholds per i tier RLCF (allineati con backend).
 */
const TIER_THRESHOLDS: Record<AuthorityTier, { min: number; max: number }> = {
  novizio: { min: 0, max: 0.4 },
  contributore: { min: 0.4, max: 0.6 },
  esperto: { min: 0.6, max: 0.8 },
  autorita: { min: 0.8, max: 1.0 },
};

/**
 * Determina il tier dall'authority score.
 */
export function getTierFromScore(score: number): AuthorityTier {
  if (score >= 0.8) return 'autorita';
  if (score >= 0.6) return 'esperto';
  if (score >= 0.4) return 'contributore';
  return 'novizio';
}

/**
 * Calcola il progresso verso il prossimo tier.
 */
export function getProgressToNextTier(score: number): {
  progress: number;
  nextThreshold: number;
  nextTier: AuthorityTier | null;
} {
  const currentTier = getTierFromScore(score);
  const { max } = TIER_THRESHOLDS[currentTier];

  if (currentTier === 'autorita') {
    return { progress: 100, nextThreshold: 1.0, nextTier: null };
  }

  const { min } = TIER_THRESHOLDS[currentTier];
  const tierRange = max - min;
  const positionInTier = score - min;
  const progress = Math.min(100, Math.max(0, (positionInTier / tierRange) * 100));

  const tierOrder: AuthorityTier[] = ['novizio', 'contributore', 'esperto', 'autorita'];
  const currentIndex = tierOrder.indexOf(currentTier);
  const nextTier = tierOrder[currentIndex + 1] || null;

  return { progress, nextThreshold: max, nextTier };
}

// =============================================================================
// HOOK STATE
// =============================================================================

interface UseProfileState {
  profile: ProfileResponse | null;
  loading: boolean;
  error: string | null;
}

interface UseProfileOptions {
  /** Auto-fetch al mount */
  autoFetch?: boolean;
}

// =============================================================================
// HOOK
// =============================================================================

/**
 * Hook per recuperare il profilo utente RLCF.
 *
 * @param userId - ID utente
 * @param options - Opzioni di configurazione
 *
 * @example
 * const { profile, loading, error, refresh } = useProfile(userId);
 *
 * if (loading) return <Spinner />;
 * if (error) return <Error message={error} />;
 *
 * return <ProfileCard profile={profile} />;
 */
export function useProfile(
  userId: string | undefined,
  options: UseProfileOptions = {}
) {
  const { autoFetch = true } = options;

  const [state, setState] = useState({
    profile: null,
    loading: autoFetch,
    error: null,
  } as UseProfileState);

  /**
   * Fetch profilo dal backend.
   */
  const fetchProfile = useCallback(async () => {
    if (!userId) {
      setState({ profile: null, loading: false, error: 'User ID required' });
      return;
    }

    setState((prev: UseProfileState) => ({ ...prev, loading: true, error: null }));

    try {
      const profile = await getUserProfile(userId);
      setState({ profile, loading: false, error: null });
    } catch (err: unknown) {
      console.error('Failed to fetch profile:', err);

      const errorMessage = err instanceof Error ? err.message : 'Failed to load profile';
      setState({
        profile: null,
        loading: false,
        error: errorMessage,
      });
    }
  }, [userId]);

  /**
   * Auto-fetch on mount o quando cambia userId.
   */
  useEffect(() => {
    if (autoFetch && userId) {
      fetchProfile();
    }
  }, [fetchProfile, autoFetch, userId]);

  /**
   * Computed: progress info.
   */
  const progressInfo = useMemo(() => {
    if (!state.profile) return null;
    return getProgressToNextTier(state.profile.authority.score);
  }, [state.profile]);

  /**
   * Computed: domini ordinati per authority.
   */
  const sortedDomains = useMemo(() => {
    if (!state.profile) return [];

    return (Object.entries(state.profile.domains) as [string, DomainStats][])
      .map(([domain, stats]) => ({
        domain: domain as LegalDomain,
        ...stats,
      }))
      .sort((a, b) => b.authority - a.authority);
  }, [state.profile]);

  /**
   * Computed: success rate globale.
   */
  const globalSuccessRate = useMemo(() => {
    if (!state.profile || state.profile.stats.total_contributions === 0) return 0;
    const { approved, total_contributions } = state.profile.stats;
    return Math.round((approved / total_contributions) * 100);
  }, [state.profile]);

  return {
    // State
    profile: state.profile,
    loading: state.loading,
    error: state.error,

    // Actions
    refresh: fetchProfile,

    // Computed
    progressInfo,
    sortedDomains,
    globalSuccessRate,

    // Utilities
    getTierFromScore,
    getProgressToNextTier,
  };
}

export default useProfile;
