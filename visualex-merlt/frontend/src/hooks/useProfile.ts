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
import type { ProfileResponse, LegalDomain, DomainStats } from '../types/merlt';

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
   * Computed: progress info — sourced directly from backend authority data.
   */
  const progressInfo = useMemo(() => {
    if (!state.profile) return null;
    const { tier, progress_to_next, next_tier_threshold } = state.profile.authority;
    return {
      progress: progress_to_next,
      nextThreshold: next_tier_threshold,
      nextTier: tier === 'autorita' ? null : tier,
    };
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

  };
}

export default useProfile;
