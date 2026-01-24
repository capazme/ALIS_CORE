/**
 * ProfilePage
 * ===========
 *
 * Pagina profilo utente completa con:
 * - Authority RLCF overview
 * - Breakdown della formula
 * - Statistiche contributi
 * - Heatmap domini legali
 * - Sezione educativa RLCF
 *
 * Responsive design: stack verticale su mobile, layout a griglia su desktop.
 */

import { motion } from 'framer-motion';
import { User, RefreshCw, AlertCircle, Calendar, Clock } from 'lucide-react';
import { cn } from '../../../../lib/utils';
import { useProfile } from '../../../../hooks/useProfile';
import { AuthorityOverview } from './AuthorityOverview';
import { AuthorityBreakdown } from './AuthorityBreakdown';
import { ContributionStats } from './ContributionStats';
import { DomainHeatmap } from './DomainHeatmap';
import { ActivityTimeline } from './ActivityTimeline';
import { RLCFExplainer } from './RLCFExplainer';

// =============================================================================
// LOADING SKELETON
// =============================================================================

function ProfileSkeleton() {
  return (
    <div className="animate-pulse space-y-6">
      {/* Header skeleton */}
      <div className="h-48 bg-slate-200 dark:bg-slate-800 rounded-xl" />

      {/* Grid skeleton */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="h-64 bg-slate-200 dark:bg-slate-800 rounded-xl" />
        <div className="h-64 bg-slate-200 dark:bg-slate-800 rounded-xl" />
      </div>

      {/* Bottom skeleton */}
      <div className="h-48 bg-slate-200 dark:bg-slate-800 rounded-xl" />
    </div>
  );
}

// =============================================================================
// ERROR STATE
// =============================================================================

interface ErrorStateProps {
  message: string;
  onRetry: () => void;
}

function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="w-16 h-16 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center mb-4">
        <AlertCircle size={32} className="text-red-500" />
      </div>
      <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">
        Errore nel caricamento
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4 max-w-sm">
        {message}
      </p>
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors"
      >
        <RefreshCw size={16} />
        Riprova
      </button>
    </div>
  );
}

// =============================================================================
// HEADER
// =============================================================================

interface ProfileHeaderProps {
  displayName?: string;
  userId: string;
  joinedAt: string;
  lastUpdated: string;
  onRefresh: () => void;
  isRefreshing: boolean;
}

function ProfileHeader({
  displayName,
  userId,
  joinedAt,
  lastUpdated,
  onRefresh,
  isRefreshing,
}: ProfileHeaderProps) {
  const formatDate = (isoString: string) => {
    return new Date(isoString).toLocaleDateString('it-IT', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    });
  };

  const formatRelativeTime = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 60) return `${diffMins} minuti fa`;
    if (diffHours < 24) return `${diffHours} ore fa`;
    return `${diffDays} giorni fa`;
  };

  return (
    <div className="flex items-start justify-between mb-6">
      <div className="flex items-center gap-4">
        {/* Avatar */}
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-indigo-500 flex items-center justify-center">
          <User size={32} className="text-white" />
        </div>

        {/* Info */}
        <div>
          <h1 className="text-xl font-bold text-slate-800 dark:text-slate-100">
            {displayName || 'Il tuo profilo'}
          </h1>
          <div className="flex items-center gap-4 mt-1 text-xs text-slate-500 dark:text-slate-400">
            <span className="flex items-center gap-1">
              <Calendar size={12} />
              Iscritto dal {formatDate(joinedAt)}
            </span>
            <span className="flex items-center gap-1">
              <Clock size={12} />
              Aggiornato {formatRelativeTime(lastUpdated)}
            </span>
          </div>
        </div>
      </div>

      {/* Refresh button */}
      <button
        onClick={onRefresh}
        disabled={isRefreshing}
        className={cn(
          'p-2 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors',
          isRefreshing && 'animate-spin'
        )}
        title="Aggiorna dati"
      >
        <RefreshCw size={18} />
      </button>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface ProfilePageProps {
  userId: string;
  className?: string;
}

export function ProfilePage({ userId, className }: ProfilePageProps) {
  const { profile, loading, error, refresh } = useProfile(userId);

  if (loading) {
    return (
      <div className={cn('p-6', className)}>
        <ProfileSkeleton />
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className={cn('p-6', className)}>
        <ErrorState
          message={error || 'Impossibile caricare il profilo'}
          onRetry={refresh}
        />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={cn('p-6 max-w-5xl mx-auto', className)}
    >
      {/* Header */}
      <ProfileHeader
        displayName={profile.display_name}
        userId={profile.user_id}
        joinedAt={profile.joined_at}
        lastUpdated={profile.last_updated}
        onRefresh={refresh}
        isRefreshing={loading}
      />

      {/* Authority Overview - Full width */}
      <AuthorityOverview
        authorityScore={profile.authority.score}
        tier={profile.authority.tier}
        breakdown={profile.authority.breakdown}
        nextTierThreshold={profile.authority.next_tier_threshold}
        progressToNext={profile.authority.progress_to_next}
        className="mb-6"
      />

      {/* Main grid: Breakdown + Stats */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <AuthorityBreakdown
          breakdown={profile.authority.breakdown}
          totalScore={profile.authority.score}
        />
        <ContributionStats
          totalContributions={profile.stats.total_contributions}
          approved={profile.stats.approved}
          rejected={profile.stats.rejected}
          pending={profile.stats.pending}
          voteWeight={profile.stats.vote_weight}
        />
      </div>

      {/* Domain Heatmap - Full width */}
      <DomainHeatmap domains={profile.domains} className="mb-6" />

      {/* Activity Timeline - Full width */}
      <ActivityTimeline activities={profile.recent_activity} className="mb-6" />

      {/* RLCF Explainer - Full width */}
      <RLCFExplainer />
    </motion.div>
  );
}

export default ProfilePage;
