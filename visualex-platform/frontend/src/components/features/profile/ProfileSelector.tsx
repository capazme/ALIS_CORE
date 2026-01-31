/**
 * ProfileSelector Component
 * =========================
 *
 * Displays the 4 ALIS profile types as selectable cards.
 * Users can choose their research style based on their expertise level.
 *
 * Profile types:
 * - ⚡ Consultazione Rapida: Quick answers, minimal interaction
 * - 📖 Ricerca Assistita: Guided exploration with suggestions
 * - 🔍 Analisi Esperta: Full access to Expert trace and feedback
 * - 🎓 Contributore Attivo: Granular feedback, training impact (requires authority ≥ 0.5)
 */
import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, Lock, AlertCircle, TrendingUp } from 'lucide-react';
import { cn } from '../../../lib/utils';
import * as profileService from '../../../services/profileService';
import type { ProfileType, ProfileDescription } from '../../../types/api';

export interface ProfileSelectorProps {
  initialProfileType?: ProfileType;
  authorityScore?: number;
  onProfileChange?: (profileType: ProfileType) => void;
  className?: string;
}

const PROFILE_ICONS: Record<ProfileType, string> = {
  quick_consultation: '⚡',
  assisted_research: '📖',
  expert_analysis: '🔍',
  active_contributor: '🎓',
};

const PROFILE_COLORS: Record<ProfileType, { bg: string; border: string; text: string }> = {
  quick_consultation: {
    bg: 'bg-amber-50 dark:bg-amber-900/20',
    border: 'border-amber-200 dark:border-amber-800',
    text: 'text-amber-700 dark:text-amber-300',
  },
  assisted_research: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-200 dark:border-blue-800',
    text: 'text-blue-700 dark:text-blue-300',
  },
  expert_analysis: {
    bg: 'bg-violet-50 dark:bg-violet-900/20',
    border: 'border-violet-200 dark:border-violet-800',
    text: 'text-violet-700 dark:text-violet-300',
  },
  active_contributor: {
    bg: 'bg-emerald-50 dark:bg-emerald-900/20',
    border: 'border-emerald-200 dark:border-emerald-800',
    text: 'text-emerald-700 dark:text-emerald-300',
  },
};

export function ProfileSelector({
  initialProfileType = 'assisted_research',
  authorityScore = 0,
  onProfileChange,
  className,
}: ProfileSelectorProps) {
  const [selectedProfile, setSelectedProfile] = useState<ProfileType>(initialProfileType);
  const [profiles, setProfiles] = useState<ProfileDescription[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Load profiles on mount
  useEffect(() => {
    const loadProfiles = async () => {
      try {
        const response = await profileService.getProfile();
        setProfiles(response.available_profiles);
        setSelectedProfile(response.profile_type);
        setLoading(false);
      } catch (err) {
        // If API fails, use default profiles
        const defaultProfiles: ProfileDescription[] = [
          {
            type: 'quick_consultation',
            emoji: '⚡',
            name: 'Consultazione Rapida',
            description: 'Risposte veloci, minima interazione',
            available: true,
          },
          {
            type: 'assisted_research',
            emoji: '📖',
            name: 'Ricerca Assistita',
            description: 'Esplorazione guidata con suggerimenti',
            available: true,
          },
          {
            type: 'expert_analysis',
            emoji: '🔍',
            name: 'Analisi Esperta',
            description: 'Accesso completo a Expert trace e feedback',
            available: true,
          },
          {
            type: 'active_contributor',
            emoji: '🎓',
            name: 'Contributore Attivo',
            description: 'Feedback granulare, impatto sul training',
            available: authorityScore >= 0.5,
            requiresAuthority: 0.5,
          },
        ];
        setProfiles(defaultProfiles);
        setLoading(false);
      }
    };

    loadProfiles();
  }, [authorityScore]);

  // Handle profile selection
  const handleSelectProfile = useCallback(
    async (profileType: ProfileType) => {
      if (profileType === selectedProfile) return;

      const profile = profiles.find((p) => p.type === profileType);
      if (!profile?.available) {
        setError(`Richiesta autorità minima: ${profile?.requiresAuthority ?? 0.5}`);
        return;
      }

      setSaving(true);
      setError(null);
      setSuccess(false);

      try {
        await profileService.updateProfile(profileType);
        setSelectedProfile(profileType);
        setSuccess(true);
        onProfileChange?.(profileType);

        // Clear success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Errore nel salvataggio del profilo';
        setError(message);
      } finally {
        setSaving(false);
      }
    },
    [selectedProfile, profiles, onProfileChange]
  );

  if (loading) {
    return (
      <div className={cn('grid grid-cols-1 md:grid-cols-2 gap-4', className)}>
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="h-32 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 animate-pulse"
          />
        ))}
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Authority Score Display */}
      <div className="flex items-center justify-between text-sm text-slate-600 dark:text-slate-400">
        <span className="flex items-center gap-1.5">
          <TrendingUp size={16} />
          Punteggio Autorità
        </span>
        <span className="font-medium text-slate-800 dark:text-slate-200">
          {authorityScore.toFixed(2)}
        </span>
      </div>

      {/* Profile Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {profiles.map((profile) => {
          const isSelected = selectedProfile === profile.type;
          const colors = PROFILE_COLORS[profile.type];

          return (
            <motion.button
              key={profile.type}
              onClick={() => handleSelectProfile(profile.type)}
              disabled={saving || !profile.available}
              className={cn(
                'relative p-4 rounded-xl border-2 text-left transition-all',
                'focus:outline-none focus:ring-2 focus:ring-offset-2',
                isSelected
                  ? cn(colors.bg, colors.border, 'focus:ring-blue-500')
                  : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600',
                !profile.available && 'opacity-60 cursor-not-allowed',
                saving && 'cursor-wait'
              )}
              whileHover={{ scale: profile.available ? 1.02 : 1 }}
              whileTap={{ scale: profile.available ? 0.98 : 1 }}
            >
              {/* Selected indicator */}
              {isSelected && (
                <div className="absolute top-3 right-3">
                  <CheckCircle className="w-5 h-5 text-emerald-500" />
                </div>
              )}

              {/* Locked indicator */}
              {!profile.available && (
                <div className="absolute top-3 right-3">
                  <Lock className="w-5 h-5 text-slate-400" />
                </div>
              )}

              {/* Profile content */}
              <div className="flex items-start gap-3">
                <span className="text-2xl" role="img" aria-label={profile.name}>
                  {PROFILE_ICONS[profile.type]}
                </span>
                <div>
                  <h4
                    className={cn(
                      'font-semibold',
                      isSelected
                        ? colors.text
                        : 'text-slate-800 dark:text-slate-100'
                    )}
                  >
                    {profile.name}
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    {profile.description}
                  </p>
                  {profile.requiresAuthority && (
                    <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                      Richiede autorità ≥ {profile.requiresAuthority}
                    </p>
                  )}
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Error message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30"
        >
          <p className="text-sm text-red-600 dark:text-red-400 flex items-center gap-2">
            <AlertCircle size={16} />
            {error}
          </p>
        </motion.div>
      )}

      {/* Success message */}
      {success && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800/30"
        >
          <p className="text-sm text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
            <CheckCircle size={16} />
            Profilo aggiornato con successo
          </p>
        </motion.div>
      )}
    </div>
  );
}

export default ProfileSelector;
