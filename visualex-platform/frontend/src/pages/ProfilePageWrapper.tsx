/**
 * ProfilePageWrapper
 * ==================
 *
 * Wrapper che passa l'userId autenticato alla ProfilePage tramite plugin slots.
 */

import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { PluginSlot } from '../lib/plugins/PluginSlot';

export function ProfilePageWrapper() {
  const navigate = useNavigate();
  const { user } = useAuth();

  if (!user?.id) {
    return (
      <div className="flex items-center justify-center min-h-full">
        <p className="text-slate-500">Caricamento...</p>
      </div>
    );
  }

  return (
    <div className="min-h-full bg-slate-50 dark:bg-slate-950">
      {/* Back button */}
      <div className="max-w-5xl mx-auto px-6 pt-6">
        <button
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 text-sm text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
        >
          <ArrowLeft size={16} />
          Indietro
        </button>
      </div>

      {/* Profile content - rendered by plugins */}
      <PluginSlot
        name="profile-tabs"
        props={{ userId: user.id }}
        fallback={
          <div className="max-w-3xl mx-auto px-6 py-10">
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                Profilo ricerca non disponibile
              </h2>
              <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
                Questo profilo riguarda le funzionalita' di ricerca collaborativa (MERL-T/RLCF).
                Per attivarle devi aderire come volontario.
              </p>
              <button
                onClick={() => navigate('/settings')}
                className="mt-4 inline-flex items-center rounded-lg bg-slate-900 text-white px-4 py-2 text-sm font-medium hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
              >
                Vai alle impostazioni
              </button>
            </div>
          </div>
        }
      />
    </div>
  );
}

export default ProfilePageWrapper;
