/**
 * ProfilePageWrapper
 * ==================
 *
 * Wrapper che passa l'userId autenticato alla ProfilePage.
 */

import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { ProfilePage } from '../components/features/merlt/profile';

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

      {/* Profile content */}
      <ProfilePage userId={user.id} />
    </div>
  );
}

export default ProfilePageWrapper;
