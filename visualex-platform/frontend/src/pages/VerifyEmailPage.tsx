/**
 * Email Verification page - Handles email verification from link
 */
import { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { verifyEmail } from '../services/authService';
import { CheckCircle, XCircle, Loader2, ArrowLeft, Mail } from 'lucide-react';

type VerificationState = 'loading' | 'success' | 'error' | 'expired';

export function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');

  const [state, setState] = useState<VerificationState>('loading');
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    const verify = async () => {
      if (!token) {
        setState('error');
        setErrorMessage('Token di verifica mancante.');
        return;
      }

      try {
        await verifyEmail(token);
        setState('success');
      } catch (error: unknown) {
        const err = error as { message?: string };
        if (err.message?.includes('expired')) {
          setState('expired');
          setErrorMessage('Il link di verifica è scaduto. Richiedi un nuovo link.');
        } else {
          setState('error');
          setErrorMessage(err.message || 'Errore durante la verifica. Riprova.');
        }
      }
    };

    verify();
  }, [token]);

  // Loading state
  if (state === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#F3F4F6] dark:bg-[#0F172A] relative overflow-hidden px-4">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-[420px] w-full relative z-10 p-6">
          <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl rounded-2xl shadow-2xl ring-1 ring-slate-900/5 dark:ring-white/10 p-8 text-center">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin mx-auto mb-4" />
            <p className="text-slate-600 dark:text-slate-400">Verifica in corso...</p>
          </div>
        </div>
      </div>
    );
  }

  // Success state
  if (state === 'success') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#F3F4F6] dark:bg-[#0F172A] relative overflow-hidden px-4">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-green-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-[420px] w-full relative z-10 p-6">
          <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl rounded-2xl shadow-2xl ring-1 ring-slate-900/5 dark:ring-white/10 p-8 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-tr from-green-500 to-emerald-500 text-white rounded-2xl shadow-lg shadow-green-500/20 mb-6">
              <CheckCircle size={32} strokeWidth={2} />
            </div>

            <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
              Email Verificata!
            </h1>

            <p className="text-slate-600 dark:text-slate-400 mb-6">
              Il tuo account è stato attivato con successo.
              <br />
              Ora puoi effettuare il login.
            </p>

            <Link
              to="/login"
              className="inline-flex items-center justify-center gap-2 w-full py-3 px-4 rounded-xl font-semibold shadow-lg shadow-blue-500/25 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white transform active:scale-[0.98] transition-all duration-200"
            >
              Vai al Login
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // Expired state
  if (state === 'expired') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#F3F4F6] dark:bg-[#0F172A] relative overflow-hidden px-4">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-amber-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-orange-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-[420px] w-full relative z-10 p-6">
          <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl rounded-2xl shadow-2xl ring-1 ring-slate-900/5 dark:ring-white/10 p-8 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-tr from-amber-500 to-orange-500 text-white rounded-2xl shadow-lg shadow-amber-500/20 mb-6">
              <Mail size={32} strokeWidth={2} />
            </div>

            <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
              Link Scaduto
            </h1>

            <p className="text-slate-600 dark:text-slate-400 mb-6">
              {errorMessage}
            </p>

            <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl p-4 mb-6">
              <p className="text-sm text-amber-700 dark:text-amber-400">
                Puoi richiedere un nuovo link di verifica dalla pagina di login.
              </p>
            </div>

            <Link
              to="/login"
              className="inline-flex items-center justify-center gap-2 w-full py-3 px-4 rounded-xl font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
            >
              <ArrowLeft size={18} />
              Torna al Login
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F3F4F6] dark:bg-[#0F172A] relative overflow-hidden px-4">
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-red-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-orange-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-[420px] w-full relative z-10 p-6">
        <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl rounded-2xl shadow-2xl ring-1 ring-slate-900/5 dark:ring-white/10 p-8 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-tr from-red-500 to-orange-500 text-white rounded-2xl shadow-lg shadow-red-500/20 mb-6">
            <XCircle size={32} strokeWidth={2} />
          </div>

          <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
            Errore di Verifica
          </h1>

          <p className="text-slate-600 dark:text-slate-400 mb-6">
            {errorMessage}
          </p>

          <Link
            to="/login"
            className="inline-flex items-center justify-center gap-2 w-full py-3 px-4 rounded-xl font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            <ArrowLeft size={18} />
            Torna al Login
          </Link>
        </div>
      </div>
    </div>
  );
}
