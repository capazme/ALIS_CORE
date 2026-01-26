/**
 * Registration form component - Invitation-based registration with email verification
 */
import { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { register } from '../../services/authService';
import { validateInvitation } from '../../services/invitationService';
import type { InvitationValidateResponse } from '../../types/api';
import { UserPlus, Mail, Lock, User, Eye, EyeOff, AlertCircle, ArrowLeft, CheckCircle, Check, X, Loader2, UserCircle } from 'lucide-react';
import { cn } from '../../lib/utils';

type RegistrationState = 'loading' | 'invalid' | 'expired' | 'form' | 'success';

export function RegisterForm() {
  const [searchParams] = useSearchParams();
  const invitationToken = searchParams.get('token');

  // State for invitation validation
  const [registrationState, setRegistrationState] = useState<RegistrationState>('loading');
  const [invitation, setInvitation] = useState<InvitationValidateResponse | null>(null);
  const [invitationError, setInvitationError] = useState('');

  // Form fields
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState<'member' | 'researcher'>('member');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [formError, setFormError] = useState('');
  const [loading, setLoading] = useState(false);

  const [focusedField, setFocusedField] = useState<string | null>(null);

  // Validate invitation on mount
  useEffect(() => {
    const checkInvitation = async () => {
      if (!invitationToken) {
        setRegistrationState('invalid');
        setInvitationError('Nessun token di invito fornito. La registrazione richiede un invito valido.');
        return;
      }

      try {
        const result = await validateInvitation(invitationToken);
        if (result.valid) {
          setInvitation(result);
          // Pre-fill email if invitation was for specific email
          if (result.email) {
            setEmail(result.email);
          }
          setRegistrationState('form');
        } else {
          setRegistrationState('invalid');
          setInvitationError('Questo invito non è valido o è già stato utilizzato.');
        }
      } catch (error: unknown) {
        const err = error as { message?: string };
        if (err.message?.includes('expired')) {
          setRegistrationState('expired');
          setInvitationError('Questo invito è scaduto. Richiedi un nuovo invito.');
        } else {
          setRegistrationState('invalid');
          setInvitationError(err.message || 'Invito non valido o scaduto.');
        }
      }
    };

    checkInvitation();
  }, [invitationToken]);

  // Password validation
  const passwordChecks = {
    minLength: password.length >= 8,
    hasUppercase: /[A-Z]/.test(password),
    hasLowercase: /[a-z]/.test(password),
    hasNumber: /\d/.test(password),
  };
  const isPasswordValid = Object.values(passwordChecks).every(Boolean);

  // Password strength calculation
  const getPasswordStrength = () => {
    const checks = Object.values(passwordChecks).filter(Boolean).length;
    if (password.length === 0) return 0;
    return checks;
  };

  const passwordStrength = getPasswordStrength();

  // Username validation
  const isUsernameValid = username.length >= 3 && /^[a-zA-Z0-9_]+$/.test(username);
  const showUsernameValidation = username.length > 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError('');

    if (!invitationToken) {
      setFormError('Token di invito mancante');
      return;
    }

    // Client-side validation
    if (!email || !username || !password || !confirmPassword) {
      setFormError('Email, username e password sono obbligatori');
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setFormError('Formato email non valido');
      return;
    }

    if (username.length < 3) {
      setFormError('Username deve essere almeno 3 caratteri');
      return;
    }

    if (!isUsernameValid) {
      setFormError('Username può contenere solo lettere, numeri e underscore');
      return;
    }

    if (!isPasswordValid) {
      setFormError('Password deve essere almeno 8 caratteri con 1 maiuscola, 1 minuscola e 1 numero');
      return;
    }

    if (password !== confirmPassword) {
      setFormError('Le password non coincidono');
      return;
    }

    try {
      setLoading(true);
      await register({
        invitation_token: invitationToken,
        email,
        username,
        password,
        name: name || undefined,
        role,
      });
      setRegistrationState('success');
    } catch (error: unknown) {
      const err = error as { message?: string };
      setFormError(err.message || 'Errore durante la registrazione. Riprova.');
    } finally {
      setLoading(false);
    }
  };

  // Loading state
  if (registrationState === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#F3F4F6] dark:bg-[#0F172A] relative overflow-hidden px-4">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-[420px] w-full relative z-10 p-6">
          <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl rounded-2xl shadow-2xl ring-1 ring-slate-900/5 dark:ring-white/10 p-8 text-center">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin mx-auto mb-4" />
            <p className="text-slate-600 dark:text-slate-400">Verifica dell'invito in corso...</p>
          </div>
        </div>
      </div>
    );
  }

  // Invalid or expired invitation
  if (registrationState === 'invalid' || registrationState === 'expired') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#F3F4F6] dark:bg-[#0F172A] relative overflow-hidden px-4">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-red-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-orange-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-[420px] w-full relative z-10 p-6">
          <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl rounded-2xl shadow-2xl ring-1 ring-slate-900/5 dark:ring-white/10 p-8 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-tr from-red-500 to-orange-500 text-white rounded-2xl shadow-lg shadow-red-500/20 mb-6">
              <AlertCircle size={32} strokeWidth={2} />
            </div>

            <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
              {registrationState === 'expired' ? 'Invito Scaduto' : 'Invito Non Valido'}
            </h1>

            <p className="text-slate-600 dark:text-slate-400 mb-6">
              {invitationError}
            </p>

            <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl p-4 mb-6">
              <p className="text-sm text-amber-700 dark:text-amber-400">
                La registrazione a VisuaLex richiede un invito valido da un membro esistente.
                Contatta un membro per ricevere un nuovo invito.
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

  // Success state - email verification pending
  if (registrationState === 'success') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#F3F4F6] dark:bg-[#0F172A] relative overflow-hidden px-4">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-green-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-[420px] w-full relative z-10 p-6">
          <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl rounded-2xl shadow-2xl ring-1 ring-slate-900/5 dark:ring-white/10 p-8 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-tr from-green-500 to-emerald-500 text-white rounded-2xl shadow-lg shadow-green-500/20 mb-6">
              <Mail size={32} strokeWidth={2} />
            </div>

            <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
              Verifica la tua Email
            </h1>

            <p className="text-slate-600 dark:text-slate-400 mb-6">
              Ti abbiamo inviato un'email di verifica a:
              <br />
              <strong className="text-slate-800 dark:text-slate-200">{email}</strong>
            </p>

            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4 mb-6">
              <p className="text-sm text-blue-700 dark:text-blue-400">
                Clicca sul link nell'email per attivare il tuo account.
                Il link scadrà tra 24 ore.
              </p>
            </div>

            <div className="text-sm text-slate-500 dark:text-slate-400 mb-6">
              Non hai ricevuto l'email? Controlla la cartella spam o{' '}
              <button className="text-blue-600 hover:underline font-medium">
                richiedi un nuovo link
              </button>
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

  // Registration form
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F3F4F6] dark:bg-[#0F172A] relative overflow-hidden px-4 py-8">
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-[420px] w-full relative z-10 p-6">
        {/* Logo/Header */}
        <div className="text-center mb-6 space-y-3">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-tr from-blue-600 to-indigo-600 text-white rounded-2xl shadow-lg shadow-blue-500/20 mb-2 transform transition-transform hover:scale-105 duration-300">
            <UserPlus size={28} strokeWidth={2.5} />
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white">
            Visua<span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">Lex</span>
          </h1>
          <p className="text-slate-500 dark:text-slate-400 font-medium">
            Crea il tuo account
          </p>
          {invitation?.inviter && (
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Invitato da <span className="font-semibold text-blue-600">{invitation.inviter.username}</span>
            </p>
          )}
        </div>

        {/* Registration Card with Glass Effect */}
        <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl rounded-2xl shadow-2xl ring-1 ring-slate-900/5 dark:ring-white/10 p-8 transition-all duration-300">
          {/* Error Message */}
          {formError && (
            <div className="mb-6 p-4 bg-red-50/50 dark:bg-red-900/10 border border-red-100 dark:border-red-900/30 rounded-xl flex items-start gap-3 animate-in fade-in slide-in-from-top-2">
              <AlertCircle size={20} className="text-red-500 shrink-0 mt-0.5" />
              <p className="text-sm font-medium text-red-600 dark:text-red-400">{formError}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email Field */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 ml-1">
                Email
              </label>
              <div
                className={cn(
                  'relative group transition-all duration-300',
                  focusedField === 'email' ? 'transform scale-[1.02]' : ''
                )}
              >
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail
                    size={18}
                    className={cn(
                      'transition-colors',
                      focusedField === 'email' ? 'text-blue-500' : 'text-slate-400'
                    )}
                  />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={() => setFocusedField('email')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all outline-none text-slate-900 dark:text-slate-100 placeholder:text-slate-400/70 disabled:opacity-60 disabled:cursor-not-allowed"
                  placeholder="nome@email.com"
                  disabled={loading || !!invitation?.email}
                  autoComplete="email"
                />
              </div>
              {invitation?.email && (
                <p className="text-xs text-slate-500 ml-1">Email preimpostata dall'invito</p>
              )}
            </div>

            {/* Username Field */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 ml-1">
                Username
              </label>
              <div
                className={cn(
                  'relative group transition-all duration-300',
                  focusedField === 'username' ? 'transform scale-[1.02]' : ''
                )}
              >
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User
                    size={18}
                    className={cn(
                      'transition-colors',
                      focusedField === 'username' ? 'text-blue-500' : 'text-slate-400'
                    )}
                  />
                </div>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onFocus={() => setFocusedField('username')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full pl-10 pr-10 py-3 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all outline-none text-slate-900 dark:text-slate-100 placeholder:text-slate-400/70"
                  placeholder="mario_rossi"
                  disabled={loading}
                  autoComplete="username"
                />
                {showUsernameValidation && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    {isUsernameValid ? (
                      <Check size={18} className="text-green-500" />
                    ) : (
                      <X size={18} className="text-red-500" />
                    )}
                  </div>
                )}
              </div>
              {showUsernameValidation && !isUsernameValid && (
                <p className="text-xs text-red-500 ml-1">Solo lettere, numeri e underscore (min 3 caratteri)</p>
              )}
            </div>

            {/* Name Field (optional) */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 ml-1">
                Nome Completo <span className="text-slate-400">(opzionale)</span>
              </label>
              <div
                className={cn(
                  'relative group transition-all duration-300',
                  focusedField === 'name' ? 'transform scale-[1.02]' : ''
                )}
              >
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <UserCircle
                    size={18}
                    className={cn(
                      'transition-colors',
                      focusedField === 'name' ? 'text-blue-500' : 'text-slate-400'
                    )}
                  />
                </div>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onFocus={() => setFocusedField('name')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all outline-none text-slate-900 dark:text-slate-100 placeholder:text-slate-400/70"
                  placeholder="Mario Rossi"
                  disabled={loading}
                  autoComplete="name"
                />
              </div>
            </div>

            {/* Role Selection */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 ml-1">
                Ruolo
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setRole('member')}
                  className={cn(
                    'py-3 px-4 rounded-xl border-2 transition-all text-sm font-medium',
                    role === 'member'
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                      : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:border-slate-300 dark:hover:border-slate-600'
                  )}
                  disabled={loading}
                >
                  Membro
                </button>
                <button
                  type="button"
                  onClick={() => setRole('researcher')}
                  className={cn(
                    'py-3 px-4 rounded-xl border-2 transition-all text-sm font-medium',
                    role === 'researcher'
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                      : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:border-slate-300 dark:hover:border-slate-600'
                  )}
                  disabled={loading}
                >
                  Ricercatore
                </button>
              </div>
            </div>

            {/* Password Field */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 ml-1">
                Password
              </label>
              <div
                className={cn(
                  'relative group transition-all duration-300',
                  focusedField === 'password' ? 'transform scale-[1.02]' : ''
                )}
              >
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock
                    size={18}
                    className={cn(
                      'transition-colors',
                      focusedField === 'password' ? 'text-blue-500' : 'text-slate-400'
                    )}
                  />
                </div>
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setFocusedField('password')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full pl-10 pr-12 py-3 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all outline-none text-slate-900 dark:text-slate-100 placeholder:text-slate-400/70"
                  placeholder="••••••••"
                  disabled={loading}
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors rounded-lg hover:bg-slate-100 dark:hover:bg-white/5"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>

              {/* Password Strength Indicator */}
              {password.length > 0 && (
                <div className="flex gap-1 h-1 mt-2">
                  {[1, 2, 3, 4].map((level) => (
                    <div
                      key={level}
                      className={cn(
                        'flex-1 rounded-full transition-colors',
                        passwordStrength >= level
                          ? level === 1 ? 'bg-red-400'
                            : level === 2 ? 'bg-yellow-400'
                            : level === 3 ? 'bg-green-400'
                            : 'bg-green-500'
                          : 'bg-slate-200 dark:bg-slate-700'
                      )}
                    />
                  ))}
                </div>
              )}

              {/* Password Requirements */}
              {password.length > 0 && (
                <div className="mt-2 space-y-1">
                  <div className={cn('flex items-center gap-2 text-xs', passwordChecks.minLength ? 'text-green-600' : 'text-slate-500')}>
                    {passwordChecks.minLength ? <Check size={12} /> : <X size={12} />}
                    Almeno 8 caratteri
                  </div>
                  <div className={cn('flex items-center gap-2 text-xs', passwordChecks.hasUppercase ? 'text-green-600' : 'text-slate-500')}>
                    {passwordChecks.hasUppercase ? <Check size={12} /> : <X size={12} />}
                    Una lettera maiuscola
                  </div>
                  <div className={cn('flex items-center gap-2 text-xs', passwordChecks.hasLowercase ? 'text-green-600' : 'text-slate-500')}>
                    {passwordChecks.hasLowercase ? <Check size={12} /> : <X size={12} />}
                    Una lettera minuscola
                  </div>
                  <div className={cn('flex items-center gap-2 text-xs', passwordChecks.hasNumber ? 'text-green-600' : 'text-slate-500')}>
                    {passwordChecks.hasNumber ? <Check size={12} /> : <X size={12} />}
                    Un numero
                  </div>
                </div>
              )}
            </div>

            {/* Confirm Password Field */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 ml-1">
                Conferma Password
              </label>
              <div
                className={cn(
                  'relative group transition-all duration-300',
                  focusedField === 'confirmPassword' ? 'transform scale-[1.02]' : ''
                )}
              >
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock
                    size={18}
                    className={cn(
                      'transition-colors',
                      focusedField === 'confirmPassword' ? 'text-blue-500' : 'text-slate-400'
                    )}
                  />
                </div>
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  onFocus={() => setFocusedField('confirmPassword')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full pl-10 pr-10 py-3 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all outline-none text-slate-900 dark:text-slate-100 placeholder:text-slate-400/70"
                  placeholder="••••••••"
                  disabled={loading}
                  autoComplete="new-password"
                />
                {confirmPassword.length > 0 && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    {password === confirmPassword ? (
                      <Check size={18} className="text-green-500" />
                    ) : (
                      <X size={18} className="text-red-500" />
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3.5 px-4 rounded-xl font-semibold shadow-lg shadow-blue-500/25 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white transform active:scale-[0.98] transition-all duration-200 flex items-center justify-center gap-2 group disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 mt-6"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  Crea Account
                  <UserPlus size={18} className="group-hover:scale-110 transition-transform" />
                </>
              )}
            </button>
          </form>

          {/* Login Link */}
          <p className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
            Hai già un account?{' '}
            <Link to="/login" className="text-blue-600 hover:underline font-medium">
              Accedi
            </Link>
          </p>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-slate-500 dark:text-slate-400 mt-6">
          © 2026 VisuaLex. All rights reserved.
        </p>
      </div>
    </div>
  );
}
