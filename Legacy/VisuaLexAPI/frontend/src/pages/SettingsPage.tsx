/**
 * SettingsPage
 * ============
 *
 * Pagina impostazioni account utente.
 * Permette di:
 * - Modificare nome utente
 * - Modificare email
 * - Cambiare password
 */

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  User,
  Mail,
  Lock,
  Save,
  AlertCircle,
  CheckCircle,
  Eye,
  EyeOff,
  ArrowLeft,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { cn } from '../lib/utils';
import { useAuth } from '../hooks/useAuth';

// =============================================================================
// INPUT COMPONENT
// =============================================================================

interface FormInputProps {
  label: string;
  type?: string;
  value: string;
  onChange: (value: string) => void;
  icon: typeof User;
  placeholder?: string;
  disabled?: boolean;
  error?: string;
  showToggle?: boolean;
}

function FormInput({
  label,
  type = 'text',
  value,
  onChange,
  icon: Icon,
  placeholder,
  disabled,
  error,
  showToggle,
}: FormInputProps) {
  const [showPassword, setShowPassword] = useState(false);
  const inputType = showToggle ? (showPassword ? 'text' : 'password') : type;

  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
        {label}
      </label>
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Icon size={18} className="text-slate-400" />
        </div>
        <input
          type={inputType}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className={cn(
            'w-full pl-10 pr-10 py-2.5 rounded-lg border transition-colors',
            'bg-white dark:bg-slate-800',
            'text-slate-900 dark:text-slate-100',
            'placeholder:text-slate-400',
            error
              ? 'border-red-300 dark:border-red-700 focus:ring-red-500'
              : 'border-slate-200 dark:border-slate-700 focus:ring-blue-500',
            'focus:outline-none focus:ring-2 focus:border-transparent',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        />
        {showToggle && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute inset-y-0 right-0 pr-3 flex items-center"
          >
            {showPassword ? (
              <EyeOff size={18} className="text-slate-400 hover:text-slate-600" />
            ) : (
              <Eye size={18} className="text-slate-400 hover:text-slate-600" />
            )}
          </button>
        )}
      </div>
      {error && (
        <p className="text-xs text-red-500 flex items-center gap-1">
          <AlertCircle size={12} />
          {error}
        </p>
      )}
    </div>
  );
}

// =============================================================================
// SECTION CARD
// =============================================================================

interface SectionCardProps {
  title: string;
  description: string;
  children: React.ReactNode;
  onSave: () => void;
  isSaving: boolean;
  isSuccess: boolean;
  error?: string;
}

function SectionCard({
  title,
  description,
  children,
  onSave,
  isSaving,
  isSuccess,
  error,
}: SectionCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden"
    >
      <div className="p-5 border-b border-slate-100 dark:border-slate-800">
        <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
          {title}
        </h3>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          {description}
        </p>
      </div>

      <div className="p-5 space-y-4">
        {children}

        {/* Error message */}
        {error && (
          <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30">
            <p className="text-sm text-red-600 dark:text-red-400 flex items-center gap-2">
              <AlertCircle size={16} />
              {error}
            </p>
          </div>
        )}

        {/* Success message */}
        {isSuccess && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800/30"
          >
            <p className="text-sm text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
              <CheckCircle size={16} />
              Modifiche salvate con successo
            </p>
          </motion.div>
        )}
      </div>

      <div className="px-5 py-4 bg-slate-50 dark:bg-slate-800/50 border-t border-slate-100 dark:border-slate-800">
        <button
          onClick={onSave}
          disabled={isSaving}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors',
            'bg-blue-500 hover:bg-blue-600 text-white',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
        >
          <Save size={16} />
          {isSaving ? 'Salvataggio...' : 'Salva modifiche'}
        </button>
      </div>
    </motion.div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function SettingsPage() {
  const navigate = useNavigate();
  const { user, changePassword } = useAuth();

  // Profile form state
  const [username, setUsername] = useState(user?.username || '');
  const [email, setEmail] = useState(user?.email || '');
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileSuccess, setProfileSuccess] = useState(false);
  const [profileError, setProfileError] = useState('');

  // Password form state
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordSaving, setPasswordSaving] = useState(false);
  const [passwordSuccess, setPasswordSuccess] = useState(false);
  const [passwordError, setPasswordError] = useState('');

  // Handle profile save
  const handleProfileSave = async () => {
    setProfileSaving(true);
    setProfileError('');
    setProfileSuccess(false);

    try {
      // TODO: Implement profile update API
      // await updateProfile({ username, email });
      await new Promise((resolve) => setTimeout(resolve, 1000)); // Simulate
      setProfileSuccess(true);
      setTimeout(() => setProfileSuccess(false), 3000);
    } catch (err: any) {
      setProfileError(err.message || 'Errore nel salvataggio');
    } finally {
      setProfileSaving(false);
    }
  };

  // Handle password change
  const handlePasswordSave = async () => {
    // Validation
    if (newPassword !== confirmPassword) {
      setPasswordError('Le password non coincidono');
      return;
    }
    if (newPassword.length < 8) {
      setPasswordError('La password deve essere di almeno 8 caratteri');
      return;
    }

    setPasswordSaving(true);
    setPasswordError('');
    setPasswordSuccess(false);

    try {
      await changePassword(currentPassword, newPassword);
      setPasswordSuccess(true);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setTimeout(() => setPasswordSuccess(false), 3000);
    } catch (err: any) {
      setPasswordError(err.message || 'Errore nel cambio password');
    } finally {
      setPasswordSaving(false);
    }
  };

  return (
    <div className="min-h-full bg-slate-50 dark:bg-slate-950">
      <div className="max-w-2xl mx-auto p-6">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={() => navigate(-1)}
            className="flex items-center gap-2 text-sm text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 mb-4"
          >
            <ArrowLeft size={16} />
            Indietro
          </button>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            Impostazioni Account
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Gestisci le tue informazioni personali e la sicurezza
          </p>
        </div>

        {/* Sections */}
        <div className="space-y-6">
          {/* Profile Section */}
          <SectionCard
            title="Informazioni Profilo"
            description="Aggiorna il tuo nome utente e indirizzo email"
            onSave={handleProfileSave}
            isSaving={profileSaving}
            isSuccess={profileSuccess}
            error={profileError}
          >
            <FormInput
              label="Nome utente"
              value={username}
              onChange={setUsername}
              icon={User}
              placeholder="Il tuo nome utente"
            />
            <FormInput
              label="Email"
              type="email"
              value={email}
              onChange={setEmail}
              icon={Mail}
              placeholder="email@esempio.com"
            />
          </SectionCard>

          {/* Password Section */}
          <SectionCard
            title="Cambia Password"
            description="Aggiorna la password del tuo account"
            onSave={handlePasswordSave}
            isSaving={passwordSaving}
            isSuccess={passwordSuccess}
            error={passwordError}
          >
            <FormInput
              label="Password attuale"
              value={currentPassword}
              onChange={setCurrentPassword}
              icon={Lock}
              placeholder="Inserisci la password attuale"
              showToggle
            />
            <FormInput
              label="Nuova password"
              value={newPassword}
              onChange={setNewPassword}
              icon={Lock}
              placeholder="Inserisci la nuova password"
              showToggle
            />
            <FormInput
              label="Conferma nuova password"
              value={confirmPassword}
              onChange={setConfirmPassword}
              icon={Lock}
              placeholder="Ripeti la nuova password"
              showToggle
              error={
                confirmPassword && newPassword !== confirmPassword
                  ? 'Le password non coincidono'
                  : undefined
              }
            />
            <p className="text-xs text-slate-500 dark:text-slate-400">
              La password deve essere di almeno 8 caratteri
            </p>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

export default SettingsPage;
