/**
 * ApiKeysSection
 * ===============
 *
 * Admin section for API key management (FR45).
 * Shows table of keys with CRUD actions and inline creation form.
 * Raw key is shown only once on creation with copy-to-clipboard.
 *
 * @example
 * ```tsx
 * <ApiKeysSection />
 * ```
 */

import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  Plus,
  Key,
  Copy,
  Check,
  XCircle,
  AlertCircle,
  Shield,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import {
  listApiKeys,
  createApiKey,
  deleteApiKey,
  bootstrapApiKey,
} from '../../../../services/apiKeyService';
import type { ApiKeyInfo, CreateApiKeyResponse } from '../../../../services/apiKeyService';

// =============================================================================
// RAW KEY DISPLAY (shown once after creation)
// =============================================================================

interface RawKeyBannerProps {
  rawKey: string;
  onDismiss: () => void;
}

function RawKeyBanner({ rawKey, onDismiss }: RawKeyBannerProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(rawKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-700 rounded-lg">
      <div className="flex items-start gap-3">
        <Shield size={20} className="text-yellow-600 dark:text-yellow-400 mt-0.5 flex-shrink-0" aria-hidden="true" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200 mb-1">
            API Key creata — salva questa chiave ora!
          </p>
          <p className="text-xs text-yellow-700 dark:text-yellow-300 mb-2">
            Non sara' mai piu' visibile.
          </p>
          <div className="flex items-center gap-2">
            <code className="flex-1 px-3 py-1.5 text-xs bg-white dark:bg-slate-800 rounded border border-yellow-300 dark:border-yellow-600 text-slate-900 dark:text-slate-100 font-mono truncate">
              {rawKey}
            </code>
            <button
              onClick={handleCopy}
              className="p-1.5 text-yellow-700 dark:text-yellow-300 hover:text-yellow-900 dark:hover:text-yellow-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-500 rounded"
              aria-label="Copia chiave"
            >
              {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
            </button>
          </div>
        </div>
        <button
          onClick={onDismiss}
          className="p-1 text-yellow-500 hover:text-yellow-700 dark:hover:text-yellow-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-500 rounded"
          aria-label="Chiudi"
        >
          <XCircle size={16} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// CREATE FORM
// =============================================================================

interface CreateFormProps {
  onCreated: (resp: CreateApiKeyResponse) => void;
  onCancel: () => void;
}

function CreateApiKeyForm({ onCreated, onCancel }: CreateFormProps) {
  const [userId, setUserId] = useState('');
  const [role, setRole] = useState('user');
  const [tier, setTier] = useState('standard');
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null as string | null);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!userId.trim()) return;

    setSubmitting(true);
    setError(null);
    try {
      const resp = await createApiKey({
        user_id: userId.trim(),
        role,
        rate_limit_tier: tier,
        description: description.trim() || undefined,
      });
      onCreated(resp);
    } catch (err) {
      setError('Errore nella creazione della API key');
      console.error('Failed to create API key:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
        <div>
          <label htmlFor="apikey-user" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
            User ID
          </label>
          <input
            id="apikey-user"
            type="text"
            value={userId}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUserId(e.target.value)}
            placeholder="es. user@example.com"
            className="w-full px-3 py-1.5 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            required
          />
        </div>
        <div>
          <label htmlFor="apikey-role" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
            Ruolo
          </label>
          <select
            id="apikey-role"
            value={role}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setRole(e.target.value)}
            className="w-full px-3 py-1.5 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          >
            <option value="admin">Admin</option>
            <option value="user">User</option>
            <option value="guest">Guest</option>
          </select>
        </div>
        <div>
          <label htmlFor="apikey-tier" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
            Rate Limit Tier
          </label>
          <select
            id="apikey-tier"
            value={tier}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setTier(e.target.value)}
            className="w-full px-3 py-1.5 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          >
            <option value="unlimited">Unlimited</option>
            <option value="premium">Premium (1000/h)</option>
            <option value="standard">Standard (100/h)</option>
            <option value="limited">Limited (10/h)</option>
          </select>
        </div>
        <div>
          <label htmlFor="apikey-desc" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
            Descrizione
          </label>
          <input
            id="apikey-desc"
            type="text"
            value={description}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDescription(e.target.value)}
            placeholder="opzionale"
            className="w-full px-3 py-1.5 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          />
        </div>
      </div>
      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
      <div className="flex gap-2">
        <button
          type="submit"
          disabled={submitting}
          className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        >
          {submitting ? 'Creazione...' : 'Crea API Key'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-1.5 text-sm bg-slate-200 dark:bg-slate-600 text-slate-700 dark:text-slate-200 rounded-md hover:bg-slate-300 dark:hover:bg-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        >
          Annulla
        </button>
      </div>
    </form>
  );
}

// =============================================================================
// ROLE/TIER BADGES
// =============================================================================

function RoleBadge({ role }: { role: string }) {
  const colors: Record<string, string> = {
    admin: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
    user: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    guest: 'bg-slate-100 text-slate-800 dark:bg-slate-700 dark:text-slate-400',
  };
  return (
    <span className={cn('px-2 py-0.5 rounded text-xs font-medium', colors[role] || colors.guest)}>
      {role}
    </span>
  );
}

function TierBadge({ tier }: { tier: string }) {
  const colors: Record<string, string> = {
    unlimited: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    premium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    standard: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    limited: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
  };
  return (
    <span className={cn('px-2 py-0.5 rounded text-xs font-medium', colors[tier] || colors.standard)}>
      {tier}
    </span>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ApiKeysSection() {
  const [keys, setKeys] = useState([] as ApiKeyInfo[]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null as string | null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [rawKeyResponse, setRawKeyResponse] = useState(null as CreateApiKeyResponse | null);
  const [revoking, setRevoking] = useState(null as string | null);

  const fetchKeys = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listApiKeys();
      setKeys(data.keys);
    } catch (err) {
      setError('Errore caricamento API keys');
      console.error('Failed to load API keys:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const handleBootstrap = async () => {
    try {
      const resp = await bootstrapApiKey();
      setRawKeyResponse(resp);
      await fetchKeys();
    } catch (err) {
      console.error('Failed to bootstrap:', err);
    }
  };

  const handleCreated = (resp: CreateApiKeyResponse) => {
    setShowCreateForm(false);
    setRawKeyResponse(resp);
    fetchKeys();
  };

  const handleRevoke = async (keyId: string) => {
    setRevoking(keyId);
    try {
      await deleteApiKey(keyId);
      await fetchKeys();
    } catch (err) {
      console.error('Failed to revoke API key:', err);
    } finally {
      setRevoking(null);
    }
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Key size={20} className="text-slate-500" aria-hidden="true" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            API Keys
          </h3>
        </div>
        <div className="flex gap-2">
          <button
            onClick={fetchKeys}
            className="p-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
            aria-label="Aggiorna lista"
          >
            <RefreshCw size={16} aria-hidden="true" />
          </button>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
          >
            <Plus size={14} aria-hidden="true" />
            Nuova
          </button>
        </div>
      </div>

      {/* Raw key banner */}
      {rawKeyResponse && (
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <RawKeyBanner
            rawKey={rawKeyResponse.raw_key}
            onDismiss={() => setRawKeyResponse(null)}
          />
        </div>
      )}

      {/* Create form */}
      {showCreateForm && (
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <CreateApiKeyForm
            onCreated={handleCreated}
            onCancel={() => setShowCreateForm(false)}
          />
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-8" role="status">
          <RefreshCw size={20} className="animate-spin text-blue-500" aria-hidden="true" />
          <span className="sr-only">Caricamento API keys...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-4 text-center" role="alert">
          <AlertCircle size={24} className="mx-auto text-red-400 mb-2" aria-hidden="true" />
          <p className="text-sm text-slate-500">{error}</p>
        </div>
      )}

      {/* Empty state with bootstrap */}
      {!loading && !error && keys.length === 0 && (
        <div className="p-8 text-center">
          <Key size={32} className="mx-auto text-slate-300 dark:text-slate-600 mb-3" aria-hidden="true" />
          <p className="text-sm text-slate-400 dark:text-slate-500 mb-4">
            Nessuna API key configurata.
          </p>
          <button
            onClick={handleBootstrap}
            className="px-4 py-2 text-sm bg-purple-600 text-white rounded-md hover:bg-purple-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2"
          >
            Bootstrap Admin Key
          </button>
        </div>
      )}

      {/* Keys table */}
      {!loading && !error && keys.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-700/50 text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                <th className="px-4 py-3 text-left">Key ID</th>
                <th className="px-4 py-3 text-center">Role</th>
                <th className="px-4 py-3 text-center">Tier</th>
                <th className="px-4 py-3 text-center">Status</th>
                <th className="px-4 py-3 text-left">User</th>
                <th className="px-4 py-3 text-right">Last Used</th>
                <th className="px-4 py-3 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {keys.map((k: ApiKeyInfo) => (
                <tr key={k.key_id} className="hover:bg-slate-50 dark:hover:bg-slate-700/30">
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs text-slate-900 dark:text-slate-100">
                      {k.key_id}
                    </span>
                    {k.description && (
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                        {k.description}
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <RoleBadge role={k.role} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <TierBadge tier={k.rate_limit_tier} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span
                      className={cn(
                        'px-2 py-0.5 rounded text-xs font-medium',
                        k.is_active
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                      )}
                    >
                      {k.is_active ? 'Active' : 'Revoked'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300">
                    {k.user_id}
                  </td>
                  <td className="px-4 py-3 text-right text-xs text-slate-500 dark:text-slate-400">
                    {k.last_used_at
                      ? new Date(k.last_used_at).toLocaleString('it-IT', {
                          day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
                        })
                      : 'Mai'}
                  </td>
                  <td className="px-4 py-3 text-center">
                    {k.is_active && (
                      <button
                        onClick={() => handleRevoke(k.key_id)}
                        disabled={revoking === k.key_id}
                        className="px-3 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
                      >
                        {revoking === k.key_id ? 'Revoking...' : 'Revoke'}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default ApiKeysSection;
