/**
 * PrivacySettings Component
 * =========================
 *
 * GDPR privacy controls for users:
 * - Data export (Art. 20 - Portability)
 * - Account deletion (Art. 17 - Erasure)
 */
import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Download,
  Trash2,
  AlertTriangle,
  CheckCircle,
  X,
  Clock,
  Shield,
  Eye,
  EyeOff,
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import * as privacyService from '../../../services/privacyService';
import type { PrivacyStatusResponse } from '../../../types/api';

export interface PrivacySettingsProps {
  onAccountDeleted?: () => void;
  className?: string;
}

export function PrivacySettings({
  onAccountDeleted,
  className,
}: PrivacySettingsProps) {
  const [status, setStatus] = useState<PrivacyStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletePassword, setDeletePassword] = useState('');
  const [deleteReason, setDeleteReason] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load privacy status
  useEffect(() => {
    const loadStatus = async () => {
      try {
        const response = await privacyService.getPrivacyStatus();
        setStatus(response);
        setLoading(false);
      } catch {
        setLoading(false);
      }
    };
    loadStatus();
  }, []);

  // Handle data export
  const handleExport = useCallback(async () => {
    setExporting(true);
    setError(null);

    try {
      const data = await privacyService.exportData();

      // Create and download file
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `visualex-data-export-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setExportSuccess(true);
      setTimeout(() => setExportSuccess(false), 3000);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Errore durante l\'esportazione';
      setError(message);
    } finally {
      setExporting(false);
    }
  }, []);

  // Handle deletion request
  const handleRequestDeletion = useCallback(async () => {
    if (!deletePassword) {
      setError('Inserisci la password per confermare');
      return;
    }

    setDeleting(true);
    setError(null);

    try {
      await privacyService.requestDeletion(deletePassword, deleteReason || undefined);

      // Refresh status
      const newStatus = await privacyService.getPrivacyStatus();
      setStatus(newStatus);

      setShowDeleteModal(false);
      setDeletePassword('');
      setDeleteReason('');

      // Notify parent (e.g., to log out)
      onAccountDeleted?.();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Errore durante la richiesta';
      setError(message);
    } finally {
      setDeleting(false);
    }
  }, [deletePassword, deleteReason, onAccountDeleted]);

  // Handle cancellation
  const handleCancelDeletion = useCallback(async () => {
    setCancelling(true);
    setError(null);

    try {
      await privacyService.cancelDeletion();

      // Refresh status
      const newStatus = await privacyService.getPrivacyStatus();
      setStatus(newStatus);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Errore durante l\'annullamento';
      setError(message);
    } finally {
      setCancelling(false);
    }
  }, []);

  if (loading) {
    return (
      <div className={cn('space-y-4', className)}>
        <div className="h-32 rounded-lg bg-slate-100 dark:bg-slate-800 animate-pulse" />
        <div className="h-32 rounded-lg bg-slate-100 dark:bg-slate-800 animate-pulse" />
      </div>
    );
  }

  return (
    <div className={cn('space-y-6', className)}>
      {/* Pending Deletion Warning */}
      {status?.deletion_pending && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/30"
        >
          <div className="flex items-start gap-3">
            <Clock className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-medium text-amber-800 dark:text-amber-200">
                Cancellazione account in corso
              </h4>
              <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                Il tuo account sarà eliminato definitivamente tra {status.days_remaining} giorni.
                Puoi annullare questa richiesta in qualsiasi momento.
              </p>
              <button
                onClick={handleCancelDeletion}
                disabled={cancelling}
                className={cn(
                  'mt-3 px-4 py-2 text-sm font-medium rounded-md transition-colors',
                  'bg-amber-600 text-white hover:bg-amber-700',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                {cancelling ? 'Annullamento...' : 'Annulla cancellazione'}
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Data Export Section */}
      <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-blue-50 dark:bg-blue-900/30">
            <Download className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div className="flex-1">
            <h4 className="font-medium text-slate-800 dark:text-slate-100">
              Esporta i tuoi dati
            </h4>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              Scarica una copia di tutti i tuoi dati personali in formato JSON
              (GDPR Art. 20 - Diritto alla portabilità).
            </p>
            <button
              onClick={handleExport}
              disabled={exporting}
              className={cn(
                'mt-3 flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors',
                'bg-blue-500 text-white hover:bg-blue-600',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              <Download size={16} />
              {exporting ? 'Preparazione...' : 'Scarica i miei dati'}
            </button>
          </div>
        </div>

        {/* Export success message */}
        {exportSuccess && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-3 p-2 rounded bg-emerald-50 dark:bg-emerald-900/20"
          >
            <p className="text-sm text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
              <CheckCircle size={14} />
              Dati esportati con successo
            </p>
          </motion.div>
        )}
      </div>

      {/* Account Deletion Section */}
      <div className="p-4 rounded-lg border border-red-200 dark:border-red-800/30 bg-red-50/50 dark:bg-red-900/10">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-red-100 dark:bg-red-900/30">
            <Trash2 className="w-5 h-5 text-red-600 dark:text-red-400" />
          </div>
          <div className="flex-1">
            <h4 className="font-medium text-red-800 dark:text-red-200">
              Elimina account
            </h4>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">
              Elimina permanentemente il tuo account e tutti i dati associati
              (GDPR Art. 17 - Diritto all'oblio).
            </p>
            <button
              onClick={() => setShowDeleteModal(true)}
              disabled={status?.deletion_pending}
              className={cn(
                'mt-3 flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors',
                'bg-red-600 text-white hover:bg-red-700',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              <Trash2 size={16} />
              {status?.deletion_pending ? 'Cancellazione già richiesta' : 'Elimina il mio account'}
            </button>
          </div>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30"
        >
          <p className="text-sm text-red-600 dark:text-red-400 flex items-center gap-2">
            <Shield size={16} />
            {error}
          </p>
        </motion.div>
      )}

      {/* Deletion Confirmation Modal */}
      <AnimatePresence>
        {showDeleteModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
            onClick={() => setShowDeleteModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-md bg-white dark:bg-slate-900 rounded-xl shadow-xl overflow-hidden"
            >
              {/* Header */}
              <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  <h3 className="font-semibold text-slate-800 dark:text-slate-100">
                    Conferma eliminazione
                  </h3>
                </div>
                <button
                  onClick={() => setShowDeleteModal(false)}
                  className="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  <X size={20} className="text-slate-500" />
                </button>
              </div>

              {/* Content */}
              <div className="p-4 space-y-4">
                <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/30">
                  <p className="text-sm text-amber-800 dark:text-amber-200">
                    <strong>Attenzione:</strong> questa azione è irreversibile dopo 30 giorni.
                    Tutti i tuoi dati personali saranno eliminati permanentemente.
                  </p>
                </div>

                {/* Password field */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    Conferma con la tua password
                  </label>
                  <div className="relative">
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={deletePassword}
                      onChange={(e) => setDeletePassword(e.target.value)}
                      placeholder="Inserisci la password"
                      className={cn(
                        'w-full px-3 py-2 pr-10 rounded-lg border transition-colors',
                        'bg-white dark:bg-slate-800',
                        'border-slate-200 dark:border-slate-700',
                        'focus:outline-none focus:ring-2 focus:ring-red-500'
                      )}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    >
                      {showPassword ? (
                        <EyeOff size={18} className="text-slate-400" />
                      ) : (
                        <Eye size={18} className="text-slate-400" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Reason field (optional) */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    Motivo (opzionale)
                  </label>
                  <textarea
                    value={deleteReason}
                    onChange={(e) => setDeleteReason(e.target.value)}
                    placeholder="Aiutaci a migliorare..."
                    rows={2}
                    className={cn(
                      'w-full px-3 py-2 rounded-lg border transition-colors resize-none',
                      'bg-white dark:bg-slate-800',
                      'border-slate-200 dark:border-slate-700',
                      'focus:outline-none focus:ring-2 focus:ring-red-500'
                    )}
                  />
                </div>
              </div>

              {/* Footer */}
              <div className="p-4 border-t border-slate-200 dark:border-slate-700 flex gap-3 justify-end">
                <button
                  onClick={() => setShowDeleteModal(false)}
                  className="px-4 py-2 text-sm font-medium rounded-md text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                  Annulla
                </button>
                <button
                  onClick={handleRequestDeletion}
                  disabled={deleting || !deletePassword}
                  className={cn(
                    'px-4 py-2 text-sm font-medium rounded-md transition-colors',
                    'bg-red-600 text-white hover:bg-red-700',
                    'disabled:opacity-50 disabled:cursor-not-allowed'
                  )}
                >
                  {deleting ? 'Elaborazione...' : 'Elimina definitivamente'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default PrivacySettings;
