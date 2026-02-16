/**
 * SchedulesSection
 * =================
 *
 * Sezione per gestione schedule di ingestion automatica.
 * Mostra tabella schedule con stato, azioni CRUD e form creazione.
 *
 * @example
 * ```tsx
 * <SchedulesSection />
 * ```
 */

import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  Plus,
  Trash2,
  PauseCircle,
  PlayCircle,
  AlertCircle,
  Clock,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import {
  getSchedules,
  createSchedule,
  deleteSchedule,
  toggleSchedule,
} from '../../../../services/scheduleService';
import type { IngestionSchedule } from '../../../../services/scheduleService';

// =============================================================================
// CREATE FORM
// =============================================================================

interface CreateFormProps {
  onCreated: () => void;
  onCancel: () => void;
}

function CreateScheduleForm({ onCreated, onCancel }: CreateFormProps) {
  const [tipoAtto, setTipoAtto] = useState('');
  const [cronExpr, setCronExpr] = useState('0 3 * * *');
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null as string | null);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!tipoAtto.trim() || !cronExpr.trim()) return;

    setSubmitting(true);
    setError(null);
    try {
      await createSchedule({
        tipo_atto: tipoAtto.trim(),
        cron_expr: cronExpr.trim(),
        enabled: true,
        description: description.trim() || undefined,
      });
      onCreated();
    } catch (err) {
      setError('Errore nella creazione dello schedule');
      console.error('Failed to create schedule:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div>
          <label htmlFor="schedule-tipo" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
            Tipo Atto
          </label>
          <input
            id="schedule-tipo"
            type="text"
            value={tipoAtto}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTipoAtto(e.target.value)}
            placeholder="es. codice_civile"
            className="w-full px-3 py-1.5 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            required
          />
        </div>
        <div>
          <label htmlFor="schedule-cron" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
            Cron Expression
          </label>
          <input
            id="schedule-cron"
            type="text"
            value={cronExpr}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCronExpr(e.target.value)}
            placeholder="0 3 * * *"
            className="w-full px-3 py-1.5 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            required
          />
        </div>
        <div>
          <label htmlFor="schedule-desc" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
            Descrizione
          </label>
          <input
            id="schedule-desc"
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
          {submitting ? 'Creazione...' : 'Crea Schedule'}
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
// MAIN COMPONENT
// =============================================================================

export function SchedulesSection() {
  const [schedules, setSchedules] = useState([] as IngestionSchedule[]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null as string | null);
  const [showCreateForm, setShowCreateForm] = useState(false);

  const fetchSchedules = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSchedules();
      setSchedules(data.schedules);
    } catch (err) {
      setError('Errore caricamento schedule');
      console.error('Failed to load schedules:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSchedules();
  }, [fetchSchedules]);

  const handleToggle = async (id: number) => {
    try {
      await toggleSchedule(id);
      await fetchSchedules();
    } catch (err) {
      console.error('Failed to toggle schedule:', err);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteSchedule(id);
      await fetchSchedules();
    } catch (err) {
      console.error('Failed to delete schedule:', err);
    }
  };

  const formatRunStatus = (status: string | null) => {
    if (!status) return null;
    const colors: Record<string, string> = {
      success: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
      failed: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
      running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    };
    return (
      <span className={cn('px-2 py-0.5 rounded text-xs', colors[status] || 'bg-slate-100 text-slate-800')}>
        {status}
      </span>
    );
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock size={20} className="text-slate-500" aria-hidden="true" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Ingestion Schedules
          </h3>
        </div>
        <div className="flex gap-2">
          <button
            onClick={fetchSchedules}
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
            Nuovo
          </button>
        </div>
      </div>

      {showCreateForm && (
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <CreateScheduleForm
            onCreated={() => { setShowCreateForm(false); fetchSchedules(); }}
            onCancel={() => setShowCreateForm(false)}
          />
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-8" role="status">
          <RefreshCw size={20} className="animate-spin text-blue-500" aria-hidden="true" />
          <span className="sr-only">Caricamento schedule...</span>
        </div>
      )}

      {error && (
        <div className="p-4 text-center" role="alert">
          <AlertCircle size={24} className="mx-auto text-red-400 mb-2" aria-hidden="true" />
          <p className="text-sm text-slate-500">{error}</p>
        </div>
      )}

      {!loading && !error && schedules.length === 0 && (
        <div className="p-8 text-center text-slate-400 dark:text-slate-500">
          <p className="text-sm">Nessuno schedule configurato.</p>
        </div>
      )}

      {!loading && !error && schedules.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-700/50 text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                <th className="px-4 py-3 text-left">Tipo Atto</th>
                <th className="px-4 py-3 text-left">Cron</th>
                <th className="px-4 py-3 text-center">Stato</th>
                <th className="px-4 py-3 text-left">Ultimo Run</th>
                <th className="px-4 py-3 text-right">Azioni</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {schedules.map((sched: IngestionSchedule) => (
                <tr key={sched.id} className="hover:bg-slate-50 dark:hover:bg-slate-700/30">
                  <td className="px-4 py-3">
                    <span className="font-medium text-slate-900 dark:text-slate-100">
                      {sched.tipo_atto}
                    </span>
                    {sched.description && (
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                        {sched.description}
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <code className="text-xs bg-slate-100 dark:bg-slate-700 px-2 py-0.5 rounded">
                      {sched.cron_expr}
                    </code>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span
                      className={cn(
                        'px-2 py-0.5 rounded text-xs font-medium',
                        sched.enabled
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-400'
                      )}
                    >
                      {sched.enabled ? 'Attivo' : 'Pausa'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    {sched.last_run_at ? (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500 dark:text-slate-400">
                          {new Date(sched.last_run_at).toLocaleString('it-IT', {
                            day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
                          })}
                        </span>
                        {formatRunStatus(sched.last_run_status)}
                      </div>
                    ) : (
                      <span className="text-xs text-slate-400">Mai eseguito</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => handleToggle(sched.id)}
                        className="p-1.5 text-slate-500 hover:text-blue-600 dark:hover:text-blue-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                        aria-label={sched.enabled ? 'Pausa' : 'Riprendi'}
                        title={sched.enabled ? 'Pausa' : 'Riprendi'}
                      >
                        {sched.enabled ? (
                          <PauseCircle size={16} aria-hidden="true" />
                        ) : (
                          <PlayCircle size={16} aria-hidden="true" />
                        )}
                      </button>
                      <button
                        onClick={() => handleDelete(sched.id)}
                        className="p-1.5 text-slate-500 hover:text-red-600 dark:hover:text-red-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                        aria-label="Elimina"
                        title="Elimina"
                      >
                        <Trash2 size={16} aria-hidden="true" />
                      </button>
                    </div>
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

export default SchedulesSection;
