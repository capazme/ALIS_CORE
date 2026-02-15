/**
 * Query Context Panel Component
 *
 * Configuration panel for query context options:
 * - Jurisdiction (nazionale, regionale, comunitario)
 * - Temporal reference (date or 'latest')
 * - User role (cittadino, avvocato, giudice, studente)
 */

import { Collapsible } from '@components/ui/Collapsible';
import { Select, type SelectOption } from '@components/ui/Select';
import { Badge } from '@components/ui/Badge';
import { Settings2 } from 'lucide-react';
import { useQueryStore } from '@/app/store/query';

const JURISDICTION_OPTIONS: SelectOption[] = [
  { value: 'nazionale', label: 'Nazionale (Italia)' },
  { value: 'regionale', label: 'Regionale' },
  { value: 'comunitario', label: 'Comunitario (UE)' },
];

const USER_ROLE_OPTIONS: SelectOption[] = [
  { value: 'cittadino', label: 'Cittadino' },
  { value: 'avvocato', label: 'Avvocato' },
  { value: 'giudice', label: 'Giudice' },
  { value: 'studente', label: 'Studente di Giurisprudenza' },
];

interface QueryContextPanelProps {
  defaultOpen?: boolean;
}

export function QueryContextPanel({ defaultOpen = false }: QueryContextPanelProps) {
  const { queryContext, updateContext } = useQueryStore();

  const handleJurisdictionChange = (value: string) => {
    updateContext({ jurisdiction: value });
  };

  const handleUserRoleChange = (value: string) => {
    updateContext({ user_role: value });
  };

  const handleTemporalReferenceChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const value = e.target.value;
    updateContext({ temporal_reference: value || 'latest' });
  };

  const isLatest =
    !queryContext.temporal_reference || queryContext.temporal_reference === 'latest';

  return (
    <Collapsible
      title="Contesto Query"
      defaultOpen={defaultOpen}
      icon={<Settings2 className="w-4 h-4" aria-hidden="true" />}
      badge={
        <Badge variant="outline">
          Opzionale
        </Badge>
      }
    >
      <div className="space-y-5">
        <p className="text-sm text-slate-400">
          Configura il contesto della tua domanda per ottenere una risposta più accurata
          e pertinente.
        </p>

        {/* Jurisdiction */}
        <Select
          label="Ambito Giurisdizionale"
          options={JURISDICTION_OPTIONS}
          value={queryContext.jurisdiction || 'nazionale'}
          onChange={handleJurisdictionChange}
          helperText="Specifica l'ambito territoriale della normativa applicabile"
        />

        {/* User Role */}
        <Select
          label="Ruolo Utente"
          options={USER_ROLE_OPTIONS}
          value={queryContext.user_role || 'cittadino'}
          onChange={handleUserRoleChange}
          helperText="Il sistema adatterà il linguaggio e il livello di dettaglio in base al tuo ruolo"
        />

        {/* Temporal Reference */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-slate-300">
            Riferimento Temporale
          </label>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <div className="flex-1">
              <input
                type="date"
                value={
                  isLatest
                    ? ''
                    : queryContext.temporal_reference?.split('T')[0] || ''
                }
                onChange={handleTemporalReferenceChange}
                aria-label="Riferimento temporale"
                className="w-full px-4 py-2.5 min-h-[44px] bg-slate-800 border border-slate-700 rounded-lg text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:border-transparent transition-all"
                max={new Date().toISOString().split('T')[0]}
              />
            </div>

            <button
              type="button"
              onClick={() => updateContext({ temporal_reference: 'latest' })}
              className={`px-4 py-2.5 min-h-[44px] rounded-lg font-medium text-sm transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                isLatest
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              Più recente
            </button>
          </div>

          <p className="text-xs text-slate-500">
            {isLatest ? (
              'Verrà utilizzata la normativa più recente disponibile'
            ) : (
              <>
                Interpretazione legale vigente al{' '}
                <span className="font-mono text-blue-400">
                  {new Date(queryContext.temporal_reference!).toLocaleDateString(
                    'it-IT',
                    {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                    }
                  )}
                </span>
              </>
            )}
          </p>
        </div>

        {/* Context Summary */}
        <div className="pt-4 border-t border-slate-700">
          <h4 className="text-sm font-medium text-slate-300 mb-3">
            Riepilogo Contesto
          </h4>
          <div className="flex flex-wrap gap-2">
            <Badge variant="default">
              {JURISDICTION_OPTIONS.find(
                (opt) => opt.value === queryContext.jurisdiction
              )?.label || 'Nazionale'}
            </Badge>
            <Badge variant="secondary">
              {USER_ROLE_OPTIONS.find((opt) => opt.value === queryContext.user_role)
                ?.label || 'Cittadino'}
            </Badge>
            <Badge variant="outline">
              {isLatest
                ? 'Normativa più recente'
                : `Vigente al ${new Date(
                    queryContext.temporal_reference!
                  ).toLocaleDateString('it-IT')}`}
            </Badge>
          </div>
        </div>
      </div>
    </Collapsible>
  );
}
