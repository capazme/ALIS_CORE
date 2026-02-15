/**
 * Query Form Component
 *
 * Main form for legal query submission with Zod validation.
 * Supports character counting, validation, and loading states.
 */

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Textarea } from '@components/ui/Textarea';
import { Button } from '@components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '@components/ui/Card';
import { Search, Loader } from 'lucide-react';
import { queryFormSchema, type QueryFormData } from '../validation';
import { useQueryStore } from '@/app/store/query';

interface QueryFormProps {
  onSubmit: (data: QueryFormData) => void | Promise<void>;
  isSubmitting?: boolean;
  error?: string | null;
}

export function QueryForm({ onSubmit, isSubmitting = false, error }: QueryFormProps) {
  const { currentQuery, setCurrentQuery } = useQueryStore();

  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
    watch,
  } = useForm<QueryFormData>({
    resolver: zodResolver(queryFormSchema),
    mode: 'onChange',
    defaultValues: {
      query: currentQuery,
    },
  });

  // Watch query value to sync with store
  const queryValue = watch('query');

  // Sync with store on change
  const handleQueryChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCurrentQuery(e.target.value);
  };

  const handleFormSubmit = async (data: QueryFormData) => {
    // Save to store before submitting
    setCurrentQuery(data.query);
    await onSubmit(data);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="w-5 h-5 text-blue-400" aria-hidden="true" />
          Interroga il Sistema MERL-T
        </CardTitle>
        <p className="text-sm text-slate-400 mt-2">
          Poni una domanda legale e ricevi una risposta AI-powered basata su normativa,
          giurisprudenza e dottrina.
        </p>
      </CardHeader>

      <CardContent>
        <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-6" role="form" aria-label="Interroga il Sistema MERL-T">
          {/* Query Input */}
          <Textarea
            {...register('query')}
            label="La tua domanda legale"
            placeholder="Es: È valido un contratto firmato da un sedicenne? Quali sono le conseguenze legali?"
            maxLength={2000}
            showCharCount
            error={errors.query?.message}
            helperText={
              !errors.query
                ? 'Fornisci tutti i dettagli rilevanti per ottenere una risposta accurata'
                : undefined
            }
            required
            disabled={isSubmitting}
            rows={6}
            className="font-mono"
            onChange={(e) => {
              register('query').onChange(e);
              handleQueryChange(e);
            }}
          />

          {/* Examples (Optional) */}
          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
            <h4 className="text-sm font-medium text-slate-300 mb-2">
              Esempi di domande:
            </h4>
            <ul className="text-sm text-slate-400 space-y-1.5">
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                <span>
                  Quali sono i requisiti per la validità di un testamento olografo secondo
                  il Codice Civile?
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                <span>
                  Un contratto stipulato sotto minaccia può essere annullato? Quali sono
                  i termini di decadenza?
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                <span>
                  Come si configura la responsabilità extracontrattuale secondo l'Art.
                  2043 CC?
                </span>
              </li>
            </ul>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4" role="alert">
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          {/* Submit Button */}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs text-slate-500">
              La risposta sarà generata consultando 4 esperti legali AI e verrà fornita
              una traccia di esecuzione completa.
            </p>

            <Button
              type="submit"
              disabled={!isValid || isSubmitting}
              loading={isSubmitting}
              className="min-w-[160px] min-h-[44px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              {isSubmitting ? (
                <>
                  <Loader className="w-4 h-4 mr-2 animate-spin" aria-hidden="true" />
                  Elaborazione...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" aria-hidden="true" />
                  Invia Query
                </>
              )}
            </Button>
          </div>

          {/* Query Stats */}
          <div className="flex items-center gap-4 text-xs text-slate-500 pt-2 border-t border-slate-700">
            <span>
              Caratteri: {queryValue?.length || 0} / 2000
            </span>
            <span>•</span>
            <span>
              Parole: {queryValue?.trim().split(/\s+/).filter(Boolean).length || 0}
            </span>
            <span>•</span>
            <span className={isValid ? 'text-green-400' : 'text-slate-500'}>
              {isValid ? '✓ Valida' : '✗ Invalida'}
            </span>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
