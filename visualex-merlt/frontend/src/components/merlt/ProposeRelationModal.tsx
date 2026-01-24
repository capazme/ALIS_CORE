/**
 * ProposeRelationModal
 * ====================
 *
 * Modal per proporre una nuova relazione tra entità nel Knowledge Graph MERL-T.
 * La relazione proposta entra nella coda di validazione community.
 */

import { useState } from 'react';
import { Modal } from '../../ui/Modal';
import { Link2, Loader2, CheckCircle2, AlertCircle, ArrowRight, Sparkles } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { merltService } from '../../../services/merltService';
import type { RelationType, PendingEntity } from '../../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

interface ProposeRelationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  // Context
  articleUrn: string;
  userId: string;
  // Available entities to link to
  availableEntities?: PendingEntity[];
}

// =============================================================================
// RELATION TYPE OPTIONS (most common)
// =============================================================================

const RELATION_TYPE_OPTIONS: { value: RelationType; label: string; description: string }[] = [
  // Semantic
  { value: 'DISCIPLINA', label: 'Disciplina', description: 'Regola una materia' },
  { value: 'DEFINISCE', label: 'Definisce', description: 'Fornisce una definizione' },
  { value: 'PREVEDE', label: 'Prevede', description: 'Stabilisce una regola' },
  { value: 'APPLICA_A', label: 'Si applica a', description: 'Ambito di applicazione' },
  // Dependency
  { value: 'PRESUPPONE', label: 'Presuppone', description: 'Richiede come condizione' },
  { value: 'DIPENDE_DA', label: 'Dipende da', description: 'Dipendenza logica' },
  { value: 'SPECIES', label: 'È tipo di', description: 'Relazione di specializzazione' },
  // Citation
  { value: 'CITA', label: 'Cita', description: 'Riferimento esplicito' },
  { value: 'INTERPRETA', label: 'Interpreta', description: 'Fornisce interpretazione' },
  // Modification
  { value: 'DEROGA_A', label: 'Deroga a', description: 'Eccezione alla regola' },
  { value: 'INTEGRA', label: 'Integra', description: 'Completa/aggiunge' },
  { value: 'SOSTITUISCE', label: 'Sostituisce', description: 'Rimpiazza precedente' },
  // Effects
  { value: 'PRODUCE_EFFETTO', label: 'Produce effetto', description: 'Conseguenza giuridica' },
  { value: 'TUTELA', label: 'Tutela', description: 'Protegge interesse' },
  { value: 'LIMITA', label: 'Limita', description: 'Pone restrizioni' },
  { value: 'PREVEDE_SANZIONE', label: 'Prevede sanzione', description: 'Conseguenza per violazione' },
  // Generic
  { value: 'CORRELATO', label: 'Correlato', description: 'Relazione generica' },
];

// =============================================================================
// COMPONENT
// =============================================================================

export function ProposeRelationModal({
  isOpen,
  onClose,
  onSuccess,
  articleUrn,
  userId,
  availableEntities = [],
}: ProposeRelationModalProps) {
  // Form state
  const [tipoRelazione, setTipoRelazione] = useState<RelationType>('DISCIPLINA');
  const [targetEntityId, setTargetEntityId] = useState('');
  const [targetManual, setTargetManual] = useState('');
  const [descrizione, setDescrizione] = useState('');
  const [certezza, setCertezza] = useState(0.7);
  const [useManualTarget, setUseManualTarget] = useState(availableEntities.length === 0);

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Reset form
  const resetForm = () => {
    setTipoRelazione('DISCIPLINA');
    setTargetEntityId('');
    setTargetManual('');
    setDescrizione('');
    setCertezza(0.7);
    setError(null);
    setSuccess(false);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  // Submit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    const target = useManualTarget ? targetManual.trim() : targetEntityId;
    if (!target) {
      setError('Seleziona o inserisci l\'entità di destinazione');
      return;
    }
    if (!descrizione.trim()) {
      setError('La descrizione è obbligatoria');
      return;
    }
    if (descrizione.trim().length < 10) {
      setError('La descrizione deve essere di almeno 10 caratteri');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await merltService.proposeRelation({
        tipo_relazione: tipoRelazione,
        source_urn: articleUrn,
        target_entity_id: target,
        article_urn: articleUrn,
        descrizione: descrizione.trim(),
        certezza,
        user_id: userId,
      });

      setSuccess(true);

      // Callback after short delay
      setTimeout(() => {
        onSuccess?.();
        handleClose();
      }, 1500);
    } catch (err: any) {
      setError(err.message || 'Errore nella proposta della relazione');
    } finally {
      setIsSubmitting(false);
    }
  };

  const selectedRelationType = RELATION_TYPE_OPTIONS.find(r => r.value === tipoRelazione);

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Proponi Nuova Relazione" size="md">
      {success ? (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 size={28} />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
            Relazione proposta!
          </h3>
          <p className="text-slate-500 dark:text-slate-400">
            La tua proposta è in coda per la validazione community.
          </p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Info banner */}
          <div className="flex items-start gap-3 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/30 rounded-lg">
            <Sparkles size={18} className="text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-amber-700 dark:text-amber-300">
              Le relazioni collegano concetti e norme nel Knowledge Graph, creando una rete di conoscenza giuridica.
            </p>
          </div>

          {/* Visual representation */}
          <div className="flex items-center justify-center gap-3 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl">
            <div className="text-center">
              <div className="px-3 py-1.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-lg text-sm font-medium">
                Articolo corrente
              </div>
              <p className="text-xs text-slate-500 mt-1 truncate max-w-[120px]">
                {articleUrn.split(';').pop() || articleUrn}
              </p>
            </div>
            <div className="flex flex-col items-center">
              <ArrowRight size={20} className="text-slate-400" />
              <span className="text-xs text-slate-500 mt-1">
                {selectedRelationType?.label || 'Relazione'}
              </span>
            </div>
            <div className="text-center">
              <div className="px-3 py-1.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg text-sm font-medium">
                {useManualTarget
                  ? (targetManual || 'Entità target')
                  : (availableEntities.find(e => e.id === targetEntityId)?.nome || 'Seleziona...')
                }
              </div>
            </div>
          </div>

          {/* Relation Type */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Tipo di relazione *
            </label>
            <select
              value={tipoRelazione}
              onChange={(e) => setTipoRelazione(e.target.value as RelationType)}
              className={cn(
                'w-full px-4 py-3 rounded-xl border transition-all',
                'bg-slate-50 dark:bg-slate-800',
                'border-slate-200 dark:border-slate-700',
                'text-slate-900 dark:text-white',
                'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none'
              )}
            >
              {RELATION_TYPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label} - {option.description}
                </option>
              ))}
            </select>
          </div>

          {/* Target Entity */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Entità di destinazione *
              </label>
              {availableEntities.length > 0 && (
                <button
                  type="button"
                  onClick={() => setUseManualTarget(!useManualTarget)}
                  className="text-xs text-primary-600 hover:text-primary-700 dark:text-primary-400"
                >
                  {useManualTarget ? 'Seleziona da lista' : 'Inserisci manualmente'}
                </button>
              )}
            </div>

            {useManualTarget ? (
              <input
                type="text"
                value={targetManual}
                onChange={(e) => setTargetManual(e.target.value)}
                placeholder="Es. Buona fede, Art. 1176 c.c., Responsabilità contrattuale..."
                className={cn(
                  'w-full px-4 py-3 rounded-xl border transition-all',
                  'bg-slate-50 dark:bg-slate-800',
                  'border-slate-200 dark:border-slate-700',
                  'text-slate-900 dark:text-white placeholder-slate-400',
                  'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none'
                )}
              />
            ) : (
              <select
                value={targetEntityId}
                onChange={(e) => setTargetEntityId(e.target.value)}
                className={cn(
                  'w-full px-4 py-3 rounded-xl border transition-all',
                  'bg-slate-50 dark:bg-slate-800',
                  'border-slate-200 dark:border-slate-700',
                  'text-slate-900 dark:text-white',
                  'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none'
                )}
              >
                <option value="">Seleziona entità...</option>
                {availableEntities.map((entity) => (
                  <option key={entity.id} value={entity.id}>
                    {entity.nome} ({entity.tipo})
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Descrizione */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Motivazione *
            </label>
            <textarea
              value={descrizione}
              onChange={(e) => setDescrizione(e.target.value)}
              placeholder="Spiega perché questa relazione esiste e come si evince dal testo..."
              rows={3}
              className={cn(
                'w-full px-4 py-3 rounded-xl border transition-all resize-none',
                'bg-slate-50 dark:bg-slate-800',
                'border-slate-200 dark:border-slate-700',
                'text-slate-900 dark:text-white placeholder-slate-400',
                'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none'
              )}
            />
            <p className="mt-1 text-xs text-slate-400">
              Minimo 10 caratteri ({descrizione.length}/10)
            </p>
          </div>

          {/* Certezza */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Certezza: {Math.round(certezza * 100)}%
            </label>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.1"
              value={certezza}
              onChange={(e) => setCertezza(parseFloat(e.target.value))}
              className="w-full accent-primary-600"
            />
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>Possibile</span>
              <span>Certa</span>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2.5 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors"
            >
              Annulla
            </button>
            <button
              type="submit"
              disabled={isSubmitting || (!targetEntityId && !targetManual.trim()) || !descrizione.trim()}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all',
                'bg-primary-600 hover:bg-primary-700 text-white',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              {isSubmitting ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Invio...
                </>
              ) : (
                <>
                  <Link2 size={18} />
                  Proponi
                </>
              )}
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}
