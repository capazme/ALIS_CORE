/**
 * ProposeEntityModal
 * ==================
 *
 * Modal per proporre una nuova entità al Knowledge Graph MERL-T.
 * L'entità proposta entra nella coda di validazione community.
 * Include gestione duplicati con warning e conferma.
 */

import { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Plus, Loader2, CheckCircle2, AlertCircle, Sparkles, AlertTriangle } from 'lucide-react';
import { cn } from '../../lib/utils';
import { merltService } from '../../services/merltService';
import { ENTITY_TYPE_OPTIONS, groupByCategory } from '../../constants/merltTypes';
import type { EntityType, PendingEntity, DuplicateCandidate } from '../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

interface ProposeEntityModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (entity: PendingEntity) => void;
  // Context
  articleUrn: string;
  articleText?: string;
  userId: string;
  ambito?: string;
}

// Grouped entity type options (from shared constants)
const groupedEntityOptions = groupByCategory(ENTITY_TYPE_OPTIONS);

// =============================================================================
// COMPONENT
// =============================================================================

export function ProposeEntityModal({
  isOpen,
  onClose,
  onSuccess,
  articleUrn,
  articleText: _articleText,
  userId,
  ambito = 'civile',
}: ProposeEntityModalProps) {
  // Form state
  const [tipo, setTipo] = useState('concetto' as EntityType);
  const [nome, setNome] = useState('');
  const [descrizione, setDescrizione] = useState('');
  const [evidence, setEvidence] = useState('');

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null as string | null);
  const [success, setSuccess] = useState(false);

  // Duplicate detection state
  const [duplicatesFound, setDuplicatesFound] = useState(false);
  const [duplicates, setDuplicates] = useState([] as DuplicateCandidate[]);
  const [selectedDuplicate, setSelectedDuplicate] = useState(null as string | null);

  // Reset form
  const resetForm = () => {
    setTipo('concetto');
    setNome('');
    setDescrizione('');
    setEvidence('');
    setError(null);
    setSuccess(false);
    setDuplicatesFound(false);
    setDuplicates([]);
    setSelectedDuplicate(null);
  };

  // Go back from duplicates view to form
  const handleBackToForm = () => {
    setDuplicatesFound(false);
    setDuplicates([]);
    setSelectedDuplicate(null);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  // Submit - handles both initial check and confirmed submission
  const handleSubmit = async (e: React.FormEvent, skipDuplicateCheck = false) => {
    e.preventDefault();

    // Validation
    if (!nome.trim()) {
      setError('Il nome è obbligatorio');
      return;
    }
    if (nome.trim().length < 3) {
      setError('Il nome deve essere di almeno 3 caratteri');
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
      const result = await merltService.proposeEntity({
        tipo,
        nome: nome.trim(),
        descrizione: descrizione.trim(),
        article_urn: articleUrn,
        ambito,
        evidence: evidence.trim() || `Proposto manualmente per ${articleUrn}`,
        user_id: userId,
        skip_duplicate_check: skipDuplicateCheck,
        acknowledged_duplicate_of: skipDuplicateCheck ? selectedDuplicate || undefined : undefined,
      });

      // Check if duplicates were found (blocking = exact match only)
      if (result.duplicate_action_required && result.duplicates.length > 0) {
        setDuplicates(result.duplicates);
        setDuplicatesFound(true);
        setIsSubmitting(false);
        return;
      }

      setSuccess(true);

      // Callback after short delay
      setTimeout(() => {
        if (result.pending_entity) {
          onSuccess?.(result.pending_entity);
        }
        handleClose();
      }, 1500);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Errore nella proposta dell\'entità';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Confirm creation despite duplicates
  const handleConfirmCreate = async () => {
    const syntheticEvent = { preventDefault: () => {} } as React.FormEvent;
    await handleSubmit(syntheticEvent, true);
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Proponi Nuova Entità" size="md">
      {success ? (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 size={28} />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
            Entità proposta!
          </h3>
          <p className="text-slate-500 dark:text-slate-400">
            La tua proposta è in coda per la validazione community.
          </p>
        </div>
      ) : duplicatesFound ? (
        /* ============================================= */
        /* DUPLICATES WARNING VIEW                       */
        /* ============================================= */
        <div className="space-y-5">
          {/* Warning header */}
          <div className="flex items-start gap-3 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-300 dark:border-amber-700 rounded-xl">
            <AlertTriangle size={24} className="text-amber-600 dark:text-amber-400 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-amber-800 dark:text-amber-200">
                Entità simili trovate
              </h3>
              <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                Abbiamo trovato {duplicates.length} entità simili a "{nome}".
                Verifica se la tua proposta è un duplicato.
              </p>
            </div>
          </div>

          {/* Your proposal summary */}
          <div className="p-3 bg-slate-100 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">La tua proposta:</p>
            <p className="font-medium text-slate-900 dark:text-white">{nome}</p>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">{tipo}</p>
          </div>

          {/* Duplicates list */}
          <div className="space-y-3 max-h-[250px] overflow-y-auto">
            <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Entità esistenti simili:
            </p>

            {duplicates.map((dup: DuplicateCandidate) => (
              <div
                key={dup.entity_id}
                onClick={() => setSelectedDuplicate(
                  selectedDuplicate === dup.entity_id ? null : dup.entity_id
                )}
                className={cn(
                  "p-4 rounded-xl border-2 cursor-pointer transition-all",
                  selectedDuplicate === dup.entity_id
                    ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
                    : "border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600"
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-slate-900 dark:text-white truncate">
                        {dup.entity_text}
                      </span>
                      <span className={cn(
                        "px-2 py-0.5 text-xs rounded-full flex-shrink-0",
                        dup.confidence === 'exact'
                          ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                          : dup.confidence === 'high'
                          ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
                          : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                      )}>
                        {dup.confidence === 'exact' ? 'Match esatto' :
                         dup.confidence === 'high' ? 'Molto simile' :
                         `${Math.round(dup.similarity_score * 100)}% simile`}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
                      Tipo: {dup.entity_type} | Stato: {dup.validation_status}
                    </p>
                    {dup.descrizione && (
                      <p className="text-sm text-slate-600 dark:text-slate-300 line-clamp-2">
                        {dup.descrizione}
                      </p>
                    )}
                  </div>

                  {/* Selection indicator */}
                  <div className={cn(
                    "w-5 h-5 rounded-full border-2 flex-shrink-0 flex items-center justify-center transition-colors",
                    selectedDuplicate === dup.entity_id
                      ? "border-primary-500 bg-primary-500"
                      : "border-slate-300 dark:border-slate-600"
                  )}>
                    {selectedDuplicate === dup.entity_id && (
                      <div className="w-2 h-2 bg-white rounded-full" />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="space-y-3 pt-4 border-t border-slate-200 dark:border-slate-700">
            {/* Option 1: Use existing (if selected) */}
            {selectedDuplicate && (
              <button
                type="button"
                onClick={handleClose}
                className={cn(
                  "w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium transition-all",
                  "bg-primary-600 hover:bg-primary-700 text-white"
                )}
              >
                <CheckCircle2 size={18} />
                Usa entità esistente
              </button>
            )}

            {/* Option 2: Create anyway */}
            <button
              type="button"
              onClick={handleConfirmCreate}
              disabled={isSubmitting}
              className={cn(
                "w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium transition-all",
                selectedDuplicate
                  ? "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700"
                  : "bg-amber-500 hover:bg-amber-600 text-white",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {isSubmitting ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Plus size={18} />
              )}
              {isSubmitting ? 'Creazione...' : 'Crea comunque nuova entità'}
            </button>

            {/* Option 3: Go back */}
            <button
              type="button"
              onClick={handleBackToForm}
              className="w-full px-4 py-2.5 text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors text-sm"
            >
              Torna indietro e modifica
            </button>
          </div>
        </div>
      ) : (
        <form onSubmit={(e) => handleSubmit(e)} className="space-y-5">
          {/* Info banner */}
          <div className="flex items-start gap-3 p-3 bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800/30 rounded-lg">
            <Sparkles size={18} className="text-primary-600 dark:text-primary-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-primary-700 dark:text-primary-300">
              La tua proposta sarà validata dalla community. Se approvata, verrà aggiunta al Knowledge Graph.
            </p>
          </div>

          {/* Entity Type */}
          <div>
            <label htmlFor="modal-entity-tipo" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Tipo di entità *
            </label>
            <select
              id="modal-entity-tipo"
              value={tipo}
              onChange={(e) => setTipo(e.target.value as EntityType)}
              className={cn(
                'w-full px-4 py-3 rounded-xl border transition-all',
                'bg-slate-50 dark:bg-slate-800',
                'border-slate-200 dark:border-slate-700',
                'text-slate-900 dark:text-white',
                'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none'
              )}
            >
              {Object.entries(groupedEntityOptions).map(([category, options]) => (
                <optgroup key={category} label={category}>
                  {options.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label} - {option.description}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          {/* Nome */}
          <div>
            <label htmlFor="modal-entity-nome" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Nome *
            </label>
            <input
              id="modal-entity-nome"
              type="text"
              value={nome}
              onChange={(e) => setNome(e.target.value)}
              placeholder="Es. Buona fede, Creditore, Mora del debitore..."
              className={cn(
                'w-full px-4 py-3 rounded-xl border transition-all',
                'bg-slate-50 dark:bg-slate-800',
                'border-slate-200 dark:border-slate-700',
                'text-slate-900 dark:text-white placeholder-slate-400',
                'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none'
              )}
            />
          </div>

          {/* Descrizione */}
          <div>
            <label htmlFor="modal-entity-descrizione" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Descrizione *
            </label>
            <textarea
              id="modal-entity-descrizione"
              value={descrizione}
              onChange={(e) => setDescrizione(e.target.value)}
              placeholder="Descrivi brevemente cosa rappresenta questa entità nel contesto giuridico..."
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

          {/* Evidence (optional) */}
          <div>
            <label htmlFor="modal-entity-evidence" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Citazione testuale (opzionale)
            </label>
            <textarea
              id="modal-entity-evidence"
              value={evidence}
              onChange={(e) => setEvidence(e.target.value)}
              placeholder="Cita il passaggio dell'articolo che supporta questa entità..."
              rows={2}
              className={cn(
                'w-full px-4 py-3 rounded-xl border transition-all resize-none',
                'bg-slate-50 dark:bg-slate-800',
                'border-slate-200 dark:border-slate-700',
                'text-slate-900 dark:text-white placeholder-slate-400',
                'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none'
              )}
            />
          </div>

          {/* Error */}
          {error && (
            <div role="alert" className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" aria-hidden="true" />
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2.5 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 min-h-[44px]"
            >
              Annulla
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !nome.trim() || !descrizione.trim()}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all min-h-[44px]',
                'bg-primary-600 hover:bg-primary-700 text-white',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
              )}
            >
              {isSubmitting ? (
                <>
                  <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                  Invio...
                </>
              ) : (
                <>
                  <Plus size={18} aria-hidden="true" />
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
