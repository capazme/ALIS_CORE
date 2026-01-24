/**
 * EditEntityDrawer
 * =================
 *
 * Drawer per modificare ogni campo di un'entità pending.
 * Le modifiche vengono inviate come suggested_edits e applicate
 * quando il consenso viene raggiunto.
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Save, Loader2, CheckCircle2, AlertCircle, Edit3 } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { merltService } from '../../../services/merltService';
import { ENTITY_TYPE_OPTIONS, groupByCategory } from '../../../constants/merltTypes';
import type { EntityType, PendingEntity } from '../../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

interface EditEntityDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  entity: PendingEntity;
  userId: string;
}

// Grouped entity options (from shared constants)
const groupedEntityOptions = groupByCategory(ENTITY_TYPE_OPTIONS);

const AMBITO_OPTIONS = [
  { value: 'civile', label: 'Diritto Civile' },
  { value: 'penale', label: 'Diritto Penale' },
  { value: 'amministrativo', label: 'Diritto Amministrativo' },
  { value: 'costituzionale', label: 'Diritto Costituzionale' },
  { value: 'commerciale', label: 'Diritto Commerciale' },
  { value: 'lavoro', label: 'Diritto del Lavoro' },
  { value: 'tributario', label: 'Diritto Tributario' },
  { value: 'processuale_civile', label: 'Procedura Civile' },
  { value: 'processuale_penale', label: 'Procedura Penale' },
  { value: 'europeo', label: 'Diritto Europeo' },
  { value: 'internazionale', label: 'Diritto Internazionale' },
  { value: 'generale', label: 'Generale' },
];

// =============================================================================
// COMPONENT
// =============================================================================

export function EditEntityDrawer({
  isOpen,
  onClose,
  onSuccess,
  entity,
  userId,
}: EditEntityDrawerProps) {
  // Form state - initialized with current entity values
  const [nome, setNome] = useState(entity.nome);
  const [tipo, setTipo] = useState<EntityType>(entity.tipo);
  const [descrizione, setDescrizione] = useState(entity.descrizione || '');
  const [ambito, setAmbito] = useState(entity.ambito || 'generale');
  const [motivazione, setMotivazione] = useState('');

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Reset form when entity changes
  useEffect(() => {
    if (entity && isOpen) {
      setNome(entity.nome);
      setTipo(entity.tipo);
      setDescrizione(entity.descrizione || '');
      setAmbito(entity.ambito || 'generale');
      setMotivazione('');
      setError(null);
      setSuccess(false);
    }
  }, [entity, isOpen]);

  // Check if any field was modified
  const hasChanges =
    nome !== entity.nome ||
    tipo !== entity.tipo ||
    descrizione !== (entity.descrizione || '') ||
    ambito !== (entity.ambito || 'generale');

  // Build suggested_edits object with only changed fields
  const buildSuggestedEdits = () => {
    const edits: Record<string, string> = {};
    if (nome !== entity.nome) edits.nome = nome;
    if (tipo !== entity.tipo) edits.tipo = tipo;
    if (descrizione !== (entity.descrizione || '')) edits.descrizione = descrizione;
    if (ambito !== (entity.ambito || 'generale')) edits.ambito = ambito;
    return edits;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!hasChanges) {
      setError('Nessuna modifica effettuata');
      return;
    }

    if (!motivazione.trim()) {
      setError('Inserisci una motivazione per la modifica');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const suggestedEdits = buildSuggestedEdits();

      // Call validation API with edit vote and suggested_edit
      await merltService.validateEntity({
        entity_id: entity.id,
        user_id: userId,
        vote: 'edit',
        comment: motivazione,
        suggested_edit: suggestedEdits,
      });

      setSuccess(true);

      // Close after success feedback
      setTimeout(() => {
        onSuccess?.();
        onClose();
      }, 1500);
    } catch (err) {
      console.error('Failed to submit edit:', err);
      setError(err instanceof Error ? err.message : 'Errore nell\'invio della modifica');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/30 backdrop-blur-[2px] z-40"
          />

          {/* Drawer */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-[480px] max-w-full bg-white dark:bg-slate-900 shadow-2xl z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
              <div className="flex items-center gap-2">
                <Edit3 className="w-5 h-5 text-amber-500" />
                <h2 className="font-semibold text-slate-900 dark:text-slate-100">
                  Modifica Entità
                </h2>
              </div>
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >
                <X size={20} className="text-slate-500" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {/* Success state */}
              {success ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <CheckCircle2 size={48} className="text-emerald-500 mb-4" />
                  <h3 className="text-lg font-medium text-slate-900 dark:text-slate-100 mb-2">
                    Modifica inviata!
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    La tua proposta di modifica è in attesa di validazione dalla community.
                  </p>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-5">
                  {/* Original values info */}
                  <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-3 text-xs">
                    <p className="text-slate-500 dark:text-slate-400 mb-1">Valori originali:</p>
                    <p className="text-slate-700 dark:text-slate-300">
                      <strong>{entity.nome}</strong> ({entity.tipo})
                    </p>
                    {entity.descrizione && (
                      <p className="text-slate-600 dark:text-slate-400 mt-1 line-clamp-2">
                        {entity.descrizione}
                      </p>
                    )}
                  </div>

                  {/* Nome */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                      Nome *
                    </label>
                    <input
                      type="text"
                      value={nome}
                      onChange={(e) => setNome(e.target.value)}
                      className={cn(
                        "w-full px-3 py-2 rounded-lg border text-sm",
                        "bg-white dark:bg-slate-800",
                        "border-slate-300 dark:border-slate-600",
                        "focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500",
                        "placeholder:text-slate-400",
                        nome !== entity.nome && "border-amber-400 bg-amber-50 dark:bg-amber-900/10"
                      )}
                      placeholder="Nome dell'entità"
                      required
                    />
                    {nome !== entity.nome && (
                      <p className="text-xs text-amber-600 mt-1">Modificato</p>
                    )}
                  </div>

                  {/* Tipo */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                      Tipo *
                    </label>
                    <select
                      value={tipo}
                      onChange={(e) => setTipo(e.target.value as EntityType)}
                      className={cn(
                        "w-full px-3 py-2 rounded-lg border text-sm",
                        "bg-white dark:bg-slate-800",
                        "border-slate-300 dark:border-slate-600",
                        "focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500",
                        tipo !== entity.tipo && "border-amber-400 bg-amber-50 dark:bg-amber-900/10"
                      )}
                      required
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
                    {tipo !== entity.tipo && (
                      <p className="text-xs text-amber-600 mt-1">Modificato</p>
                    )}
                  </div>

                  {/* Descrizione */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                      Descrizione
                    </label>
                    <textarea
                      value={descrizione}
                      onChange={(e) => setDescrizione(e.target.value)}
                      rows={4}
                      className={cn(
                        "w-full px-3 py-2 rounded-lg border text-sm resize-none",
                        "bg-white dark:bg-slate-800",
                        "border-slate-300 dark:border-slate-600",
                        "focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500",
                        "placeholder:text-slate-400",
                        descrizione !== (entity.descrizione || '') && "border-amber-400 bg-amber-50 dark:bg-amber-900/10"
                      )}
                      placeholder="Descrizione dettagliata dell'entità giuridica..."
                    />
                    {descrizione !== (entity.descrizione || '') && (
                      <p className="text-xs text-amber-600 mt-1">Modificato</p>
                    )}
                  </div>

                  {/* Ambito */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                      Ambito Giuridico
                    </label>
                    <select
                      value={ambito}
                      onChange={(e) => setAmbito(e.target.value)}
                      className={cn(
                        "w-full px-3 py-2 rounded-lg border text-sm",
                        "bg-white dark:bg-slate-800",
                        "border-slate-300 dark:border-slate-600",
                        "focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500",
                        ambito !== (entity.ambito || 'generale') && "border-amber-400 bg-amber-50 dark:bg-amber-900/10"
                      )}
                    >
                      {AMBITO_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                    {ambito !== (entity.ambito || 'generale') && (
                      <p className="text-xs text-amber-600 mt-1">Modificato</p>
                    )}
                  </div>

                  {/* Motivazione (required) */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                      Motivazione della modifica *
                    </label>
                    <textarea
                      value={motivazione}
                      onChange={(e) => setMotivazione(e.target.value)}
                      rows={3}
                      className={cn(
                        "w-full px-3 py-2 rounded-lg border text-sm resize-none",
                        "bg-white dark:bg-slate-800",
                        "border-slate-300 dark:border-slate-600",
                        "focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500",
                        "placeholder:text-slate-400"
                      )}
                      placeholder="Spiega perché proponi questa modifica (es. correzione errore, precisazione, fonte dottrinale...)"
                      required
                    />
                  </div>

                  {/* Error message */}
                  {error && (
                    <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                      <AlertCircle size={16} className="text-red-500 flex-shrink-0" />
                      <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
                    </div>
                  )}

                  {/* Summary of changes */}
                  {hasChanges && (
                    <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3">
                      <p className="text-xs font-medium text-amber-800 dark:text-amber-300 mb-2">
                        Modifiche proposte:
                      </p>
                      <ul className="text-xs text-amber-700 dark:text-amber-400 space-y-1">
                        {nome !== entity.nome && (
                          <li>• Nome: "{entity.nome}" → "{nome}"</li>
                        )}
                        {tipo !== entity.tipo && (
                          <li>• Tipo: {entity.tipo} → {tipo}</li>
                        )}
                        {descrizione !== (entity.descrizione || '') && (
                          <li>• Descrizione modificata</li>
                        )}
                        {ambito !== (entity.ambito || 'generale') && (
                          <li>• Ambito: {entity.ambito || 'generale'} → {ambito}</li>
                        )}
                      </ul>
                    </div>
                  )}
                </form>
              )}
            </div>

            {/* Footer */}
            {!success && (
              <div className="p-4 border-t border-slate-200 dark:border-slate-700">
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={onClose}
                    className="flex-1 px-4 py-2.5 text-sm font-medium text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors"
                  >
                    Annulla
                  </button>
                  <button
                    onClick={handleSubmit}
                    disabled={isSubmitting || !hasChanges || !motivazione.trim()}
                    className={cn(
                      "flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-colors",
                      "bg-amber-500 hover:bg-amber-600 text-white",
                      "disabled:opacity-50 disabled:cursor-not-allowed"
                    )}
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 size={16} className="animate-spin" />
                        Invio...
                      </>
                    ) : (
                      <>
                        <Save size={16} />
                        Invia Modifica
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
