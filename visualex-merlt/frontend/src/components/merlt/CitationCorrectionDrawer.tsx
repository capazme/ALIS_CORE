/**
 * CitationCorrectionDrawer
 * =========================
 *
 * Drawer laterale per correggere/annotare citazioni normative rilevate automaticamente.
 *
 * Due modalità di utilizzo:
 * 1. CitationPreviewPopup (correzione) - L'utente vede una citazione parsed automaticamente
 *    e può correggerla se errata
 * 2. SelectionPopup (annotazione) - L'utente seleziona testo che contiene una citazione
 *    non rilevata e la annota manualmente
 *
 * Flusso NER Training:
 * - Ogni correzione/annotazione viene inviata al backend MERL-T
 * - Il backend salva l'esempio nel buffer RLCF
 * - Quando il buffer raggiunge la soglia, parte il fine-tuning del modello NER
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertCircle, CheckCircle2, Loader2, Save } from 'lucide-react';
import { cn } from '../../lib/utils';
import { merltService } from '../../services/merltService';
import type { NERFeedbackRequest, NERFeedbackResponse, ParsedCitationData } from '../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface CitationCorrectionDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  // Dati selezione
  selectedText: string;
  startOffset: number;
  endOffset: number;
  articleText: string;  // Testo completo per context window
  articleUrn: string;
  // Parsing originale (se da hover preview)
  originalParsed?: ParsedCitationData;
  confidenceBefore?: number;
  // Source
  source: 'citation_preview' | 'selection_popup';
  // User
  userId: string;
  // Callback
  onSuccess?: (response: NERFeedbackResponse) => void;
}

// Opzioni dropdown tipo atto
const TIPO_ATTO_OPTIONS = [
  { value: 'legge', label: 'Legge' },
  { value: 'decreto legislativo', label: 'Decreto Legislativo (D.Lgs.)' },
  { value: 'decreto legge', label: 'Decreto Legge (D.L.)' },
  { value: 'decreto del presidente della repubblica', label: 'D.P.R.' },
  { value: 'codice civile', label: 'Codice Civile' },
  { value: 'codice penale', label: 'Codice Penale' },
  { value: 'codice di procedura civile', label: 'C.P.C.' },
  { value: 'codice di procedura penale', label: 'C.P.P.' },
  { value: 'costituzione', label: 'Costituzione' },
  { value: 'regio decreto', label: 'Regio Decreto' },
  { value: 'regolamento ue', label: 'Regolamento UE' },
  { value: 'direttiva ue', label: 'Direttiva UE' },
];

// =============================================================================
// COMPONENT
// =============================================================================

export function CitationCorrectionDrawer({
  isOpen,
  onClose,
  selectedText,
  startOffset,
  endOffset,
  articleText,
  articleUrn,
  originalParsed,
  confidenceBefore,
  source,
  userId,
  onSuccess,
}: CitationCorrectionDrawerProps) {
  // Form state
  const [tipoAtto, setTipoAtto] = useState('');
  const [numero, setNumero] = useState('');
  const [anno, setAnno] = useState('');
  const [articoli, setArticoli] = useState('');

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null as string | null);
  const [success, setSuccess] = useState(false);

  // Pre-fill from original parsed data
  useEffect(() => {
    if (originalParsed && isOpen) {
      setTipoAtto(originalParsed.actType || '');
      setNumero(originalParsed.actNumber || '');
      setAnno(originalParsed.date || '');
      setArticoli(originalParsed.articles?.join(', ') || '');
    } else if (isOpen) {
      // Reset for new annotation
      setTipoAtto('');
      setNumero('');
      setAnno('');
      setArticoli('');
    }
  }, [originalParsed, isOpen]);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        handleClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen]);

  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Reset form
  const resetForm = () => {
    setTipoAtto('');
    setNumero('');
    setAnno('');
    setArticoli('');
    setError(null);
    setSuccess(false);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  // Extract context window (250 chars before + after)
  const getContextWindow = (text: string, start: number, end: number, windowSize = 250) => {
    const before = text.slice(Math.max(0, start - windowSize), start);
    const selected = text.slice(start, end);
    const after = text.slice(end, Math.min(text.length, end + windowSize));
    return { before, selected, after };
  };

  // Submit handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    if (!tipoAtto) {
      setError('Il tipo di atto è obbligatorio');
      return;
    }

    // Backend requires min 5 characters for selected_text
    if (selectedText.length < 5) {
      setError('Il testo selezionato è troppo corto (min 5 caratteri)');
      return;
    }

    const articlesArray = articoli
      .split(',')
      .map((a: string) => a.trim())
      .filter(Boolean);

    if (articlesArray.length === 0) {
      setError('Inserisci almeno un numero di articolo');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const context = getContextWindow(articleText, startOffset, endOffset, 250);
      const contextWindow = `${context.before}[${context.selected}]${context.after}`;

      // TODO: Implement actual API call when backend is ready
      // For now, simulate the request structure
      const request: NERFeedbackRequest = {
        article_urn: articleUrn,
        user_id: userId,
        selected_text: selectedText,
        start_offset: startOffset,
        end_offset: endOffset,
        context_window: contextWindow,
        feedback_type: originalParsed ? 'correction' : 'annotation',
        original_parsed: originalParsed,
        correct_reference: {
          tipo_atto: tipoAtto,
          numero_atto: numero || undefined,
          anno: anno || undefined,
          articoli: articlesArray,
        },
        confidence_before: confidenceBefore,
        source,
      };

      // Submit NER feedback to backend
      const response = await merltService.submitNERFeedback(request);

      setSuccess(true);

      // Callback after short delay
      setTimeout(() => {
        onSuccess?.(response);
        handleClose();
      }, 1500);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Errore nell\'invio del feedback';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Format article URN for display
  const formatArticleDisplay = (urn: string): string => {
    const parts = urn.split(';');
    const article = parts[parts.length - 1]?.replace('art', 'Art. ') || urn;
    return article;
  };

  // Compute context for display
  const context = getContextWindow(articleText, startOffset, endOffset, 250);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop - semi-transparent to keep article visible */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/20 backdrop-blur-[2px] z-40"
            onClick={handleClose}
          />

          {/* Drawer Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{
              type: 'spring',
              damping: 30,
              stiffness: 300
            }}
            role="dialog"
            aria-label={originalParsed ? 'Correggi citazione' : 'Annota citazione'}
            className={cn(
              "fixed right-0 top-0 bottom-0 z-50",
              "w-full sm:w-[450px] md:w-[500px]",
              "bg-white dark:bg-slate-900",
              "shadow-2xl",
              "flex flex-col"
            )}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 shrink-0">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                  <Save size={20} className="text-blue-600 dark:text-blue-400" aria-hidden="true" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                    {originalParsed ? 'Correggi Citazione' : 'Annota Citazione'}
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {formatArticleDisplay(articleUrn)}
                  </p>
                </div>
              </div>

              <button
                onClick={handleClose}
                className="p-2 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                title="Chiudi pannello (Esc)"
                aria-label="Chiudi pannello"
              >
                <X size={20} className="text-slate-500 dark:text-slate-400 group-hover:text-slate-700 dark:group-hover:text-slate-200" aria-hidden="true" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {success ? (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <CheckCircle2 size={28} aria-hidden="true" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2" role="status">
                    Feedback inviato!
                  </h3>
                  <p className="text-slate-500 dark:text-slate-400">
                    Grazie per aver migliorato il sistema di riconoscimento.
                  </p>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-5">
                  {/* Info banner */}
                  <div className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800/30 rounded-lg">
                    <AlertCircle size={18} className="text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" aria-hidden="true" />
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      Questo feedback migliora il riconoscimento automatico delle citazioni normative.
                    </p>
                  </div>

                  {/* Selected text */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Testo selezionato
                    </label>
                    <div className="p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/30 rounded-lg">
                      <p className="text-sm font-medium text-amber-900 dark:text-amber-100">
                        "{selectedText}"
                      </p>
                    </div>
                  </div>

                  {/* Original parsed (if from preview) */}
                  {originalParsed && (
                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Rilevato automaticamente
                      </label>
                      <div className="p-3 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg">
                        <p className="text-sm text-slate-900 dark:text-white">
                          {originalParsed.actType || 'N/A'} {originalParsed.actNumber || ''} {originalParsed.date || ''}
                          {originalParsed.articles && originalParsed.articles.length > 0 && (
                            <span> - Art. {originalParsed.articles.join(', ')}</span>
                          )}
                        </p>
                        {confidenceBefore && (
                          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            Confidence: {Math.round(confidenceBefore * 100)}%
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                  {!originalParsed && (
                    <div className="p-3 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg">
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        Nessun riferimento trovato automaticamente
                      </p>
                    </div>
                  )}

                  {/* Correct reference form */}
                  <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-white mb-4">
                      Riferimento corretto
                    </h3>

                    {/* Tipo atto */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Tipo atto *
                      </label>
                      <select
                        value={tipoAtto}
                        onChange={(e) => setTipoAtto(e.target.value)}
                        className={cn(
                          'w-full px-4 py-3 rounded-xl border transition-all',
                          'bg-slate-50 dark:bg-slate-800',
                          'border-slate-200 dark:border-slate-700',
                          'text-slate-900 dark:text-white',
                          'focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none'
                        )}
                      >
                        <option value="">-- Seleziona tipo --</option>
                        {TIPO_ATTO_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Numero */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Numero
                      </label>
                      <input
                        type="text"
                        value={numero}
                        onChange={(e) => setNumero(e.target.value)}
                        placeholder="Es. 241"
                        className={cn(
                          'w-full px-4 py-3 rounded-xl border transition-all',
                          'bg-slate-50 dark:bg-slate-800',
                          'border-slate-200 dark:border-slate-700',
                          'text-slate-900 dark:text-white placeholder-slate-400',
                          'focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none'
                        )}
                      />
                    </div>

                    {/* Anno */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Anno
                      </label>
                      <input
                        type="text"
                        value={anno}
                        onChange={(e) => setAnno(e.target.value)}
                        placeholder="Es. 1990"
                        maxLength={4}
                        className={cn(
                          'w-full px-4 py-3 rounded-xl border transition-all',
                          'bg-slate-50 dark:bg-slate-800',
                          'border-slate-200 dark:border-slate-700',
                          'text-slate-900 dark:text-white placeholder-slate-400',
                          'focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none'
                        )}
                      />
                    </div>

                    {/* Articoli */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Articoli *
                      </label>
                      <input
                        type="text"
                        value={articoli}
                        onChange={(e) => setArticoli(e.target.value)}
                        placeholder="Es. 3, 4, 5"
                        className={cn(
                          'w-full px-4 py-3 rounded-xl border transition-all',
                          'bg-slate-50 dark:bg-slate-800',
                          'border-slate-200 dark:border-slate-700',
                          'text-slate-900 dark:text-white placeholder-slate-400',
                          'focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none'
                        )}
                      />
                      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                        Separa con virgola se più articoli
                      </p>
                    </div>
                  </div>

                  {/* Context window */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Contesto
                    </label>
                    <div className="p-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg">
                      <p className="text-xs text-slate-600 dark:text-slate-400 font-mono leading-relaxed">
                        {context.before}
                        <span className="bg-amber-200 dark:bg-amber-900/50 px-1 rounded">
                          [{context.selected}]
                        </span>
                        {context.after}
                      </p>
                    </div>
                  </div>

                  {/* Error */}
                  {error && (
                    <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400" role="alert">
                      <AlertCircle size={16} className="flex-shrink-0 mt-0.5" aria-hidden="true" />
                      {error}
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex justify-end gap-3 pt-2">
                    <button
                      type="button"
                      onClick={handleClose}
                      className="px-4 py-2.5 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    >
                      Annulla
                    </button>
                    <button
                      type="submit"
                      disabled={isSubmitting || !tipoAtto || !articoli}
                      className={cn(
                        'flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all',
                        'bg-blue-600 hover:bg-blue-700 text-white',
                        'disabled:opacity-50 disabled:cursor-not-allowed',
                        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2'
                      )}
                    >
                      {isSubmitting ? (
                        <>
                          <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                          <span>Invio...</span>
                          <span className="sr-only">Invio in corso</span>
                        </>
                      ) : (
                        <>
                          <Save size={18} aria-hidden="true" />
                          Salva Correzione
                        </>
                      )}
                    </button>
                  </div>
                </form>
              )}
            </div>

            {/* Footer Hint */}
            <div className="px-6 py-3 border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 shrink-0">
              <p className="text-xs text-slate-500 dark:text-slate-400">
                <kbd className="px-2 py-1 bg-white dark:bg-slate-900 rounded border border-slate-300 dark:border-slate-700 text-xs">
                  Esc
                </kbd>
                {' '}per chiudere
              </p>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
