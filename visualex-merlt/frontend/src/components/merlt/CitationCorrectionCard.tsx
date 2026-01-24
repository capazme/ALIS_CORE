/**
 * CitationCorrectionCard
 * ======================
 *
 * Floating DRAGGABLE card per correggere/annotare citazioni normative.
 *
 * Design principles:
 * - Compatta (~320px) e trascinabile liberamente
 * - NON blocca l'articolo - resta scrollabile e visibile
 * - NO backdrop oscurante
 * - Form essenziale con smart defaults
 * - Drag handle visibile per spostare la card
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence, useDragControls } from 'framer-motion';
import { X, ChevronDown, Send, Loader2, CheckCircle2, Sparkles, GripVertical } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { merltService } from '../../../services/merltService';
import type { NERFeedbackRequest, NERFeedbackResponse, ParsedCitationData } from '../../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface CitationCorrectionCardProps {
  isOpen: boolean;
  onClose: () => void;
  // Position - dove posizionare la card
  anchorPosition: { top: number; left: number };
  containerRef: React.RefObject<HTMLElement | null>;
  // Dati selezione
  selectedText: string;
  articleUrn: string;
  // Parsing originale (se da hover preview)
  originalParsed?: ParsedCitationData;
  confidenceBefore?: number;
  // Source
  source: 'citation_preview' | 'selection_popup';
  // User
  userId: string;
  // Context extractor - estrae contesto dal DOM
  getContextWindow: () => string;
  // Callback
  onSuccess?: (response: NERFeedbackResponse) => void;
}

// =============================================================================
// TIPI ATTO - Aligned with citationMatcher.ts SUFFIX_TO_ACT_TYPE
// =============================================================================

// Tipi atto comuni (abbreviati per UI compatta)
const TIPO_ATTO_QUICK = [
  { value: 'codice civile', label: 'C.C.', full: 'Codice Civile' },
  { value: 'codice penale', label: 'C.P.', full: 'Codice Penale' },
  { value: 'codice di procedura civile', label: 'C.P.C.', full: 'Cod. Proc. Civile' },
  { value: 'codice di procedura penale', label: 'C.P.P.', full: 'Cod. Proc. Penale' },
  { value: 'costituzione', label: 'Cost.', full: 'Costituzione' },
  { value: 'legge', label: 'L.', full: 'Legge' },
  { value: 'decreto legislativo', label: 'D.Lgs.', full: 'Decreto Legislativo' },
  { value: 'decreto legge', label: 'D.L.', full: 'Decreto Legge' },
];

// Tutti i tipi atto organizzati per categoria
const TIPO_ATTO_ALL = [
  // --- Quick types ---
  ...TIPO_ATTO_QUICK,

  // --- Altri codici ---
  { value: 'preleggi', label: 'Prel.', full: 'Preleggi (Disp. Prel.)' },
  { value: 'codice della strada', label: 'C.d.S.', full: 'Codice della Strada' },
  { value: 'codice della navigazione', label: 'C.N.', full: 'Codice della Navigazione' },
  { value: 'codice dei contratti pubblici', label: 'C.C.P.', full: 'Codice Contratti Pubblici' },
  { value: 'codice del consumo', label: 'C.d.C.', full: 'Codice del Consumo' },
  { value: "codice dell'amministrazione digitale", label: 'CAD', full: 'Cod. Amm. Digitale' },
  { value: "codice della crisi d'impresa e dell'insolvenza", label: 'CCI', full: 'Cod. Crisi Impresa' },
  { value: 'codice dei beni culturali e del paesaggio', label: 'C.B.C.', full: 'Cod. Beni Culturali' },
  { value: 'codice delle comunicazioni elettroniche', label: 'C.C.E.', full: 'Cod. Comunicazioni Elettr.' },
  { value: 'codice delle assicurazioni private', label: 'C.A.P.', full: 'Cod. Assicurazioni' },
  { value: 'codice antimafia', label: 'C.A.M.', full: 'Codice Antimafia' },
  { value: 'norme in materia ambientale', label: 'T.U.A.', full: 'Testo Unico Ambiente' },
  { value: 'testo unico bancario', label: 'T.U.B.', full: 'Testo Unico Bancario' },
  { value: 'testo unico finanza', label: 'T.U.F.', full: 'Testo Unico Finanza' },
  { value: 'testo unico edilizia', label: 'T.U.E.', full: 'Testo Unico Edilizia' },
  { value: 'codice del terzo settore', label: 'C.T.S.', full: 'Codice Terzo Settore' },
  { value: 'codice del turismo', label: 'C.T.U.', full: 'Codice del Turismo' },
  { value: 'codice proprietà industriale', label: 'C.P.I.', full: 'Cod. Proprietà Industriale' },

  // --- Disposizioni attuative ---
  { value: 'disposizioni attuative codice civile', label: 'Disp. Att. C.C.', full: 'Disp. Att. Cod. Civile' },
  { value: 'disposizioni attuative codice procedura civile', label: 'Disp. Att. C.P.C.', full: 'Disp. Att. Cod. Proc. Civ.' },
  { value: 'disposizioni attuative codice penale', label: 'Disp. Att. C.P.', full: 'Disp. Att. Cod. Penale' },
  { value: 'disposizioni attuative codice procedura penale', label: 'Disp. Att. C.P.P.', full: 'Disp. Att. Cod. Proc. Pen.' },

  // --- Atti normativi ---
  { value: 'decreto del presidente della repubblica', label: 'D.P.R.', full: 'D.P.R.' },
  { value: 'decreto ministeriale', label: 'D.M.', full: 'Decreto Ministeriale' },
  { value: 'decreto del presidente del consiglio', label: 'D.P.C.M.', full: 'D.P.C.M.' },
  { value: 'regio decreto', label: 'R.D.', full: 'Regio Decreto' },
  { value: 'regio decreto legge', label: 'R.D.L.', full: 'Regio Decreto Legge' },
  { value: 'legge costituzionale', label: 'L.Cost.', full: 'Legge Costituzionale' },
  { value: 'legge regionale', label: 'L.R.', full: 'Legge Regionale' },

  // --- Normativa UE ---
  { value: 'regolamento ue', label: 'Reg. UE', full: 'Regolamento UE' },
  { value: 'direttiva ue', label: 'Dir. UE', full: 'Direttiva UE' },
  { value: 'decisione ue', label: 'Dec. UE', full: 'Decisione UE' },

  // --- Convenzioni internazionali ---
  { value: 'convenzione europea diritti uomo', label: 'CEDU', full: 'Conv. Europea Diritti Uomo' },
  { value: 'trattato sul funzionamento ue', label: 'TFUE', full: 'Trattato Funz. UE' },
];

// =============================================================================
// COMPONENT
// =============================================================================

export function CitationCorrectionCard({
  isOpen,
  onClose,
  anchorPosition,
  containerRef,
  selectedText,
  articleUrn,
  originalParsed,
  confidenceBefore,
  source,
  userId,
  getContextWindow,
  onSuccess,
}: CitationCorrectionCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const dragControls = useDragControls();

  // Form state
  const [tipoAtto, setTipoAtto] = useState('');
  const [showAllTypes, setShowAllTypes] = useState(false);
  const [numero, setNumero] = useState('');
  const [anno, setAnno] = useState('');
  const [articoli, setArticoli] = useState('');

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // Position state - viewport coordinates for fixed positioning
  const [initialPosition, setInitialPosition] = useState({ top: 0, left: 0 });

  // Calculate smart initial position in VIEWPORT coordinates
  useEffect(() => {
    if (!isOpen || !containerRef.current) return;

    const container = containerRef.current;
    const containerRect = container.getBoundingClientRect();
    const cardWidth = 320;
    const cardHeight = 400; // approximate
    const padding = 16;

    // Convert container-relative position to viewport coordinates
    let top = containerRect.top + anchorPosition.top + 20; // Below selection
    let left = containerRect.left + anchorPosition.left - cardWidth / 2; // Center on selection

    // Adjust horizontal to stay within viewport initially
    if (left + cardWidth > window.innerWidth - padding) {
      left = window.innerWidth - cardWidth - padding;
    }
    if (left < padding) {
      left = padding;
    }

    // Adjust vertical - prefer below, but flip if not enough space
    const spaceBelow = window.innerHeight - top;
    if (spaceBelow < cardHeight && top > cardHeight + 50) {
      top = containerRect.top + anchorPosition.top - cardHeight - 10; // Above selection
    }

    // Ensure top is not negative
    if (top < padding) {
      top = padding;
    }

    setInitialPosition({ top, left });
  }, [isOpen, anchorPosition, containerRef]);

  // Pre-fill from original parsed data
  useEffect(() => {
    if (originalParsed && isOpen) {
      setTipoAtto(originalParsed.actType || '');
      setNumero(originalParsed.actNumber || '');
      setAnno(originalParsed.date || '');
      setArticoli(originalParsed.articles?.join(', ') || '');
    } else if (isOpen) {
      // Try to extract article numbers from selected text
      const artMatch = selectedText.match(/art(?:icol[oi])?\.?\s*(\d+(?:\s*(?:,|e)\s*\d+)*)/i);
      if (artMatch) {
        setArticoli(artMatch[1].replace(/\s*e\s*/g, ', '));
      }
      setTipoAtto('');
      setNumero('');
      setAnno('');
    }
    setError(null);
    setSuccess(false);
  }, [originalParsed, isOpen, selectedText]);

  // Close on click outside
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (cardRef.current && !cardRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    // Delay to avoid immediate close
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  const handleSubmit = useCallback(async () => {
    // Validation
    if (!tipoAtto) {
      setError('Seleziona il tipo di atto');
      return;
    }

    // Backend requires min 5 characters for selected_text
    if (selectedText.length < 5) {
      setError('Il testo selezionato è troppo corto (min 5 caratteri)');
      return;
    }

    const articlesArray = articoli
      .split(/[,\s]+/)
      .map(a => a.trim())
      .filter(Boolean);

    if (articlesArray.length === 0) {
      setError('Inserisci almeno un articolo');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const contextWindow = getContextWindow();

      const request: NERFeedbackRequest = {
        article_urn: articleUrn,
        user_id: userId,
        selected_text: selectedText,
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

      const response = await merltService.submitNERFeedback(request);
      setSuccess(true);

      setTimeout(() => {
        onSuccess?.(response);
        onClose();
      }, 1200);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Errore';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  }, [
    tipoAtto, numero, anno, articoli, articleUrn, userId, selectedText,
    originalParsed, confidenceBefore, source,
    getContextWindow, onSuccess, onClose
  ]);

  // Quick type select
  const handleQuickType = (value: string) => {
    setTipoAtto(value);
    setShowAllTypes(false);
  };

  const visibleTypes = showAllTypes ? TIPO_ATTO_ALL : TIPO_ATTO_QUICK;

  // Start drag from header
  const startDrag = (event: React.PointerEvent) => {
    dragControls.start(event);
  };

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <motion.div
          ref={cardRef}
          initial={{ opacity: 0, scale: 0.95, y: -10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: -10 }}
          transition={{ duration: 0.15, ease: 'easeOut' }}
          drag
          dragControls={dragControls}
          dragListener={false}
          dragElastic={0}
          dragMomentum={false}
          onDragStart={() => setIsDragging(true)}
          onDragEnd={() => setIsDragging(false)}
          whileDrag={{ scale: 1.02, boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)' }}
          className={cn(
            "fixed z-[200]",
            isDragging && "cursor-grabbing"
          )}
          style={{
            top: initialPosition.top,
            left: initialPosition.left,
            width: 320,
          }}
        >
          <div className={cn(
            "bg-white dark:bg-slate-900 rounded-xl",
            "shadow-xl border border-slate-200 dark:border-slate-700",
            "overflow-hidden",
            isDragging && "ring-2 ring-blue-500/50"
          )}>
            {/* Header - Draggable Handle */}
            <div
              onPointerDown={startDrag}
              className={cn(
                "flex items-center justify-between px-3 py-2",
                "bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30",
                "border-b border-slate-200 dark:border-slate-700",
                "cursor-grab active:cursor-grabbing",
                "select-none touch-none"
              )}
            >
              <div className="flex items-center gap-2">
                {/* Drag handle indicator */}
                <GripVertical size={14} className="text-slate-400 dark:text-slate-500" />
                <Sparkles size={14} className="text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium text-slate-900 dark:text-white">
                  {originalParsed ? 'Correggi' : 'Annota'} citazione
                </span>
              </div>
              <button
                onClick={onClose}
                onPointerDown={(e) => e.stopPropagation()}
                className="p-1 hover:bg-slate-200/50 dark:hover:bg-slate-700/50 rounded transition-colors"
              >
                <X size={16} className="text-slate-500" />
              </button>
            </div>

            {success ? (
              /* Success state */
              <div className="p-6 text-center">
                <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 rounded-full flex items-center justify-center mx-auto mb-3">
                  <CheckCircle2 size={24} />
                </div>
                <p className="text-sm font-medium text-slate-900 dark:text-white">
                  Feedback salvato!
                </p>
              </div>
            ) : (
              /* Form */
              <div className="p-3 space-y-3">
                {/* Selected text preview */}
                <div className="px-2 py-1.5 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800/40">
                  <p className="text-xs text-amber-800 dark:text-amber-200 font-medium truncate">
                    "{selectedText}"
                  </p>
                </div>

                {/* Tipo atto - Quick chips */}
                <div>
                  <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1.5">
                    Tipo atto
                  </label>
                  <div className="flex flex-wrap gap-1.5">
                    {visibleTypes.map(({ value, label }) => (
                      <button
                        key={value}
                        type="button"
                        onClick={() => handleQuickType(value)}
                        className={cn(
                          "px-2 py-1 text-xs rounded-lg border transition-all",
                          tipoAtto === value
                            ? "bg-blue-600 text-white border-blue-600"
                            : "bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-blue-400"
                        )}
                      >
                        {label}
                      </button>
                    ))}
                    {!showAllTypes && (
                      <button
                        type="button"
                        onClick={() => setShowAllTypes(true)}
                        className="px-2 py-1 text-xs text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 flex items-center gap-0.5"
                      >
                        Altri <ChevronDown size={12} />
                      </button>
                    )}
                  </div>
                </div>

                {/* Numero + Anno row */}
                <div className="flex gap-2">
                  <div className="flex-1">
                    <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                      Numero
                    </label>
                    <input
                      type="text"
                      value={numero}
                      onChange={(e) => setNumero(e.target.value)}
                      placeholder="241"
                      className={cn(
                        "w-full px-2.5 py-1.5 text-sm rounded-lg border",
                        "bg-slate-50 dark:bg-slate-800",
                        "border-slate-200 dark:border-slate-700",
                        "text-slate-900 dark:text-white placeholder-slate-400",
                        "focus:ring-1 focus:ring-blue-500 focus:border-blue-500 outline-none"
                      )}
                    />
                  </div>
                  <div className="w-20">
                    <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                      Anno
                    </label>
                    <input
                      type="text"
                      value={anno}
                      onChange={(e) => setAnno(e.target.value)}
                      placeholder="1990"
                      maxLength={4}
                      className={cn(
                        "w-full px-2.5 py-1.5 text-sm rounded-lg border",
                        "bg-slate-50 dark:bg-slate-800",
                        "border-slate-200 dark:border-slate-700",
                        "text-slate-900 dark:text-white placeholder-slate-400",
                        "focus:ring-1 focus:ring-blue-500 focus:border-blue-500 outline-none"
                      )}
                    />
                  </div>
                </div>

                {/* Articoli */}
                <div>
                  <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                    Articoli
                  </label>
                  <input
                    type="text"
                    value={articoli}
                    onChange={(e) => setArticoli(e.target.value)}
                    placeholder="3, 4, 5"
                    className={cn(
                      "w-full px-2.5 py-1.5 text-sm rounded-lg border",
                      "bg-slate-50 dark:bg-slate-800",
                      "border-slate-200 dark:border-slate-700",
                      "text-slate-900 dark:text-white placeholder-slate-400",
                      "focus:ring-1 focus:ring-blue-500 focus:border-blue-500 outline-none"
                    )}
                  />
                </div>

                {/* Error */}
                {error && (
                  <p className="text-xs text-red-600 dark:text-red-400">
                    {error}
                  </p>
                )}

                {/* Submit */}
                <button
                  onClick={handleSubmit}
                  disabled={isSubmitting || !tipoAtto || !articoli}
                  className={cn(
                    "w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                    "bg-blue-600 hover:bg-blue-700 text-white",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      Invio...
                    </>
                  ) : (
                    <>
                      <Send size={14} />
                      Invia feedback
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}
