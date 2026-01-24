/**
 * ProposeRelationDrawer
 * =====================
 *
 * Drawer laterale per proporre una nuova relazione tra entità nel Knowledge Graph MERL-T.
 * Sostituisce ProposeRelationModal per mantenere visibile l'articolo durante la proposta.
 * Include autocomplete per cercare entità esistenti (R3 integration).
 * Integra citationParser per risolvere norme in linguaggio naturale (R5 integration).
 */

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Link2, Loader2, CheckCircle2, AlertCircle, ArrowRight, Sparkles, Search, Plus, Scale, Copy } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { merltService } from '../../../services/merltService';
import { RELATION_TYPE_OPTIONS, groupByCategory } from '../../../constants/merltTypes';
import { parseLegalCitation, isSearchReady, formatParsedCitation, type ParsedCitation } from '../../../utils/citationParser';
import type { RelationType, PendingEntity, NormResolveResponse, ValidationStatus, EntityType, RelationDuplicateCandidate } from '../../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

interface EntitySearchResult {
  id: string;
  nome: string;
  tipo: EntityType;
  approval_score: number;
  validation_status: ValidationStatus;
}

interface ProposeRelationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  // Context
  articleUrn: string;
  userId: string;
  // Pre-loaded entities (fallback if search fails)
  availableEntities?: PendingEntity[];
}

// Grouped relation type options (from shared constants)
const groupedRelationOptions = groupByCategory(RELATION_TYPE_OPTIONS);

// =============================================================================
// DEBOUNCE HOOK
// =============================================================================

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// =============================================================================
// COMPONENT
// =============================================================================

export function ProposeRelationDrawer({
  isOpen,
  onClose,
  onSuccess,
  articleUrn,
  userId,
  availableEntities = [],
}: ProposeRelationDrawerProps) {
  // Form state
  const [tipoRelazione, setTipoRelazione] = useState<RelationType>('DISCIPLINA');
  const [descrizione, setDescrizione] = useState('');
  const [certezza, setCertezza] = useState(0.7);

  // Target entity state (autocomplete)
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedEntity, setSelectedEntity] = useState<EntitySearchResult | null>(null);
  const [isNewEntity, setIsNewEntity] = useState(false);
  const [searchResults, setSearchResults] = useState<EntitySearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  // Norm resolution state (R5)
  const [parsedCitation, setParsedCitation] = useState<ParsedCitation | null>(null);
  const [isNormSelected, setIsNormSelected] = useState(false);
  const [resolvedNorm, setResolvedNorm] = useState<NormResolveResponse | null>(null);
  const [isResolvingNorm, setIsResolvingNorm] = useState(false);

  // Duplicate detection state
  const [duplicatesFound, setDuplicatesFound] = useState(false);
  const [duplicates, setDuplicates] = useState<RelationDuplicateCandidate[]>([]);
  const [exactDuplicate, setExactDuplicate] = useState<RelationDuplicateCandidate | null>(null);
  const [isCheckingDuplicates, setIsCheckingDuplicates] = useState(false);
  const [acknowledgedDuplicate, setAcknowledgedDuplicate] = useState(false);

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Refs
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Debounced search query
  const debouncedQuery = useDebounce(searchQuery, 300);

  // Parse citation when query changes (R5)
  useEffect(() => {
    if (debouncedQuery.length >= 2) {
      const parsed = parseLegalCitation(debouncedQuery);
      setParsedCitation(parsed);
    } else {
      setParsedCitation(null);
    }
  }, [debouncedQuery]);

  // Search entities when query changes
  useEffect(() => {
    const searchEntities = async () => {
      if (debouncedQuery.length < 2) {
        setSearchResults([]);
        return;
      }

      setIsSearching(true);
      try {
        // Try backend search first
        if (merltService.searchEntities) {
          const results = await merltService.searchEntities(debouncedQuery, articleUrn);
          setSearchResults(results as EntitySearchResult[]);
        } else {
          // Fallback to local filtering of available entities
          const filtered = availableEntities
            .filter(e =>
              e.nome.toLowerCase().includes(debouncedQuery.toLowerCase()) ||
              e.descrizione?.toLowerCase().includes(debouncedQuery.toLowerCase())
            )
            .slice(0, 10)
            .map(e => ({
              id: e.id,
              nome: e.nome,
              tipo: e.tipo,
              approval_score: e.approval_score || 0,
              validation_status: e.validation_status || 'pending',
            }));
          setSearchResults(filtered);
        }
      } catch {
        // Fallback to local filtering
        const filtered = availableEntities
          .filter(e =>
            e.nome.toLowerCase().includes(debouncedQuery.toLowerCase())
          )
          .slice(0, 10)
          .map(e => ({
            id: e.id,
            nome: e.nome,
            tipo: e.tipo,
            approval_score: e.approval_score || 0,
            validation_status: e.validation_status || 'pending',
          }));
        setSearchResults(filtered);
      } finally {
        setIsSearching(false);
      }
    };

    searchEntities();
  }, [debouncedQuery, articleUrn, availableEntities]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        if (showDropdown) {
          setShowDropdown(false);
        } else {
          handleClose();
        }
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, showDropdown]);

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
    setTipoRelazione('DISCIPLINA');
    setDescrizione('');
    setCertezza(0.7);
    setSearchQuery('');
    setSelectedEntity(null);
    setIsNewEntity(false);
    setSearchResults([]);
    setError(null);
    setSuccess(false);
    // Reset norm state (R5)
    setParsedCitation(null);
    setIsNormSelected(false);
    setResolvedNorm(null);
    setIsResolvingNorm(false);
    // Reset duplicate state
    setDuplicatesFound(false);
    setDuplicates([]);
    setExactDuplicate(null);
    setIsCheckingDuplicates(false);
    setAcknowledgedDuplicate(false);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  // Select entity from dropdown
  const handleSelectEntity = (entity: EntitySearchResult) => {
    setSelectedEntity(entity);
    setSearchQuery(entity.nome);
    setIsNewEntity(false);
    setShowDropdown(false);
  };

  // Create new entity option
  const handleCreateNew = () => {
    setSelectedEntity(null);
    setIsNewEntity(true);
    setIsNormSelected(false);
    setResolvedNorm(null);
    setShowDropdown(false);
  };

  // Select norm from citation parsing (R5)
  const handleSelectNorm = async () => {
    if (!parsedCitation || !isSearchReady(parsedCitation)) return;

    setIsResolvingNorm(true);
    setError(null);

    try {
      const result = await merltService.resolveNorm({
        act_type: parsedCitation.act_type || '',
        article: parsedCitation.article || '',
        act_number: parsedCitation.act_number,
        date: parsedCitation.date,
        source_article_urn: articleUrn,
        user_id: userId,
      });

      if (result.resolved) {
        setResolvedNorm(result);
        setIsNormSelected(true);
        setSelectedEntity(null);
        setIsNewEntity(false);
        setSearchQuery(result.display_label);
        setShowDropdown(false);
      } else {
        setError(result.error_message || 'Errore nella risoluzione della norma');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Errore nella risoluzione della norma';
      setError(errorMessage);
    } finally {
      setIsResolvingNorm(false);
    }
  };

  // Get target entity ID based on current selection
  const getTargetEntityId = (): string | null => {
    const targetName = resolvedNorm?.display_label || selectedEntity?.nome || (isNewEntity ? searchQuery.trim() : '');

    if (isNormSelected && resolvedNorm) {
      return resolvedNorm.entity_id;
    } else if (selectedEntity) {
      return selectedEntity.id;
    } else if (isNewEntity && targetName) {
      return targetName;
    }
    return null;
  };

  // Check for duplicates before submitting
  const checkDuplicates = async (): Promise<boolean> => {
    const targetEntityId = getTargetEntityId();
    if (!targetEntityId) return false;

    setIsCheckingDuplicates(true);
    setError(null);

    try {
      const result = await merltService.checkRelationDuplicate({
        source_entity_id: articleUrn,
        target_entity_id: targetEntityId,
        relation_type: tipoRelazione,
      });

      if (result.has_duplicates) {
        setDuplicatesFound(true);
        setDuplicates(result.duplicates || []);
        setExactDuplicate(result.exact_match || null);
        return true; // Found duplicates
      }
      return false; // No duplicates
    } catch {
      // If check fails, allow proceeding (don't block on check errors)
      return false;
    } finally {
      setIsCheckingDuplicates(false);
    }
  };

  // Handle proceed after acknowledging duplicates
  const handleProceedAnyway = () => {
    setAcknowledgedDuplicate(true);
    setDuplicatesFound(false);
    // Continue with submit, skipping duplicate check
    doSubmit(true);
  };

  // Actual submission logic with integrated deduplication
  const doSubmit = async (skipDuplicateCheck: boolean = false) => {
    const targetEntityId = getTargetEntityId();

    if (!targetEntityId) {
      setError('Seleziona o inserisci l\'entità di destinazione');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const result = await merltService.proposeRelation({
        tipo_relazione: tipoRelazione,
        source_urn: articleUrn,
        target_entity_id: targetEntityId,
        article_urn: articleUrn,
        descrizione: descrizione.trim(),
        certezza,
        user_id: userId,
        skip_duplicate_check: skipDuplicateCheck,
      });

      // Handle deduplication response
      if (!result.success && result.duplicate_action_required) {
        // Exact match found - show duplicates and block
        setDuplicatesFound(true);
        setDuplicates(result.duplicates || []);
        const exact = result.duplicates?.find(d => d.similarity_score === 1.0);
        setExactDuplicate(exact || null);
        return; // Don't proceed
      }

      // Success (may have fuzzy duplicates as info)
      if (result.has_duplicates && result.duplicates?.length > 0) {
        // Inform user about similar relations but proceed
        console.info('Relazioni simili trovate:', result.duplicates);
      }

      setSuccess(true);

      // Callback after short delay
      setTimeout(() => {
        onSuccess?.();
        handleClose();
      }, 1500);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Errore nella proposta della relazione';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Submit handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation - check for norm, entity, or new entity
    const targetName = resolvedNorm?.display_label || selectedEntity?.nome || (isNewEntity ? searchQuery.trim() : '');
    if (!targetName && !isNormSelected) {
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

    // If already acknowledged duplicate, skip deduplication check
    await doSubmit(acknowledgedDuplicate);
  };

  // Format article URN for display
  const formatArticleDisplay = (urn: string): string => {
    const parts = urn.split(';');
    const article = parts[parts.length - 1]?.replace('art', 'Art. ') || urn;
    return article;
  };

  const selectedRelationType = RELATION_TYPE_OPTIONS.find(r => r.value === tipoRelazione);
  const targetDisplay = isNormSelected && resolvedNorm
    ? resolvedNorm.display_label
    : selectedEntity?.nome || (isNewEntity ? searchQuery : 'Seleziona...');

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
                <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                  <Link2 size={20} className="text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                    Proponi Nuova Relazione
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {formatArticleDisplay(articleUrn)}
                  </p>
                </div>
              </div>

              <button
                onClick={handleClose}
                className="p-2 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors group"
                title="Chiudi pannello (Esc)"
              >
                <X size={20} className="text-slate-500 dark:text-slate-400 group-hover:text-slate-700 dark:group-hover:text-slate-200" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
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
                        {formatArticleDisplay(articleUrn)}
                      </p>
                    </div>
                    <div className="flex flex-col items-center">
                      <ArrowRight size={20} className="text-slate-400" />
                      <span className="text-xs text-slate-500 mt-1">
                        {selectedRelationType?.label || 'Relazione'}
                      </span>
                    </div>
                    <div className="text-center">
                      <div className={cn(
                        "px-3 py-1.5 rounded-lg text-sm font-medium",
                        isNormSelected && resolvedNorm
                          ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                          : selectedEntity
                          ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"
                          : isNewEntity
                          ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                          : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400"
                      )}>
                        {targetDisplay}
                      </div>
                      {isNormSelected && resolvedNorm && (
                        <p className="text-xs text-amber-500 mt-1 flex items-center justify-center gap-1">
                          <Scale size={12} />
                          {resolvedNorm.exists_in_graph ? 'Nel grafo' : 'Pending'}
                        </p>
                      )}
                      {!isNormSelected && isNewEntity && (
                        <p className="text-xs text-amber-500 mt-1">Nuova entità</p>
                      )}
                      {!isNormSelected && selectedEntity?.validation_status === 'approved' && (
                        <p className="text-xs text-emerald-500 mt-1">Validata</p>
                      )}
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
                      {Object.entries(groupedRelationOptions).map(([category, options]) => (
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

                  {/* Target Entity - Autocomplete */}
                  <div className="relative">
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Entità di destinazione *
                    </label>
                    <div className="relative">
                      <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                      <input
                        ref={inputRef}
                        type="text"
                        value={searchQuery}
                        onChange={(e) => {
                          setSearchQuery(e.target.value);
                          setSelectedEntity(null);
                          setIsNewEntity(false);
                          setShowDropdown(true);
                        }}
                        onFocus={() => setShowDropdown(true)}
                        placeholder="Cerca entità esistente o inserisci nuova..."
                        className={cn(
                          'w-full pl-11 pr-4 py-3 rounded-xl border transition-all',
                          'bg-slate-50 dark:bg-slate-800',
                          'border-slate-200 dark:border-slate-700',
                          'text-slate-900 dark:text-white placeholder-slate-400',
                          'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none',
                          selectedEntity && 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20'
                        )}
                      />
                      {isSearching && (
                        <Loader2 size={18} className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 animate-spin" />
                      )}
                    </div>

                    {/* Dropdown */}
                    <AnimatePresence>
                      {showDropdown && (searchResults.length > 0 || searchQuery.length >= 2) && (
                        <motion.div
                          ref={dropdownRef}
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          className={cn(
                            "absolute z-10 w-full mt-2 py-2 rounded-xl shadow-lg",
                            "bg-white dark:bg-slate-800",
                            "border border-slate-200 dark:border-slate-700",
                            "max-h-[250px] overflow-y-auto"
                          )}
                        >
                          {searchResults.map((entity) => (
                            <button
                              key={entity.id}
                              type="button"
                              onClick={() => handleSelectEntity(entity)}
                              className={cn(
                                "w-full px-4 py-2.5 text-left hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors",
                                "flex items-center justify-between gap-2"
                              )}
                            >
                              <div>
                                <span className="font-medium text-slate-900 dark:text-white">
                                  {entity.nome}
                                </span>
                                <span className="ml-2 text-xs px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400">
                                  {entity.tipo}
                                </span>
                              </div>
                              {entity.validation_status === 'approved' && (
                                <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0" />
                              )}
                            </button>
                          ))}

                          {/* Norm detected option (R5) */}
                          {parsedCitation && isSearchReady(parsedCitation) && (
                            <div className="border-t border-slate-200 dark:border-slate-700">
                              <div className="px-4 py-1.5 text-xs font-medium text-amber-600 dark:text-amber-400 uppercase tracking-wide bg-amber-50/50 dark:bg-amber-900/10">
                                Norma riconosciuta
                              </div>
                              <button
                                type="button"
                                onClick={handleSelectNorm}
                                disabled={isResolvingNorm}
                                className={cn(
                                  "w-full px-4 py-2.5 text-left",
                                  "bg-amber-50 dark:bg-amber-900/20 hover:bg-amber-100 dark:hover:bg-amber-900/30",
                                  "text-amber-700 dark:text-amber-300",
                                  "flex items-center gap-3",
                                  "disabled:opacity-50 disabled:cursor-wait"
                                )}
                              >
                                {isResolvingNorm ? (
                                  <Loader2 size={16} className="animate-spin" />
                                ) : (
                                  <Scale size={16} />
                                )}
                                <span className="flex-1 font-medium">
                                  {formatParsedCitation(parsedCitation)}
                                </span>
                                {parsedCitation.fromAlias && (
                                  <span className="text-xs px-2 py-0.5 rounded bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200">
                                    alias
                                  </span>
                                )}
                              </button>
                            </div>
                          )}

                          {/* Create new option */}
                          {searchQuery.length >= 2 && !searchResults.find(e => e.nome.toLowerCase() === searchQuery.toLowerCase()) && !isNormSelected && (
                            <button
                              type="button"
                              onClick={handleCreateNew}
                              className={cn(
                                "w-full px-4 py-2.5 text-left",
                                "bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-700",
                                "text-slate-700 dark:text-slate-300",
                                "flex items-center gap-2",
                                "border-t border-slate-200 dark:border-slate-700"
                              )}
                            >
                              <Plus size={16} />
                              <span>Crea "<strong>{searchQuery}</strong>" come nuova entità</span>
                            </button>
                          )}

                          {searchResults.length === 0 && searchQuery.length >= 2 && !isSearching && (
                            <div className="px-4 py-2 text-sm text-slate-500 dark:text-slate-400">
                              Nessuna entità trovata
                            </div>
                          )}
                        </motion.div>
                      )}
                    </AnimatePresence>
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

                  {/* Duplicate Warning */}
                  {duplicatesFound && (
                    <div className="p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/30 rounded-xl">
                      <div className="flex items-start gap-3 mb-3">
                        <Copy size={18} className="text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                        <div>
                          <h4 className="font-medium text-amber-800 dark:text-amber-300">
                            {exactDuplicate ? 'Relazione già esistente!' : 'Relazioni simili trovate'}
                          </h4>
                          <p className="text-sm text-amber-600 dark:text-amber-400 mt-1">
                            {exactDuplicate
                              ? 'Esiste già una relazione identica nel sistema.'
                              : 'Esistono relazioni simili. Verifica prima di procedere.'}
                          </p>
                        </div>
                      </div>

                      {/* Exact duplicate - blocking */}
                      {exactDuplicate && (
                        <div className="mb-3 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30 rounded-lg">
                          <div className="flex items-center gap-2">
                            <AlertCircle size={16} className="text-red-500 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <span className="font-medium text-red-700 dark:text-red-300 text-sm">
                                {exactDuplicate.source_text}
                              </span>
                              <span className="mx-2 text-red-500">→</span>
                              <span className="text-red-600 dark:text-red-400 text-sm">
                                [{exactDuplicate.relation_type}]
                              </span>
                              <span className="mx-2 text-red-500">→</span>
                              <span className="font-medium text-red-700 dark:text-red-300 text-sm">
                                {exactDuplicate.target_text}
                              </span>
                            </div>
                          </div>
                          <p className="text-xs text-red-500 mt-2">
                            Non puoi creare una relazione identica. Modifica tipo o target.
                          </p>
                        </div>
                      )}

                      {/* Similar duplicates - informational */}
                      {!exactDuplicate && duplicates.length > 0 && (
                        <div className="space-y-2 mb-3 max-h-[150px] overflow-y-auto">
                          {duplicates.map((dup) => (
                            <div
                              key={dup.relation_id}
                              className="flex items-center gap-2 p-2 bg-white dark:bg-slate-800/50 rounded-lg border border-amber-200 dark:border-amber-800/20"
                            >
                              <div className="flex-1 min-w-0 text-sm">
                                <span className="font-medium text-slate-900 dark:text-white">
                                  {dup.source_text}
                                </span>
                                <span className="mx-1 text-slate-400">→</span>
                                <span className="text-amber-600 dark:text-amber-400">
                                  [{dup.relation_type}]
                                </span>
                                <span className="mx-1 text-slate-400">→</span>
                                <span className="font-medium text-slate-900 dark:text-white">
                                  {dup.target_text}
                                </span>
                              </div>
                              <span className={cn(
                                "text-xs px-2 py-0.5 rounded",
                                dup.confidence === 'high' ? 'bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200' :
                                dup.confidence === 'medium' ? 'bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200' :
                                'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
                              )}>
                                {Math.round(dup.similarity_score * 100)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex justify-end gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            setDuplicatesFound(false);
                            setDuplicates([]);
                            setExactDuplicate(null);
                          }}
                          className="px-3 py-1.5 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
                        >
                          Modifica
                        </button>
                        {!exactDuplicate && (
                          <button
                            type="button"
                            onClick={handleProceedAnyway}
                            className="px-3 py-1.5 text-sm bg-amber-600 hover:bg-amber-700 text-white rounded-lg transition-colors"
                          >
                            Procedi comunque
                          </button>
                        )}
                      </div>
                    </div>
                  )}

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
                      disabled={
                        isSubmitting ||
                        isCheckingDuplicates ||
                        (duplicatesFound && !!exactDuplicate) ||
                        (!selectedEntity && !isNewEntity && !isNormSelected) ||
                        !descrizione.trim()
                      }
                      className={cn(
                        'flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all',
                        'bg-primary-600 hover:bg-primary-700 text-white',
                        'disabled:opacity-50 disabled:cursor-not-allowed'
                      )}
                    >
                      {isSubmitting || isCheckingDuplicates ? (
                        <>
                          <Loader2 size={18} className="animate-spin" />
                          {isCheckingDuplicates ? 'Verifico...' : 'Invio...'}
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
