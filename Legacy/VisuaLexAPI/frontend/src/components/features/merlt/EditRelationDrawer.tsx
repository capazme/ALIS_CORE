/**
 * EditRelationDrawer
 * ==================
 *
 * Drawer per modificare ogni campo di una relazione pending.
 * Include autocomplete per cercare entità esistenti di QUALSIASI tipo
 * (non solo norme: sentenze, principi, concetti, etc.).
 * Le modifiche vengono inviate come suggested_edits e applicate
 * quando il consenso viene raggiunto.
 */

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Save,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Edit3,
  Search,
  Plus,
  ArrowRight,
  Scale,
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import { merltService } from '../../../services/merltService';
import { RELATION_TYPE_OPTIONS, ENTITY_TYPE_LABELS, groupByCategory } from '../../../constants/merltTypes';
import { parseLegalCitation, isSearchReady, formatParsedCitation, type ParsedCitation } from '../../../utils/citationParser';
import type { RelationType, PendingRelation, EntityType, NormResolveResponse, ValidationStatus } from '../../../types/merlt';

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

interface EditRelationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  relation: PendingRelation;
  userId: string;
}

// Grouped relation options (from shared constants)
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

export function EditRelationDrawer({
  isOpen,
  onClose,
  onSuccess,
  relation,
  userId,
}: EditRelationDrawerProps) {
  // Form state - initialized with current relation values
  const [relationType, setRelationType] = useState<RelationType>(relation.relation_type);
  const [evidence, setEvidence] = useState(relation.evidence || '');
  const [motivazione, setMotivazione] = useState('');

  // Target entity state (autocomplete) - inizia con il target attuale
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedEntity, setSelectedEntity] = useState<EntitySearchResult | null>(null);
  const [isNewEntity, setIsNewEntity] = useState(false);
  const [newEntityName, setNewEntityName] = useState('');
  const [searchResults, setSearchResults] = useState<EntitySearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  // Original target display
  const [originalTargetDisplay, setOriginalTargetDisplay] = useState(relation.target_urn);

  // Norm resolution state (R5)
  const [parsedCitation, setParsedCitation] = useState<ParsedCitation | null>(null);
  const [isNormSelected, setIsNormSelected] = useState(false);
  const [resolvedNorm, setResolvedNorm] = useState<NormResolveResponse | null>(null);
  const [isResolvingNorm, setIsResolvingNorm] = useState(false);

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Refs
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Debounced search query
  const debouncedQuery = useDebounce(searchQuery, 300);

  // Reset form when relation changes
  useEffect(() => {
    if (relation && isOpen) {
      setRelationType(relation.relation_type);
      setEvidence(relation.evidence || '');
      setMotivazione('');
      setSearchQuery('');
      setSelectedEntity(null);
      setIsNewEntity(false);
      setNewEntityName('');
      setOriginalTargetDisplay(relation.target_urn);
      // Reset norm state
      setParsedCitation(null);
      setIsNormSelected(false);
      setResolvedNorm(null);
      setIsResolvingNorm(false);
      setError(null);
      setSuccess(false);
    }
  }, [relation, isOpen]);

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
        // Cerca qualsiasi tipo di entità (non solo norme)
        if (merltService.searchEntities) {
          const results = await merltService.searchEntities(debouncedQuery);
          setSearchResults(results as EntitySearchResult[]);
        } else {
          setSearchResults([]);
        }
      } catch {
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    };

    searchEntities();
  }, [debouncedQuery]);

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

  // Get current target value (either resolved norm, selected entity ID, new entity name, or original)
  const getCurrentTargetUrn = (): string => {
    if (isNormSelected && resolvedNorm) {
      return resolvedNorm.entity_id;
    }
    if (selectedEntity) {
      return selectedEntity.id;
    }
    if (isNewEntity && newEntityName) {
      return newEntityName; // Sarà creata come nuova entità
    }
    return relation.target_urn;
  };

  // Check if target was modified
  const targetModified = getCurrentTargetUrn() !== relation.target_urn;

  // Check if any field was modified
  const hasChanges =
    relationType !== relation.relation_type ||
    targetModified ||
    evidence !== (relation.evidence || '');

  // Build suggested_edits object with only changed fields
  const buildSuggestedEdits = () => {
    const edits: Record<string, string> = {};
    if (relationType !== relation.relation_type) edits.relation_type = relationType;
    if (targetModified) edits.target_urn = getCurrentTargetUrn();
    if (evidence !== (relation.evidence || '')) edits.evidence = evidence;
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
      await merltService.validateRelation({
        relation_id: relation.id,
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

  // Select entity from dropdown
  const handleSelectEntity = (entity: EntitySearchResult) => {
    setSelectedEntity(entity);
    setSearchQuery(entity.nome);
    setIsNewEntity(false);
    setNewEntityName('');
    setShowDropdown(false);
  };

  // Create new entity option
  const handleCreateNew = () => {
    setSelectedEntity(null);
    setIsNewEntity(true);
    setNewEntityName(searchQuery.trim());
    setShowDropdown(false);
  };

  // Clear target selection
  const handleClearTarget = () => {
    setSelectedEntity(null);
    setIsNewEntity(false);
    setNewEntityName('');
    setSearchQuery('');
    setIsNormSelected(false);
    setResolvedNorm(null);
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
        source_article_urn: relation.source_urn,
        user_id: userId,
      });

      if (result.resolved) {
        setResolvedNorm(result);
        setIsNormSelected(true);
        setSelectedEntity(null);
        setIsNewEntity(false);
        setNewEntityName('');
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


  // Current target display
  const currentTargetDisplay = isNormSelected && resolvedNorm
    ? resolvedNorm.display_label
    : selectedEntity
    ? selectedEntity.nome
    : isNewEntity
    ? newEntityName
    : originalTargetDisplay;

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
            className="fixed right-0 top-0 h-full w-[520px] max-w-full bg-white dark:bg-slate-900 shadow-2xl z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
              <div className="flex items-center gap-2">
                <Edit3 className="w-5 h-5 text-indigo-500" />
                <h2 className="font-semibold text-slate-900 dark:text-slate-100">
                  Modifica Relazione
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
                      <strong>{relation.relation_type}</strong>
                    </p>
                    <p className="text-slate-600 dark:text-slate-400 mt-1 truncate">
                      Source: {relation.source_urn}
                    </p>
                    <p className="text-slate-600 dark:text-slate-400 truncate">
                      Target: {relation.target_urn}
                    </p>
                  </div>

                  {/* Visual representation */}
                  <div className="flex items-center justify-center gap-2 p-3 bg-slate-50 dark:bg-slate-800/50 rounded-xl text-sm">
                    <div className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded text-xs font-medium truncate max-w-[120px]">
                      Source
                    </div>
                    <ArrowRight size={16} className="text-slate-400 flex-shrink-0" />
                    <div className={cn(
                      "px-2 py-1 rounded text-xs font-medium",
                      targetModified
                        ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                        : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400"
                    )}>
                      {selectedEntity?.tipo && (
                        <span className="opacity-70">{ENTITY_TYPE_LABELS[selectedEntity.tipo]} · </span>
                      )}
                      {isNewEntity && <span className="opacity-70">Nuovo · </span>}
                      <span className="truncate max-w-[100px] inline-block align-bottom">
                        {currentTargetDisplay}
                      </span>
                    </div>
                  </div>

                  {/* Relation Type */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                      Tipo Relazione *
                    </label>
                    <select
                      value={relationType}
                      onChange={(e) => setRelationType(e.target.value as RelationType)}
                      className={cn(
                        "w-full px-3 py-2 rounded-lg border text-sm",
                        "bg-white dark:bg-slate-800",
                        "border-slate-300 dark:border-slate-600",
                        "focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500",
                        relationType !== relation.relation_type && "border-indigo-400 bg-indigo-50 dark:bg-indigo-900/10"
                      )}
                      required
                    >
                      {Object.entries(groupedRelationOptions).map(([category, options]) => (
                        <optgroup key={category} label={category}>
                          {options.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </optgroup>
                      ))}
                    </select>
                    {relationType !== relation.relation_type && (
                      <p className="text-xs text-indigo-600 mt-1">Modificato</p>
                    )}
                  </div>

                  {/* Target Entity - Autocomplete */}
                  <div className="relative">
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                      Target (Entità di destinazione)
                    </label>
                    <p className="text-xs text-slate-500 mb-2">
                      Cerca qualsiasi entità: norme, sentenze, principi, concetti...
                    </p>
                    <div className="relative">
                      <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
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
                          "w-full pl-10 pr-4 py-2 rounded-lg border text-sm",
                          "bg-white dark:bg-slate-800",
                          "border-slate-300 dark:border-slate-600",
                          "focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500",
                          "placeholder:text-slate-400",
                          (selectedEntity || isNewEntity) && "border-indigo-400 bg-indigo-50 dark:bg-indigo-900/10"
                        )}
                      />
                      {isSearching && (
                        <Loader2 size={18} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 animate-spin" />
                      )}
                    </div>

                    {/* Selected entity/norm badge */}
                    {(selectedEntity || isNewEntity || isNormSelected) && (
                      <div className="mt-2 flex items-center gap-2">
                        <div className={cn(
                          "inline-flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs",
                          isNormSelected
                            ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                            : selectedEntity
                            ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"
                            : "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                        )}>
                          {isNormSelected && resolvedNorm ? (
                            <>
                              <Scale size={12} />
                              <span className="font-medium">{resolvedNorm.display_label}</span>
                              <span className="opacity-70">
                                ({resolvedNorm.exists_in_graph ? 'Nel grafo' : 'Pending'})
                              </span>
                            </>
                          ) : selectedEntity ? (
                            <>
                              <CheckCircle2 size={12} />
                              <span className="font-medium">{selectedEntity.nome}</span>
                              <span className="opacity-70">({ENTITY_TYPE_LABELS[selectedEntity.tipo]})</span>
                            </>
                          ) : (
                            <>
                              <Plus size={12} />
                              <span>Nuova: "{newEntityName}"</span>
                            </>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={handleClearTarget}
                          className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    )}

                    {/* Dropdown */}
                    <AnimatePresence>
                      {showDropdown && (searchResults.length > 0 || searchQuery.length >= 2) && (
                        <motion.div
                          ref={dropdownRef}
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          className={cn(
                            "absolute z-10 w-full mt-1 py-1 rounded-lg shadow-lg",
                            "bg-white dark:bg-slate-800",
                            "border border-slate-200 dark:border-slate-700",
                            "max-h-[200px] overflow-y-auto"
                          )}
                        >
                          {searchResults.map((entity) => (
                            <button
                              key={entity.id}
                              type="button"
                              onClick={() => handleSelectEntity(entity)}
                              className={cn(
                                "w-full px-3 py-2 text-left hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors",
                                "flex items-center justify-between gap-2"
                              )}
                            >
                              <div className="min-w-0">
                                <span className="font-medium text-slate-900 dark:text-white text-sm">
                                  {entity.nome}
                                </span>
                                <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400">
                                  {ENTITY_TYPE_LABELS[entity.tipo] || entity.tipo}
                                </span>
                              </div>
                              {entity.validation_status === 'approved' && (
                                <CheckCircle2 size={14} className="text-emerald-500 flex-shrink-0" />
                              )}
                            </button>
                          ))}

                          {/* Norm detected option (R5) */}
                          {parsedCitation && isSearchReady(parsedCitation) && (
                            <div className="border-t border-slate-200 dark:border-slate-700">
                              <div className="px-3 py-1 text-xs font-medium text-amber-600 dark:text-amber-400 uppercase tracking-wide bg-amber-50/50 dark:bg-amber-900/10">
                                Norma riconosciuta
                              </div>
                              <button
                                type="button"
                                onClick={handleSelectNorm}
                                disabled={isResolvingNorm}
                                className={cn(
                                  "w-full px-3 py-2 text-left",
                                  "bg-amber-50 dark:bg-amber-900/20 hover:bg-amber-100 dark:hover:bg-amber-900/30",
                                  "text-amber-700 dark:text-amber-300",
                                  "flex items-center gap-2 text-sm",
                                  "disabled:opacity-50 disabled:cursor-wait"
                                )}
                              >
                                {isResolvingNorm ? (
                                  <Loader2 size={14} className="animate-spin" />
                                ) : (
                                  <Scale size={14} />
                                )}
                                <span className="flex-1 font-medium">
                                  {formatParsedCitation(parsedCitation)}
                                </span>
                                {parsedCitation.fromAlias && (
                                  <span className="text-xs px-1.5 py-0.5 rounded bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200">
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
                                "w-full px-3 py-2 text-left",
                                "bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-700",
                                "text-slate-700 dark:text-slate-300",
                                "flex items-center gap-2 text-sm",
                                "border-t border-slate-200 dark:border-slate-700"
                              )}
                            >
                              <Plus size={14} />
                              <span>Crea "<strong>{searchQuery}</strong>" come nuovo target</span>
                            </button>
                          )}

                          {searchResults.length === 0 && searchQuery.length >= 2 && !isSearching && (
                            <div className="px-3 py-2 text-sm text-slate-500 dark:text-slate-400">
                              Nessuna entità trovata
                            </div>
                          )}
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {targetModified && (
                      <p className="text-xs text-indigo-600 mt-1">Target modificato</p>
                    )}
                  </div>

                  {/* Evidence */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                      Evidenza / Giustificazione
                    </label>
                    <textarea
                      value={evidence}
                      onChange={(e) => setEvidence(e.target.value)}
                      rows={4}
                      className={cn(
                        "w-full px-3 py-2 rounded-lg border text-sm resize-none",
                        "bg-white dark:bg-slate-800",
                        "border-slate-300 dark:border-slate-600",
                        "focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500",
                        "placeholder:text-slate-400",
                        evidence !== (relation.evidence || '') && "border-indigo-400 bg-indigo-50 dark:bg-indigo-900/10"
                      )}
                      placeholder="Testo che giustifica questa relazione (citazione, riferimento dottrinale...)"
                    />
                    {evidence !== (relation.evidence || '') && (
                      <p className="text-xs text-indigo-600 mt-1">Modificato</p>
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
                        "focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500",
                        "placeholder:text-slate-400"
                      )}
                      placeholder="Spiega perché proponi questa modifica..."
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
                    <div className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg p-3">
                      <p className="text-xs font-medium text-indigo-800 dark:text-indigo-300 mb-2">
                        Modifiche proposte:
                      </p>
                      <ul className="text-xs text-indigo-700 dark:text-indigo-400 space-y-1">
                        {relationType !== relation.relation_type && (
                          <li>• Tipo: {relation.relation_type} → {relationType}</li>
                        )}
                        {targetModified && (
                          <li>• Target: {relation.target_urn} → {getCurrentTargetUrn()}</li>
                        )}
                        {evidence !== (relation.evidence || '') && (
                          <li>• Evidenza modificata</li>
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
                      "bg-indigo-500 hover:bg-indigo-600 text-white",
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
