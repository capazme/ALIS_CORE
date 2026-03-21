import { useState, useCallback, useRef } from 'react';
import { merltService } from '../services/merltService';
import type { EntityType, PendingEntity, DuplicateCandidate } from '../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface ProposeEntityFormData {
  tipo: EntityType;
  nome: string;
  descrizione: string;
  evidence: string;
}

export interface UseProposeEntityFormOptions {
  articleUrn: string;
  userId: string;
  ambito?: string;
  onSuccess?: (entity: PendingEntity) => void;
  onClose: () => void;
  /** Called after successful submit, before auto-close timer fires */
  onBeforeClose?: (entity: PendingEntity | undefined) => void;
}

export interface UseProposeEntityFormReturn {
  formData: ProposeEntityFormData;
  setFormField: (field: keyof ProposeEntityFormData, value: string) => void;
  error: string | null;
  isValid: boolean;
  duplicates: DuplicateCandidate[];
  duplicatesFound: boolean;
  selectedDuplicate: string | null;
  setSelectedDuplicate: (id: string | null) => void;
  isSubmitting: boolean;
  success: boolean;
  handleSubmit: (e: { preventDefault: () => void }, skipDuplicateCheck?: boolean) => Promise<void>;
  handleConfirmCreate: () => Promise<void>;
  handleBackToForm: () => void;
  handleClose: () => void;
  /** Ref for cancellable auto-close timer — clear it on manual close */
  timerRef: { current: ReturnType<typeof setTimeout> | undefined };
}

const DEFAULT_FORM_DATA: ProposeEntityFormData = {
  tipo: 'concetto' as EntityType,
  nome: '',
  descrizione: '',
  evidence: '',
};

// =============================================================================
// HOOK
// =============================================================================

export function useProposeEntityForm({
  articleUrn,
  userId,
  ambito = 'civile',
  onSuccess,
  onClose,
}: UseProposeEntityFormOptions): UseProposeEntityFormReturn {
  const [formData, setFormDataState] = useState({ ...DEFAULT_FORM_DATA } as ProposeEntityFormData);
  const [error, setError] = useState(null as string | null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [duplicatesFound, setDuplicatesFound] = useState(false);
  const [duplicates, setDuplicates] = useState([] as DuplicateCandidate[]);
  const [selectedDuplicate, setSelectedDuplicate] = useState(null as string | null);
  const timerRef = useRef(undefined as ReturnType<typeof setTimeout> | undefined);

  const isValid =
    formData.nome.trim().length >= 3 && formData.descrizione.trim().length >= 10;

  const setFormField = useCallback(
    (field: keyof ProposeEntityFormData, value: string) => {
      setFormDataState((prev: ProposeEntityFormData) => ({ ...prev, [field]: value } as ProposeEntityFormData));
    },
    []
  );

  const resetForm = useCallback(() => {
    setFormDataState(DEFAULT_FORM_DATA);
    setError(null);
    setSuccess(false);
    setDuplicatesFound(false);
    setDuplicates([]);
    setSelectedDuplicate(null);
  }, []);

  const handleClose = useCallback(() => {
    clearTimeout(timerRef.current);
    resetForm();
    onClose();
  }, [onClose, resetForm]);

  const handleBackToForm = useCallback(() => {
    setDuplicatesFound(false);
    setDuplicates([]);
    setSelectedDuplicate(null);
  }, []);

  const handleSubmit = useCallback(
    async (e: { preventDefault: () => void }, skipDuplicateCheck = false) => {
      e.preventDefault();

      if (!formData.nome.trim()) {
        setError('Il nome è obbligatorio');
        return;
      }
      if (formData.nome.trim().length < 3) {
        setError('Il nome deve essere di almeno 3 caratteri');
        return;
      }
      if (!formData.descrizione.trim()) {
        setError('La descrizione è obbligatoria');
        return;
      }
      if (formData.descrizione.trim().length < 10) {
        setError('La descrizione deve essere di almeno 10 caratteri');
        return;
      }

      setIsSubmitting(true);
      setError(null);

      try {
        const result = await merltService.proposeEntity({
          tipo: formData.tipo,
          nome: formData.nome.trim(),
          descrizione: formData.descrizione.trim(),
          article_urn: articleUrn,
          ambito,
          evidence:
            formData.evidence.trim() || `Proposto manualmente per ${articleUrn}`,
          user_id: userId,
          skip_duplicate_check: skipDuplicateCheck,
          acknowledged_duplicate_of: skipDuplicateCheck
            ? selectedDuplicate ?? undefined
            : undefined,
        });

        if (result.duplicate_action_required && result.duplicates.length > 0) {
          setDuplicates(result.duplicates);
          setDuplicatesFound(true);
          setIsSubmitting(false);
          return;
        }

        setSuccess(true);

        timerRef.current = setTimeout(() => {
          if (result.pending_entity) {
            onSuccess?.(result.pending_entity);
          }
          handleClose();
        }, 1500);
      } catch (err: unknown) {
        const errorMessage =
          err instanceof Error ? err.message : "Errore nella proposta dell'entità";
        setError(errorMessage);
      } finally {
        setIsSubmitting(false);
      }
    },
    [formData, articleUrn, ambito, userId, selectedDuplicate, onSuccess, handleClose]
  );

  const handleConfirmCreate = useCallback(async () => {
    const syntheticEvent = { preventDefault: () => {} } as { preventDefault: () => void };
    await handleSubmit(syntheticEvent, true);
  }, [handleSubmit]);

  return {
    formData,
    setFormField,
    error,
    isValid,
    duplicates,
    duplicatesFound,
    selectedDuplicate,
    setSelectedDuplicate,
    isSubmitting,
    success,
    handleSubmit,
    handleConfirmCreate,
    handleBackToForm,
    handleClose,
    timerRef,
  };
}
