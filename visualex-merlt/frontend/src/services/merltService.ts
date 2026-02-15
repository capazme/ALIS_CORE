/**
 * MERL-T Service Layer
 * ====================
 *
 * Service layer per l'integrazione con il backend MERL-T.
 * Gestisce Expert System Q&A, Enrichment, Validation e Graph.
 *
 * Tutte le chiamate passano attraverso il proxy VisuaLex backend
 * che forwarda a MERL-T (configurato in MERLT_API_URL).
 */

import { get, post } from './api';
import type {
  EntityType,
  RelationType,
  // Expert System
  ExpertQueryRequest,
  ExpertQueryResponse,
  InlineFeedbackRequest,
  DetailedFeedbackRequest,
  SourceFeedbackRequest,
  // Citation Export
  CitationExportRequest,
  CitationExportResponse,
  // Enrichment & Validation
  PendingEntity,
  PendingRelation,
  PendingQueueResponse,
  EntityValidationRequest,
  EntityValidationResponse,
  RelationValidationRequest,
  RelationValidationResponse,
  // Graph
  GraphData,
  ArticleGraphStatus,
  NodeDetails,
  // Subgraph API
  SubgraphRequest,
  SubgraphResponse,
  // Authority
  UserAuthority,
  // Norm Resolver (R5)
  NormResolveRequest,
  NormResolveResponse,
  // Deduplication
  DuplicateCheckResponse,
  EntityProposalResponse,
  RelationDuplicateCheckResponse,
  RelationProposalResponse,
  // Profile
  ProfileResponse,
  // Issue Reporting
  IssueType,
  IssueSeverity,
  IssueStatus,
  ReportIssueRequest,
  ReportIssueResponse,
  VoteIssueRequest,
  VoteIssueResponse,
  GetEntityIssuesResponse,
  OpenIssuesRequest,
  OpenIssuesResponse,
  // Dossier Training Set (R5)
  DossierTrainingSetExportFullRequest,
  DossierTrainingSetExportResponse,
  LoadDossierTrainingRequest,
  LoadDossierTrainingResponse,
  // NER Feedback
  NERFeedbackRequest,
  NERFeedbackResponse,
  NERConfirmRequest,
  RouterFeedbackRequest,
} from '../types/merlt';

// =============================================================================
// MERL-T API PREFIX
// =============================================================================

const MERLT_PREFIX = '/merlt';  // apiClient adds /api prefix; Vite proxy rewrites /api/merlt/* to /api/v1/* on MERL-T backend (port 8000)

// =============================================================================
// EXPERT SYSTEM (Q&A)
// =============================================================================

/**
 * Invia una query al sistema multi-esperto MERL-T.
 *
 * @param request.include_trace - Se true, salva il reasoning trace per consultazione successiva
 * @param request.consent_level - Livello consenso GDPR: 'anonymous' | 'basic' | 'full'
 *
 * @example
 * const response = await queryExperts({
 *   query: "Cos'è la legittima difesa?",
 *   user_id: "user123",
 *   include_trace: true,
 *   consent_level: 'basic',
 * });
 * console.log(response.trace_id);
 */
export async function queryExperts(request: ExpertQueryRequest): Promise<ExpertQueryResponse> {
  return post<ExpertQueryResponse>(`${MERLT_PREFIX}/experts/query`, request);
}

/**
 * Invia feedback rapido (thumbs up/down) per una risposta Q&A.
 *
 * @param trace_id - ID della traccia Q&A
 * @param user_id - ID utente
 * @param rating - Valutazione 1-5 (1=thumbs down, 5=thumbs up)
 */
export async function submitInlineFeedback(
  trace_id: string,
  user_id: string,
  rating: 1 | 2 | 3 | 4 | 5
): Promise<{ success: boolean; message: string }> {
  const request: InlineFeedbackRequest = { trace_id, user_id, rating };
  return post(`${MERLT_PREFIX}/experts/feedback/inline`, request);
}

/**
 * Invia feedback dettagliato su 3 dimensioni per una risposta Q&A.
 *
 * @param data - Feedback con accuracy, completeness, relevance (0-1)
 */
export async function submitDetailedFeedback(
  data: DetailedFeedbackRequest
): Promise<{ success: boolean; message: string }> {
  return post(`${MERLT_PREFIX}/experts/feedback/detailed`, {
    trace_id: data.trace_id,
    user_id: data.user_id,
    retrieval_score: (data.accuracy - 1) / 4,      // 1→0, 5→1 (backend expects 0-1)
    reasoning_score: (data.completeness - 1) / 4,
    synthesis_score: (data.relevance - 1) / 4,
    comment: data.comment,
  });
}

/**
 * Invia feedback su una singola fonte citata nella risposta.
 *
 * @param data - Feedback con source_urn e rating
 */
export async function submitSourceFeedback(
  data: SourceFeedbackRequest
): Promise<{ success: boolean; message: string }> {
  return post(`${MERLT_PREFIX}/experts/feedback/source`, {
    trace_id: data.trace_id,
    user_id: data.user_id,
    source_id: data.source_urn,
    relevance: data.rating,
  });
}

/**
 * Invia feedback sulla preferenza Expert in caso di interpretazioni divergenti.
 *
 * Quando mode=divergent, l'utente puo' indicare quale Expert
 * ha fornito l'interpretazione piu' utile. Questo feedback
 * viene usato per:
 * - Training del sistema RLCF
 * - Migliorare i pesi degli Expert
 * - Ottimizzare la sintesi delle risposte
 *
 * @param trace_id - ID della traccia Q&A
 * @param user_id - ID utente
 * @param preferred_expert - Expert preferito (literal, systemic, principles, precedent)
 * @param comment - Commento opzionale con motivazione
 *
 * @example
 * await submitExpertPreferenceFeedback(
 *   "trace_123",
 *   "user_456",
 *   "systemic",
 *   "L'interpretazione sistematica e' piu' completa"
 * );
 */
export async function submitExpertPreferenceFeedback(
  trace_id: string,
  user_id: string,
  preferred_expert: string,
  comment?: string
): Promise<{ success: boolean; message: string }> {
  return post(`${MERLT_PREFIX}/experts/feedback/preference`, {
    trace_id,
    user_id,
    preferred_expert,
    comment,
  });
}

/**
 * Submit router feedback from high-authority users (F2).
 *
 * Only users with authority >= 0.7 can evaluate routing decisions.
 * Authority is verified server-side — no need to send user_authority.
 */
export async function submitRouterFeedback(
  data: RouterFeedbackRequest
): Promise<{ success: boolean; message: string }> {
  return post(`${MERLT_PREFIX}/experts/feedback/router`, data);
}


// =============================================================================
// ENRICHMENT
// =============================================================================

/**
 * Verifica se un articolo esiste nel Knowledge Graph MERL-T.
 *
 * @param tipo_atto - Tipo di atto (es. "codice civile", "legge")
 * @param articolo - Numero articolo (es. "1218", "52")
 * @param numero_atto - Numero atto opzionale (es. "241" per legge 241/1990)
 * @param data - Data atto opzionale (es. "1990-08-07")
 *
 * @example
 * const status = await checkArticleInGraph("codice civile", "1218");
 * if (status.in_graph) {
 *   console.log(`Articolo nel grafo con ${status.node_count} entità`);
 * }
 */
export async function checkArticleInGraph(
  tipo_atto: string,
  articolo: string,
  numero_atto?: string,
  data?: string
): Promise<ArticleGraphStatus> {
  const params = new URLSearchParams({
    tipo_atto,
    articolo,
    ...(numero_atto && { numero_atto }),
    ...(data && { data }),
  });

  const response = await get<{
    in_graph: boolean;
    node_count: number;
    has_entities: boolean;
    last_updated: string | null;
    article_urn: string | null;
  }>(`${MERLT_PREFIX}/enrichment/check-article?${params}`);

  return {
    exists: response.in_graph,
    node_id: response.article_urn || undefined,
    pending_validation: false,  // TODO: backend should return this
    entity_count: response.node_count,
  };
}

/**
 * Richiede live enrichment per un articolo.
 *
 * Estrae automaticamente entità e relazioni usando LLM,
 * salvandole come "pending" per validazione community.
 *
 * @param tipo_atto - Tipo di atto
 * @param articolo - Numero articolo
 * @param user_id - ID utente che richiede l'enrichment
 * @param user_authority - Authority score utente (0-1), default 0.5 per nuovi utenti
 * @param article_text - Testo dell'articolo (opzionale, verrà scaricato se non fornito)
 */
export async function requestLiveEnrichment(
  tipo_atto: string,
  articolo: string,
  user_id: string,
  user_authority: number = 0.5,
  article_text?: string
): Promise<{
  pending_entities: PendingEntity[];
  pending_relations: PendingRelation[];
  article_urn: string;
}> {
  // LLM extraction can take 1-2 minutes, so we need a longer timeout
  return post(
    `${MERLT_PREFIX}/enrichment/live`,
    {
      tipo_atto,
      articolo,
      user_id,
      user_authority,
      article_text,
    },
    { timeout: 120000 } // 2 minutes
  );
}

/**
 * SSE Event types per streaming enrichment
 */
export interface StreamingEvent {
  type: 'start' | 'progress' | 'entity' | 'relation' | 'complete' | 'error' | 'warning' | 'waiting';
  article?: {
    urn: string;
    tipo_atto: string;
    numero_articolo: string;
    rubrica: string;
  };
  entity?: PendingEntity;
  relation?: PendingRelation;
  message?: string;
  summary?: {
    entities_count: number;
    relations_count: number;
    extraction_time_ms: number;
    sources_used: string[];
  };
  // Waiting event fields (when another user is already extracting)
  progress?: number;
  elapsed?: number;
}

/**
 * Richiede live enrichment con streaming SSE.
 *
 * Le entità vengono ricevute una alla volta man mano che sono estratte.
 * Migliore UX per estrazioni lunghe.
 *
 * @param tipo_atto - Tipo di atto
 * @param articolo - Numero articolo
 * @param user_id - ID utente
 * @param onEvent - Callback chiamata per ogni evento ricevuto
 * @param user_authority - Authority score utente (default 0.5)
 *
 * @returns Funzione per chiudere la connessione
 *
 * @example
 * const close = requestLiveEnrichmentStreaming(
 *   "codice civile",
 *   "1337",
 *   "user123",
 *   (event) => {
 *     if (event.type === 'entity') {
 *       console.log('New entity:', event.entity);
 *     }
 *   }
 * );
 * // Later: close() to abort
 */
export function requestLiveEnrichmentStreaming(
  tipo_atto: string,
  articolo: string,
  user_id: string,
  onEvent: (event: StreamingEvent) => void,
  user_authority: number = 0.5
): () => void {
  const params = new URLSearchParams({
    tipo_atto,
    articolo,
    user_id,
    user_authority: user_authority.toString(),
    include_brocardi: 'true',
  });

  // Use the API base URL for SSE
  const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
  const url = `${API_BASE_URL}${MERLT_PREFIX}/enrichment/live/stream?${params}`;

  const eventSource = new EventSource(url);

  // Handle different event types
  const eventTypes = ['start', 'progress', 'entity', 'relation', 'complete', 'error', 'warning', 'waiting'];

  eventTypes.forEach(eventType => {
    eventSource.addEventListener(eventType, (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        onEvent(data as StreamingEvent);

        // Auto-close on complete or error
        if (eventType === 'complete' || eventType === 'error') {
          eventSource.close();
        }
      } catch (e) {
        console.error('Failed to parse SSE event:', e);
      }
    });
  });

  eventSource.onerror = (error) => {
    console.error('SSE connection error:', error);
    onEvent({ type: 'error', message: 'Connection lost' });
    eventSource.close();
  };

  // Return close function
  return () => {
    eventSource.close();
  };
}

// =============================================================================
// VALIDATION QUEUE
// =============================================================================

/**
 * Recupera la coda di entità e relazioni pending per validazione.
 *
 * @param user_id - ID utente (per calcolare authority)
 * @param options - Opzioni di filtro
 *
 * @example
 * const queue = await getPendingQueue("user123", {
 *   legal_domain: "civile",
 *   limit: 20
 * });
 */
export async function getPendingQueue(
  user_id: string,
  options?: {
    legal_domain?: string;
    tipo_atto?: string;     // Filter by act type (e.g., "codice civile", "costituzione")
    article_urn?: string;   // Filter by article URN pattern (e.g., "~art1")
    limit?: number;
    updated_after?: Date;
  }
): Promise<PendingQueueResponse> {
  const params = new URLSearchParams({
    user_id,
    ...(options?.legal_domain && { legal_domain: options.legal_domain }),
    ...(options?.tipo_atto && { tipo_atto: options.tipo_atto }),
    ...(options?.article_urn && { article_urn: options.article_urn }),
    ...(options?.limit && { limit: options.limit.toString() }),
    ...(options?.updated_after && { updated_after: options.updated_after.toISOString() }),
  });

  return get<PendingQueueResponse>(`${MERLT_PREFIX}/enrichment/pending?${params}`);
}

/**
 * Valida una singola entità con voto pesato per authority.
 *
 * Il voto viene pesato per l'authority dell'utente nel dominio legale.
 * Se si raggiunge il consenso (threshold 2.0), l'entità viene scritta nel grafo.
 *
 * @param request - Richiesta di validazione
 */
export async function validateEntity(
  request: EntityValidationRequest
): Promise<EntityValidationResponse> {
  // Build suggested_edits with all provided fields
  const suggestedEdits = request.suggested_edit ? {
    ...(request.suggested_edit.nome !== undefined && { nome: request.suggested_edit.nome }),
    ...(request.suggested_edit.tipo !== undefined && { tipo: request.suggested_edit.tipo }),
    ...(request.suggested_edit.descrizione !== undefined && { descrizione: request.suggested_edit.descrizione }),
    ...(request.suggested_edit.ambito !== undefined && { ambito: request.suggested_edit.ambito }),
  } : undefined;

  return post<EntityValidationResponse>(`${MERLT_PREFIX}/enrichment/validate-entity`, {
    entity_id: request.entity_id,
    user_id: request.user_id,
    vote: request.vote,
    reason: request.comment,
    suggested_edits: Object.keys(suggestedEdits || {}).length > 0 ? suggestedEdits : undefined,
  });
}

/**
 * Valida una singola relazione.
 */
export async function validateRelation(
  request: RelationValidationRequest
): Promise<RelationValidationResponse> {
  // Build suggested_edits with all provided fields
  const suggestedEdits = request.suggested_edit ? {
    ...(request.suggested_edit.relation_type !== undefined && { relation_type: request.suggested_edit.relation_type }),
    ...(request.suggested_edit.target_urn !== undefined && { target_urn: request.suggested_edit.target_urn }),
    ...(request.suggested_edit.evidence !== undefined && { evidence: request.suggested_edit.evidence }),
  } : undefined;

  return post<RelationValidationResponse>(`${MERLT_PREFIX}/enrichment/validate-relation`, {
    relation_id: request.relation_id,
    user_id: request.user_id,
    vote: request.vote,
    reason: request.comment,
    suggested_edits: Object.keys(suggestedEdits || {}).length > 0 ? suggestedEdits : undefined,
  });
}

// =============================================================================
// DEDUPLICATION
// =============================================================================

/**
 * Verifica se esistono entità duplicate prima di proporre.
 *
 * Usa questa funzione prima di proposeEntity per:
 * 1. Evitare duplicati
 * 2. Mostrare entità simili all'utente
 * 3. Permettere di modificare entità esistenti
 *
 * @example
 * const result = await checkEntityDuplicate({
 *   entity_text: "Buona fede",
 *   entity_type: "concetto",
 *   article_urn: "urn:nir:stato:codice.civile:1942;art1337"
 * });
 *
 * if (result.has_duplicates) {
 *   // Mostra duplicati all'utente
 *   showDuplicateWarning(result.duplicates);
 * }
 */
export async function checkEntityDuplicate(data: {
  entity_text: string;
  entity_type: EntityType;
  article_urn?: string;
  scope?: 'global' | 'article' | 'type';
}): Promise<DuplicateCheckResponse> {
  return post(`${MERLT_PREFIX}/enrichment/check-duplicate`, data);
}

/**
 * Verifica se esistono relazioni duplicate prima di proporre.
 *
 * Usa questa funzione prima di proposeRelation per evitare duplicati.
 * Le relazioni sono considerate duplicate se:
 * - Stesso source e target
 * - Stesso tipo di relazione
 *
 * @example
 * const result = await checkRelationDuplicate({
 *   source_entity_id: "urn:nir:stato:codice.civile:1942;art1337",
 *   target_entity_id: "concetto:buona_fede",
 *   relation_type: "DISCIPLINA"
 * });
 *
 * if (result.has_duplicates) {
 *   showDuplicateWarning(result.duplicates);
 * }
 */
export async function checkRelationDuplicate(data: {
  source_entity_id: string;
  target_entity_id: string;
  relation_type: RelationType;
}): Promise<RelationDuplicateCheckResponse> {
  return post(`${MERLT_PREFIX}/enrichment/check-relation-duplicate`, data);
}


// =============================================================================
// ENTITY/RELATION PROPOSALS
// =============================================================================

/**
 * Propone una nuova entità per validazione community.
 *
 * Se vengono trovati duplicati e skip_duplicate_check=false (default),
 * la response conterrà i duplicati e l'entità NON verrà creata.
 * L'utente deve confermare per procedere con skip_duplicate_check=true.
 *
 * @example
 * // Prima chiamata - potrebbe tornare duplicati
 * const result = await proposeEntity({
 *   tipo: "principio",
 *   nome: "Principio di buona fede",
 *   descrizione: "Obbligo di comportamento leale",
 *   article_urn: "urn:nir:stato:codice.civile:1942;art1175",
 *   ambito: "civile",
 *   evidence: "Art. 1175 c.c. stabilisce che...",
 *   user_id: "user123"
 * });
 *
 * if (result.duplicate_action_required) {
 *   // Mostra duplicati all'utente e chiedi conferma
 *   const confirmed = await showDuplicateDialog(result.duplicates);
 *   if (confirmed) {
 *     // Seconda chiamata - salta il check
 *     await proposeEntity({
 *       ...data,
 *       skip_duplicate_check: true,
 *       acknowledged_duplicate_of: result.duplicates[0].entity_id
 *     });
 *   }
 * }
 */
export async function proposeEntity(data: {
  tipo: EntityType;
  nome: string;
  descrizione: string;
  article_urn: string;
  ambito: string;
  evidence: string;
  user_id: string;
  skip_duplicate_check?: boolean;
  acknowledged_duplicate_of?: string;
}): Promise<EntityProposalResponse> {
  return post(`${MERLT_PREFIX}/enrichment/propose-entity`, data);
}

/**
 * Propone una nuova relazione per validazione community.
 *
 * Se vengono trovati duplicati e skip_duplicate_check=false (default),
 * la response conterrà i duplicati e la relazione NON verrà creata
 * se c'è un exact match.
 *
 * @example
 * const result = await proposeRelation({
 *   tipo_relazione: "DISCIPLINA",
 *   source_urn: "urn:nir:stato:codice.civile:1942;art1337",
 *   target_entity_id: "concetto:buona_fede",
 *   article_urn: "urn:nir:stato:codice.civile:1942;art1337",
 *   descrizione: "L'articolo disciplina il concetto di buona fede",
 *   certezza: 0.8,
 *   user_id: "user123"
 * });
 *
 * if (result.duplicate_action_required) {
 *   // Exact match - cannot proceed
 *   showError("Relazione già esistente");
 * } else if (result.has_duplicates) {
 *   // Fuzzy match - show warning but relation was created
 *   showWarning("Relazioni simili esistenti", result.duplicates);
 * }
 */
export async function proposeRelation(data: {
  tipo_relazione: RelationType;
  source_urn: string;
  target_entity_id: string;
  article_urn: string;
  descrizione: string;
  certezza: number;
  user_id: string;
  skip_duplicate_check?: boolean;
  acknowledged_duplicate_of?: string;
}): Promise<RelationProposalResponse> {
  return post(`${MERLT_PREFIX}/enrichment/propose-relation`, data);
}

// =============================================================================
// USER AUTHORITY
// =============================================================================

/**
 * Recupera l'authority score di un utente.
 *
 * L'authority determina il peso dei voti nelle validazioni.
 * Si basa su:
 * - Contributi validati (entità/relazioni approvate)
 * - Qualità del feedback Q&A
 * - Accuratezza dei voti precedenti
 */
export async function getUserAuthority(user_id: string): Promise<UserAuthority> {
  return get<UserAuthority>(`${MERLT_PREFIX}/authority/${user_id}`);
}

/**
 * Recupera il profilo completo dell'utente con authority e statistiche.
 *
 * Include:
 * - Authority RLCF con breakdown (B_u, T_u, P_u)
 * - Tier attuale e progress verso il prossimo
 * - Statistiche contributi (totali, approvati, rigettati)
 * - Authority per dominio legale
 * - Attività recente
 *
 * @param user_id - ID utente
 *
 * @example
 * const profile = await getUserProfile("user123");
 * console.log(`${profile.authority.tier}: ${profile.authority.score}`);
 */
export async function getUserProfile(user_id: string): Promise<ProfileResponse> {
  return get<ProfileResponse>(`${MERLT_PREFIX}/profile/full?user_id=${encodeURIComponent(user_id)}`);
}

// =============================================================================
// ENTITY SEARCH (R3 Autocomplete)
// =============================================================================

/**
 * Ricerca entità nel grafo per autocomplete.
 *
 * Usato in ProposeRelationDrawer per selezionare entità esistenti.
 * Ordina per: exact match > approval_score > llm_confidence
 *
 * @param query - Stringa di ricerca (min 2 caratteri)
 * @param article_urn - URN articolo opzionale per filtrare
 * @param limit - Numero massimo risultati (default 10)
 *
 * @example
 * const results = await searchEntities("buona fede");
 * // [{ id: "...", nome: "Buona fede", tipo: "principio", ... }]
 */
export async function searchEntities(
  query: string,
  article_urn?: string,
  limit: number = 10
): Promise<Array<{
  id: string;
  nome: string;
  tipo: string;
  approval_score: number;
  validation_status: 'pending' | 'approved' | 'rejected';
}>> {
  const params = new URLSearchParams({
    q: query,
    limit: limit.toString(),
    ...(article_urn && { article_urn }),
  });

  return get(`${MERLT_PREFIX}/graph/entities/search?${params}`);
}

// =============================================================================
// NORM RESOLVER (R5)
// =============================================================================

/**
 * Risolve una citazione normativa per ProposeRelationDrawer.
 *
 * Quando l'utente inserisce una norma in linguaggio naturale (es. "Art. 1218 c.c."),
 * questa funzione:
 * 1. Genera l'URN corretto
 * 2. Verifica se la norma esiste nel grafo
 * 3. Se non esiste, crea una PendingEntity tipo=norma
 *
 * Il risultato può essere usato direttamente in proposeRelation.
 *
 * @param request - Dati parsed dal citationParser
 * @returns entity_id e metadata per la UI
 *
 * @example
 * // Frontend: l'utente digita "Art. 1218 c.c."
 * const parsed = parseLegalCitation("Art. 1218 c.c.");
 * if (parsed && isSearchReady(parsed)) {
 *   const result = await resolveNorm({
 *     act_type: parsed.actType,
 *     article: parsed.article,
 *     source_article_urn: currentArticleUrn,
 *     user_id: userId,
 *   });
 *   if (result.resolved) {
 *     // Ora puoi usare result.entity_id in proposeRelation
 *     await proposeRelation({
 *       target_entity_id: result.entity_id,
 *       // ...
 *     });
 *   }
 * }
 */
export async function resolveNorm(
  request: NormResolveRequest
): Promise<NormResolveResponse> {
  return post<NormResolveResponse>(`${MERLT_PREFIX}/graph/resolve-norm`, request);
}

// =============================================================================
// RLCF FEEDBACK (R2 + 7.2 NER Training)
// =============================================================================

/**
 * Invia feedback sulla selezione testuale per il training NER.
 *
 * Quando l'utente seleziona testo e propone un'entità (R2),
 * questo feedback viene usato per addestrare il modello NER
 * per l'estrazione automatica di entità giuridiche.
 *
 * @param data - Dati della selezione con offset e tipo entità
 */
export async function submitEntitySelectionFeedback(data: {
  selected_text: string;
  start_offset: number;
  end_offset: number;
  entity_type: string;
  article_urn: string;
  article_text: string;
  user_id: string;
}): Promise<{ success: boolean }> {
  return post(`${MERLT_PREFIX}/rlcf/entity-selection-feedback`, data);
}

// =============================================================================
// ISSUE REPORTING (RLCF Feedback Loop)
// =============================================================================

/**
 * Segnala un problema su un'entita' approvata nel Knowledge Graph.
 *
 * Flusso RLCF:
 * 1. Utente vede incongruenza nel grafo
 * 2. Crea issue con tipo, gravita' e descrizione
 * 3. Community vota sulla validita' dell'issue
 * 4. Se upvote_score >= threshold → entity torna in needs_revision
 *
 * @param request - Dati della segnalazione
 *
 * @example
 * const result = await reportNodeIssue({
 *   entity_id: "concetto:legittima_difesa",
 *   issue_type: "factual_error",
 *   severity: "high",
 *   description: "La definizione non e' corretta...",
 *   user_id: "user123"
 * });
 */
export async function reportNodeIssue(
  request: ReportIssueRequest
): Promise<ReportIssueResponse> {
  return post<ReportIssueResponse>(`${MERLT_PREFIX}/enrichment/report-issue`, request);
}

/**
 * Vota su una issue esistente.
 *
 * - upvote: Conferma che l'issue e' valida
 * - downvote: L'issue non e' valida
 *
 * Il voto e' pesato per l'authority dell'utente.
 * Se upvote_score raggiunge la soglia (2.0), l'entita' torna in validazione.
 *
 * @param request - Dati del voto
 *
 * @example
 * const result = await voteOnIssue({
 *   issue_id: "issue:abc123",
 *   vote: "upvote",
 *   comment: "Confermo, l'articolo citato e' errato",
 *   user_id: "user123"
 * });
 *
 * if (result.entity_reopened) {
 *   toast.success("Entita' riaperta per validazione");
 * }
 */
export async function voteOnIssue(
  request: VoteIssueRequest
): Promise<VoteIssueResponse> {
  return post<VoteIssueResponse>(`${MERLT_PREFIX}/enrichment/vote-issue`, request);
}

/**
 * Recupera le issue associate a un'entita'.
 *
 * Utile per:
 * - Mostrare issue aperte nel NodeDetailsPanel
 * - Controllare se un nodo ha problemi segnalati
 *
 * @param entity_id - ID dell'entita'
 *
 * @example
 * const { issues, open_count } = await getEntityIssues("concetto:abc123");
 * if (open_count > 0) {
 *   showIssuesBadge(open_count);
 * }
 */
export async function getEntityIssues(
  entity_id: string
): Promise<GetEntityIssuesResponse> {
  return get<GetEntityIssuesResponse>(
    `${MERLT_PREFIX}/enrichment/entity-issues/${encodeURIComponent(entity_id)}`
  );
}

/**
 * Recupera lista issue aperte per moderazione/votazione.
 *
 * Supporta filtri per stato, gravita' e tipo.
 * Usato nella tab "Segnalazioni" del forum.
 *
 * @param options - Filtri opzionali
 *
 * @example
 * const { issues, total } = await getOpenIssues({ status: 'open', limit: 20 });
 * // Mostra lista issue votabili
 */
export async function getOpenIssues(
  options?: OpenIssuesRequest
): Promise<OpenIssuesResponse> {
  const params = new URLSearchParams();

  if (options?.status) params.append('status', options.status);
  if (options?.severity) params.append('severity', options.severity);
  if (options?.issue_type) params.append('issue_type', options.issue_type);
  if (options?.offset !== undefined) params.append('offset', options.offset.toString());
  if (options?.limit !== undefined) params.append('limit', options.limit.toString());

  const queryString = params.toString();
  const url = `${MERLT_PREFIX}/enrichment/open-issues${queryString ? `?${queryString}` : ''}`;

  return get<OpenIssuesResponse>(url);
}

// =============================================================================
// GRAPH ARTICLE ENTITIES/RELATIONS
// =============================================================================

/**
 * Recupera le entità associate a un articolo.
 *
 * @param articleUrn - URN dell'articolo
 * @param validationStatus - Filtro opzionale per stato validazione
 */
export async function getArticleEntities(
  articleUrn: string,
  validationStatus?: string
): Promise<{ entities: Array<{ id: string; name: string; type: string; confidence: number; position?: { start: number; end: number } }> }> {
  const params = new URLSearchParams({ article_urn: articleUrn });
  if (validationStatus) params.set('validation_status', validationStatus);
  return get(`${MERLT_PREFIX}/graph/article-entities?${params}`);
}

/**
 * Recupera le relazioni associate a un articolo.
 *
 * @param articleUrn - URN dell'articolo
 * @param relationType - Filtro opzionale per tipo relazione
 */
export async function getArticleRelations(
  articleUrn: string,
  relationType?: string
): Promise<{ relations: Array<{ id: string; sourceId: string; targetId: string; type: string; confidence: number }> }> {
  const params = new URLSearchParams({ article_urn: articleUrn });
  if (relationType) params.set('relation_type', relationType);
  return get(`${MERLT_PREFIX}/graph/article-relations?${params}`);
}

// =============================================================================
// GRAPH QUERIES (Future)
// =============================================================================

/**
 * Recupera i dettagli completi di un nodo del grafo.
 *
 * @param node_id - ID del nodo (es. "concetto:abc123")
 */
export async function getNodeDetails(node_id: string): Promise<NodeDetails> {
  return get<NodeDetails>(`${MERLT_PREFIX}/graph/node/${encodeURIComponent(node_id)}`);
}

/**
 * Recupera il sottografo locale intorno a un articolo.
 *
 * @param article_urn - URN dell'articolo
 * @param depth - Profondità di traversal (default 2)
 */
export async function getArticleSubgraph(
  article_urn: string,
  depth: number = 2
): Promise<GraphData> {
  const params = new URLSearchParams({
    article_urn,
    depth: depth.toString(),
  });
  return get<GraphData>(`${MERLT_PREFIX}/graph/subgraph?${params}`);
}

/**
 * Recupera il subgraph per la visualizzazione Knowledge Graph.
 *
 * Endpoint ottimizzato per graph visualization con:
 * - Nodi con label, tipo, proprietà e metadati
 * - Archi con tipo e proprietà
 * - Metadati query (tempo, filtri, etc.)
 *
 * @param request - Parametri richiesta subgraph
 *
 * @example
 * const subgraph = await getSubgraph({
 *   root_urn: "https://www.normattiva.it/...~art1218",
 *   depth: 2,
 *   max_nodes: 50,
 *   include_metadata: true
 * });
 * // Usa subgraph.nodes e subgraph.edges per visualizzazione
 */
export async function getSubgraph(request: SubgraphRequest): Promise<SubgraphResponse> {
  const params = new URLSearchParams({
    root_urn: request.root_urn,
    ...(request.depth !== undefined && { depth: request.depth.toString() }),
    ...(request.max_nodes !== undefined && { max_nodes: request.max_nodes.toString() }),
    ...(request.include_metadata !== undefined && { include_metadata: request.include_metadata.toString() }),
  });

  // Add array parameters
  if (request.entity_types?.length) {
    request.entity_types.forEach(t => params.append('entity_types', t));
  }
  if (request.relation_types?.length) {
    request.relation_types.forEach(t => params.append('relation_types', t));
  }

  return get<SubgraphResponse>(`${MERLT_PREFIX}/graph/subgraph?${params}`);
}

/**
 * Get a graph overview (most-connected nodes) for the bulletin board explorer.
 */
export async function getGraphOverview(maxNodes: number = 50): Promise<SubgraphResponse> {
  return get<SubgraphResponse>(`${MERLT_PREFIX}/graph/overview?max_nodes=${maxNodes}`);
}

// =============================================================================
// DOSSIER TRAINING SET (R5)
// =============================================================================

/**
 * Esporta un dossier come training set per RLCF.
 *
 * Versione completa che invia i dati del dossier e cerca
 * sessioni Q&A correlate agli articoli.
 *
 * @example
 * const trainingSet = await exportDossierTrainingSet({
 *   dossier_title: "Studio responsabilità",
 *   articles: [{tipo_atto: "codice civile", numero_articolo: "1453", ...}],
 *   user_id: "user123"
 * });
 * console.log(trainingSet.qa_sessions_count); // 5
 */
export async function exportDossierTrainingSet(
  request: DossierTrainingSetExportFullRequest
): Promise<DossierTrainingSetExportResponse> {
  return post<DossierTrainingSetExportResponse>(
    `${MERLT_PREFIX}/enrichment/dossier-training-export-full`,
    request
  );
}

/**
 * Carica un training set esportato nel buffer RLCF.
 *
 * I dossier curati hanno priorità elevata perché rappresentano
 * dati di alta qualità (studio umano verificato).
 *
 * @example
 * const result = await loadDossierTrainingSet({
 *   training_set: exportedTrainingSet,
 *   priority_boost: 0.3  // +30% priorità
 * });
 * console.log(result.training_ready); // true se soglia raggiunta
 */
export async function loadDossierTrainingSet(
  request: LoadDossierTrainingRequest
): Promise<LoadDossierTrainingResponse> {
  return post<LoadDossierTrainingResponse>(
    `${MERLT_PREFIX}/enrichment/load-dossier-training`,
    request
  );
}

// =============================================================================
// CITATION EXPORT
// =============================================================================

/**
 * Esporta citazioni da un trace in formato file scaricabile.
 *
 * Formati disponibili: italian_legal, bibtex, plain_text, json
 *
 * @param request - Richiesta di export con trace_id e formato
 * @returns Response con download_url per il file generato
 *
 * @example
 * const result = await exportCitations({
 *   trace_id: "trace_123",
 *   format: "bibtex",
 *   include_query_summary: true
 * });
 * // Scarica il file da result.download_url
 */
export async function exportCitations(
  request: CitationExportRequest
): Promise<CitationExportResponse> {
  return post<CitationExportResponse>(`${MERLT_PREFIX}/citations/export`, request);
}

/**
 * Trigger file download via hidden <a> element.
 *
 * Rewrites the backend download URL through the frontend proxy path
 * and initiates the download without popup blockers.
 *
 * @param downloadUrl - URL from exportCitations response (e.g. "/api/v1/citations/download/...")
 * @param filename - Suggested filename for the download
 */
export function downloadCitationFile(downloadUrl: string, filename: string = 'citations'): void {
  const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
  const downloadPath = downloadUrl.replace('/api/v1/', '/merlt/');
  triggerFileDownload(`${API_BASE_URL}${downloadPath}`, filename);
}

/**
 * Shared download helper — creates a hidden <a> element to trigger file download.
 */
export function triggerFileDownload(url: string, filename: string): void {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// =============================================================================
// NER FEEDBACK (Citation Correction)
// =============================================================================

/**
 * Invia feedback NER per correggere una citazione.
 * Usato quando l'utente corregge una citazione rilevata erroneamente.
 *
 * @param request - Dati completi del feedback con citazione corretta
 *
 * @example
 * const result = await submitNERFeedback({
 *   article_urn: "urn:nir:stato:codice.civile:1942;art1218",
 *   user_id: "user123",
 *   selected_text: "art. 1219 c.c.",
 *   start_offset: 42,
 *   end_offset: 57,
 *   context_window: "...come previsto dall'art. 1219 c.c. che...",
 *   feedback_type: "correction",
 *   original_parsed: { tipo_atto: "codice civile", articoli: ["1218"] },
 *   correct_reference: { tipo_atto: "codice civile", articoli: ["1219"] },
 *   confidence_before: 0.85,
 *   source: "citation_preview"
 * });
 *
 * if (result.training_triggered) {
 *   toast.success(`Training NER aggiornato con ${result.patterns_updated} pattern`);
 * }
 */
export async function submitNERFeedback(
  request: NERFeedbackRequest
): Promise<NERFeedbackResponse> {
  return post<NERFeedbackResponse>(
    `${MERLT_PREFIX}/enrichment/ner-feedback`,
    request
  );
}

/**
 * Conferma che una citazione è corretta.
 * Usato quando l'utente clicca "Corretto" sul preview della citazione.
 *
 * @param request - Dati citazione confermata
 *
 * @example
 * const result = await confirmCitation({
 *   article_urn: "urn:nir:stato:codice.civile:1942;art1218",
 *   text: "art. 1218 c.c.",
 *   parsed: { tipo_atto: "codice civile", articoli: ["1218"], confidence: 0.95 },
 *   user_id: "user123"
 * });
 *
 * console.log(result.buffer_size); // Numero feedback nel buffer
 */
export async function confirmCitation(
  request: NERConfirmRequest
): Promise<NERFeedbackResponse> {
  return post<NERFeedbackResponse>(
    `${MERLT_PREFIX}/enrichment/ner-feedback-confirm`,
    request
  );
}

// =============================================================================
// EXPORTS
// =============================================================================

export const merltService = {
  // Expert System
  queryExperts,
  submitInlineFeedback,
  submitDetailedFeedback,
  submitSourceFeedback,
  submitExpertPreferenceFeedback,  // R4: Divergent interpretation feedback
  submitRouterFeedback,            // F2: Router feedback (high-authority)
  // Enrichment
  checkArticleInGraph,
  requestLiveEnrichment,
  requestLiveEnrichmentStreaming,  // SSE streaming version
  // Validation
  getPendingQueue,
  validateEntity,
  validateRelation,
  // Deduplication
  checkEntityDuplicate,
  checkRelationDuplicate,
  // Proposals
  proposeEntity,
  proposeRelation,
  // Authority & Profile
  getUserAuthority,
  getUserProfile,
  // Entity Search (R3)
  searchEntities,
  // Norm Resolver (R5)
  resolveNorm,
  // RLCF Feedback (R2 + 7.2)
  submitEntitySelectionFeedback,
  // Issue Reporting (RLCF Feedback Loop)
  reportNodeIssue,
  voteOnIssue,
  getEntityIssues,
  getOpenIssues,
  // Graph
  getArticleEntities,
  getArticleRelations,
  getNodeDetails,
  getArticleSubgraph,
  getSubgraph,  // Knowledge Graph visualization
  getGraphOverview,  // Graph overview for bulletin board
  // Dossier Training Set (R5)
  exportDossierTrainingSet,
  loadDossierTrainingSet,
  // NER Feedback
  submitNERFeedback,
  confirmCitation,
  // Citation Export
  exportCitations,
  downloadCitationFile,
  triggerFileDownload,
};

export default merltService;
