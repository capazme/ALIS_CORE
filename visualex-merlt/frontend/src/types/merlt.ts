/**
 * MERL-T Integration Types
 * ========================
 *
 * TypeScript types for MERL-T backend integration.
 *
 * Aligned with:
 * - docs/archive/02-methodology/knowledge-graph.md (23 Node Types, 65 Relation Types)
 * - merlt/pipeline/enrichment/models.py
 */

// =============================================================================
// ENUMS
// =============================================================================

export type ValidationStatus = 'pending' | 'approved' | 'rejected' | 'needs_revision' | 'expired';
export type VoteType = 'approve' | 'reject' | 'edit';
export type SynthesisMode = 'convergent' | 'divergent';

// =============================================================================
// ENTITY TYPES (23 types from knowledge-graph.md)
// =============================================================================

/**
 * Tipi di entità nel Knowledge Graph MERL-T (23 tipi)
 *
 * Tutti i tipi sono potenzialmente estraibili da LLM.
 * Se l'LLM incontra un riferimento nuovo (es. Direttiva UE),
 * può creare uno stub da arricchire successivamente.
 *
 * From: docs/archive/02-methodology/knowledge-graph.md
 */
export type EntityType =
  // Fonti Normative
  | 'norma'
  | 'versione'
  | 'direttiva_ue'
  | 'regolamento_ue'
  // Struttura Testuale
  | 'comma'
  | 'lettera'
  | 'numero'
  | 'definizione_legale'
  // Giurisprudenza e Dottrina
  | 'atto_giudiziario'
  | 'caso'
  | 'dottrina'
  | 'precedente'
  | 'brocardo'
  // Soggetti e Ruoli
  | 'soggetto_giuridico'
  | 'ruolo_giuridico'
  | 'organo'
  // Concetti Giuridici
  | 'concetto'
  | 'principio'
  | 'diritto_soggettivo'
  | 'interesse_legittimo'
  | 'responsabilita'
  // Dinamiche
  | 'fatto_giuridico'
  | 'procedura'
  | 'sanzione'
  | 'termine'
  // Logica e Reasoning
  | 'regola'
  | 'proposizione'
  | 'modalita_giuridica';

// =============================================================================
// RELATION TYPES (65 types from knowledge-graph.md)
// =============================================================================

/**
 * All relation types in MERL-T Knowledge Graph (65 types)
 * From: docs/archive/02-methodology/knowledge-graph.md
 */
export type RelationType =
  // Structural Relations (5)
  | 'CONTIENE'
  | 'PARTE_DI'
  | 'VERSIONE_PRECEDENTE'
  | 'VERSIONE_SUCCESSIVA'
  | 'HA_VERSIONE'
  // Modification Relations (9)
  | 'SOSTITUISCE'
  | 'INSERISCE'
  | 'ABROGA_TOTALMENTE'
  | 'ABROGA_PARZIALMENTE'
  | 'SOSPENDE'
  | 'PROROGA'
  | 'INTEGRA'
  | 'DEROGA_A'
  | 'CONSOLIDA'
  // Semantic Relations (7)
  | 'DISCIPLINA'
  | 'APPLICA'
  | 'APPLICA_A'
  | 'DEFINISCE'
  | 'PREVEDE_SANZIONE'
  | 'STABILISCE_TERMINE'
  | 'PREVEDE'
  // Dependency Relations (3)
  | 'DIPENDE_DA'
  | 'PRESUPPONE'
  | 'SPECIES'
  // Citation & Interpretation (3)
  | 'CITA'
  | 'INTERPRETA'
  | 'COMMENTA'
  // European Relations (3)
  | 'ATTUA'
  | 'RECEPISCE'
  | 'CONFORME_A'
  // Institutional Relations (3)
  | 'EMESSO_DA'
  | 'HA_COMPETENZA_SU'
  | 'GERARCHICAMENTE_SUPERIORE'
  // Case-based Relations (3)
  | 'RIGUARDA'
  | 'APPLICA_NORMA_A_CASO'
  | 'PRECEDENTE_DI'
  // Classification (2)
  | 'FONTE'
  | 'CLASSIFICA_IN'
  // LKIF Modalities & Reasoning (28)
  | 'IMPONE'
  | 'CONFERISCE'
  | 'TITOLARE_DI'
  | 'RIVESTE_RUOLO'
  | 'ATTRIBUISCE_RESPONSABILITA'
  | 'RESPONSABILE_PER'
  | 'ESPRIME_PRINCIPIO'
  | 'CONFORMA_A'
  | 'DEROGA_PRINCIPIO'
  | 'BILANCIA_CON'
  | 'PRODUCE_EFFETTO'
  | 'PRESUPPOSTO_DI'
  | 'COSTITUTIVO_DI'
  | 'ESTINGUE'
  | 'MODIFICA_EFFICACIA'
  | 'APPLICA_REGOLA'
  | 'IMPLICA'
  | 'CONTRADICE'
  | 'GIUSTIFICA'
  | 'LIMITA'
  | 'TUTELA'
  | 'VIOLA'
  | 'COMPATIBILE_CON'
  | 'INCOMPATIBILE_CON'
  | 'SPECIFICA'
  | 'ESEMPLIFICA'
  | 'CAUSA_DI'
  | 'CONDIZIONE_DI'
  // Generic fallback
  | 'CORRELATO';

// =============================================================================
// EXPERT SYSTEM (Q&A)
// =============================================================================

export interface ExpertQueryRequest {
  query: string;
  user_id: string;
  context?: Record<string, unknown>;
  max_experts?: number;
  include_trace?: boolean;
  consent_level?: 'anonymous' | 'basic' | 'full';
}

export interface SourceReference {
  article_urn: string;
  expert: string;
  relevance: number;
  excerpt?: string;
}

export interface ExpertQueryResponse {
  trace_id: string;
  synthesis: string;
  mode: SynthesisMode;
  alternatives?: Array<Record<string, unknown>>;
  sources: SourceReference[];
  experts_used: string[];
  confidence: number;
  execution_time_ms: number;
}

export interface InlineFeedbackRequest {
  trace_id: string;
  user_id: string;
  rating: 1 | 2 | 3 | 4 | 5;
}

export interface DetailedFeedbackRequest {
  trace_id: string;
  user_id: string;
  accuracy: 1 | 2 | 3 | 4 | 5;
  completeness: 1 | 2 | 3 | 4 | 5;
  relevance: 1 | 2 | 3 | 4 | 5;
  comment?: string;
}

export interface SourceFeedbackRequest {
  trace_id: string;
  user_id: string;
  source_urn: string;
  rating: 1 | 2 | 3 | 4 | 5;
  comment?: string;
}

/**
 * Request per feedback su interpretazioni divergenti.
 *
 * Quando mode=divergent, l'utente puo' indicare quale expert
 * ha fornito l'interpretazione piu' utile.
 */
export interface ExpertPreferenceFeedbackRequest {
  trace_id: string;
  user_id: string;
  /** Expert preferito (literal, systemic, principles, precedent) */
  preferred_expert: string;
  /** Commento opzionale con motivazione */
  comment?: string;
}

/** Router feedback request (F2) — only for users with authority >= 0.7 */
export interface RouterFeedbackRequest {
  trace_id: string;
  user_id: string;
  /** Whether the routing decision was appropriate */
  routing_correct: boolean;
  /** Suggested expert weights if routing was incorrect */
  suggested_weights?: Record<string, number>;
  /** Alternative query type classification */
  suggested_query_type?: string;
  comment?: string;
}

/**
 * Alternativa divergente da un Expert.
 * Usata in ExpertCompareView quando mode='divergent'.
 */
export interface ExpertAlternative {
  /** Tipo expert (literal, systemic, principles, precedent) */
  expert: string;
  /** Interpretazione dell'expert (max 500 chars) */
  position: string;
  /** Confidenza [0-1] */
  confidence: number;
  /** Citazioni delle fonti giuridiche */
  legal_basis: string[];
  /** Descrizione del tipo di ragionamento */
  reasoning_type: string;
}

// =============================================================================
// ENRICHMENT & VALIDATION
// =============================================================================

export interface PendingEntity {
  id: string;
  nome: string;
  tipo: EntityType;
  descrizione: string;
  articoli_correlati: string[];
  ambito: string;
  fonte: string;
  llm_confidence: number;
  raw_context: string;
  validation_status: ValidationStatus;
  approval_score: number;
  rejection_score: number;
  votes_count: number;
  contributed_by: string;
  contributor_authority: number;
  created_at: string;
}

export interface PendingRelation {
  id: string;
  source_urn: string;
  target_urn: string;
  relation_type: RelationType;
  fonte: string;
  llm_confidence: number;
  evidence: string;
  validation_status: ValidationStatus;
  approval_score: number;
  rejection_score: number;
  votes_count: number;
  contributed_by: string;
  contributor_authority: number;
  created_at: string;
}

export interface EntityValidationRequest {
  entity_id: string;
  user_id: string;
  vote: VoteType;
  comment?: string;
  suggested_edit?: Partial<PendingEntity>;
}

export interface EntityValidationResponse {
  entity_id: string;
  new_status: ValidationStatus;
  approval_score: number;
  rejection_score: number;
  votes_count: number;
  vote_weight: number;
  threshold_reached: boolean;
}

export interface RelationValidationRequest {
  relation_id: string;
  user_id: string;
  vote: VoteType;
  comment?: string;
  suggested_edit?: Partial<PendingRelation>;
}

export interface RelationValidationResponse {
  relation_id: string;
  new_status: ValidationStatus;
  approval_score: number;
  rejection_score: number;
  votes_count: number;
  vote_weight: number;
  threshold_reached: boolean;
}

export interface PendingQueueResponse {
  pending_entities: PendingEntity[];
  pending_relations: PendingRelation[];
  total_entities: number;
  total_relations: number;
}

// =============================================================================
// GRAPH NAVIGATION
// =============================================================================

export interface GraphNode {
  id: string;
  label: string;
  type: EntityType;
  status: ValidationStatus;
  confidence: number;
  properties?: Record<string, unknown>;
}

export interface GraphLink {
  source: string;
  target: string;
  type: RelationType;
  status: ValidationStatus;
  confidence: number;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface ArticleGraphStatus {
  exists: boolean;
  node_id?: string;
  pending_validation?: boolean;
  entity_count?: number;
}

export interface NodeDetails {
  node: Record<string, unknown>;
  relations: Array<{
    type: string;
    target_id: string;
    target_label: string;
  }>;
}

// =============================================================================
// USER AUTHORITY (RLCF Framework)
// =============================================================================

/**
 * Breakdown dei componenti dell'authority secondo formula RLCF:
 * A_u(t) = α·B_u + β·T_u + γ·P_u
 *
 * Con pesi default: α=0.3, β=0.5, γ=0.2
 */
export interface AuthorityBreakdown {
  /** B_u - Baseline credentials da qualifiche professionali (peso 0.3) */
  baseline: number;
  /** T_u - Track Record storico pesato esponenzialmente (peso 0.5) */
  track_record: number;
  /** P_u - Performance recente / Level Authority (peso 0.2) */
  level_authority: number;
}

/**
 * Authority utente nel sistema RLCF.
 *
 * L'authority determina il peso del voto:
 * - Voto di utente con authority 0.8 vale 0.8
 * - Threshold approvazione: Σ(weighted_votes) >= 2.0
 */
export interface UserAuthority {
  user_id: string;
  /** Authority globale calcolata [0-1] */
  authority_score: number;
  /** Breakdown della formula RLCF */
  authority_breakdown?: AuthorityBreakdown;
  /** Authority per dominio giuridico (civile, penale, amministrativo, etc.) */
  domain_authorities: Record<string, number>;
  /** Totale contributi (entità + relazioni proposte) */
  total_contributions: number;
  /** Contributi approvati dalla community */
  approved_contributions: number;
  /** Ultimo aggiornamento ISO */
  last_updated: string;
}

// =============================================================================
// NORM RESOLVER (R5)
// =============================================================================

/**
 * Request per risolvere una citazione normativa.
 *
 * Usato in ProposeRelationDrawer quando l'utente inserisce una norma
 * come target (es. "Art. 1218 c.c.") invece di un'entità esistente.
 */
export interface NormResolveRequest {
  /** Tipo atto normalizzato (es. "codice civile", "legge", "d.lgs.") */
  act_type: string;
  /** Numero articolo (es. "1218", "52-bis") */
  article: string;
  /** Numero atto (es. "241" per L. 241/1990) */
  act_number?: string;
  /** Anno o data atto (es. "1990", "2016-05-04") */
  date?: string;
  /** URN dell'articolo da cui parte la relazione */
  source_article_urn: string;
  /** UUID utente VisuaLex */
  user_id: string;
}

/**
 * Response con risultato risoluzione norma.
 *
 * Possibili scenari:
 * 1. Norma esiste nel grafo → resolved=true, exists_in_graph=true
 * 2. Norma non esiste → crea PendingEntity tipo=norma → resolved=true, created_pending=true
 * 3. Errore parsing/risoluzione → resolved=false, error_message valorizzato
 */
export interface NormResolveResponse {
  /** True se risoluzione riuscita */
  resolved: boolean;
  /** ID entità da usare in proposeRelation (entity_id o node_id) */
  entity_id: string;
  /** Label per UI (es. "Art. 1218 Codice Civile") */
  display_label: string;
  /** URN risolto */
  urn: string;
  /** True se la norma era già nel grafo (approvata) */
  exists_in_graph: boolean;
  /** True se la norma è un'entità pending */
  is_pending: boolean;
  /** True se è stata creata una nuova PendingEntity */
  created_pending: boolean;
  /** Messaggio errore se resolved=false */
  error_message?: string;
}

// =============================================================================
// DEDUPLICATION
// =============================================================================

export type DuplicateConfidence = 'exact' | 'high' | 'medium' | 'low';

export interface DuplicateCandidate {
  entity_id: string;
  entity_text: string;
  entity_type: string;
  descrizione?: string;
  article_urn: string;
  similarity_score: number;
  confidence: DuplicateConfidence;
  match_reason: 'exact_normalized' | 'fuzzy_name' | 'fuzzy_description';
  validation_status: ValidationStatus;
  votes_count: number;
  net_score: number;
}

export interface DuplicateCheckRequest {
  entity_text: string;
  entity_type: EntityType;
  article_urn?: string;
  scope?: 'global' | 'article' | 'type';
}

export interface DuplicateCheckResponse {
  query_text: string;
  query_type: string;
  normalized_query: string;
  has_duplicates: boolean;
  exact_match?: DuplicateCandidate;
  duplicates: DuplicateCandidate[];
  recommendation: 'create' | 'merge' | 'ask_user';
}

export interface EntityProposalRequest {
  article_urn: string;
  nome: string;
  tipo: EntityType;
  descrizione: string;
  ambito?: string;
  evidence?: string;
  user_id: string;
  skip_duplicate_check?: boolean;
  acknowledged_duplicate_of?: string;
}

export interface EntityProposalResponse {
  success: boolean;
  pending_entity?: PendingEntity;
  message: string;
  has_duplicates: boolean;
  duplicates: DuplicateCandidate[];
  duplicate_action_required: boolean;
}

// Relation Deduplication
export interface RelationDuplicateCandidate {
  relation_id: string;
  source_text: string;
  target_text: string;
  relation_type: string;
  similarity_score: number;
  confidence: DuplicateConfidence;
  validation_status: ValidationStatus;
}

export interface RelationDuplicateCheckRequest {
  source_entity_id: string;
  target_entity_id: string;
  relation_type: RelationType;
}

export interface RelationDuplicateCheckResponse {
  has_duplicates: boolean;
  exact_match?: RelationDuplicateCandidate;
  duplicates: RelationDuplicateCandidate[];
  recommendation: 'create' | 'merge' | 'ask_user';
}

export interface RelationProposalRequest {
  tipo_relazione: RelationType;
  source_urn: string;
  target_entity_id: string;
  article_urn: string;
  descrizione: string;
  certezza: number;
  user_id: string;
  skip_duplicate_check?: boolean;
  acknowledged_duplicate_of?: string;
}

export interface RelationProposalResponse {
  success: boolean;
  relation_id?: string;
  message: string;
  has_duplicates: boolean;
  duplicates: RelationDuplicateCandidate[];
  duplicate_action_required: boolean;
}


// =============================================================================
// USER PROFILE (RLCF)
// =============================================================================

/**
 * Tier utente nel sistema RLCF.
 * I tier determinano il livello di reputazione dell'utente.
 */
export type AuthorityTier = 'novizio' | 'contributore' | 'esperto' | 'autorita';

/**
 * 8 Domini legali italiani nel sistema MERL-T.
 * Ogni utente può avere authority diversa per dominio.
 */
export type LegalDomain =
  | 'civile'
  | 'penale'
  | 'amministrativo'
  | 'costituzionale'
  | 'lavoro'
  | 'commerciale'
  | 'tributario'
  | 'internazionale';

/**
 * Statistiche per singolo dominio legale.
 */
export interface DomainStats {
  /** Authority nel dominio [0-1] */
  authority: number;
  /** Numero contributi nel dominio */
  contributions: number;
  /** Percentuale successo [0-100] */
  success_rate: number;
}

/**
 * Entry nell'activity feed dell'utente.
 */
export interface ProfileActivityEntry {
  /** ID univoco attività */
  id?: string;
  /** Tipo di attività */
  type: 'vote' | 'proposal' | 'edit' | 'ner_feedback';
  /** Nome dell'item (entità, relazione, o citazione) */
  item_name: string;
  /** Tipo item */
  item_type: 'entity' | 'relation' | 'citation';
  /** Esito dell'attività */
  outcome: 'approved' | 'rejected' | 'pending';
  /** Timestamp ISO */
  timestamp: string;
  /** Variazione del track record causata da questa attività */
  track_record_delta?: number;
  /** Dominio legale coinvolto */
  domain?: LegalDomain;
  /** ID dell'item per link */
  item_id?: string;
}

/**
 * Response completa del profilo utente.
 *
 * Contiene tutti i dati necessari per visualizzare:
 * - Authority RLCF con breakdown
 * - Progress verso prossimo tier
 * - Statistiche contributi
 * - Authority per dominio
 * - Attività recente
 */
export interface ProfileResponse {
  user_id: string;
  /** Nome visualizzato utente */
  display_name?: string;
  /** Authority completa */
  authority: {
    /** Score calcolato A_u [0-1] */
    score: number;
    /** Tier attuale */
    tier: AuthorityTier;
    /** Breakdown formula RLCF */
    breakdown: AuthorityBreakdown;
    /** Threshold del prossimo tier */
    next_tier_threshold: number;
    /** Progresso verso prossimo tier [0-100] */
    progress_to_next: number;
  };
  /** Authority e stats per ciascuno degli 8 domini */
  domains: Record<LegalDomain, DomainStats>;
  /** Statistiche globali contributi */
  stats: {
    total_contributions: number;
    approved: number;
    rejected: number;
    pending: number;
    /** Peso voto attuale (= authority.score) */
    vote_weight: number;
  };
  /** Ultime 10 attività */
  recent_activity: ProfileActivityEntry[];
  /** Data iscrizione ISO */
  joined_at: string;
  /** Ultimo aggiornamento authority ISO */
  last_updated: string;
}

// =============================================================================
// SUBGRAPH API (Knowledge Graph Visualization)
// =============================================================================

/**
 * Nodo nel subgraph restituito dall'API /api/v1/graph/subgraph.
 * Usato per la visualizzazione del Knowledge Graph.
 */
export interface SubgraphNode {
  /** ID univoco del nodo (tipicamente URN o entity_id) */
  id: string;
  /** URN se disponibile */
  urn?: string;
  /** Tipo nodo dal grafo (Norma, Entity, Comma, etc.) */
  type: string;
  /** Label per visualizzazione UI */
  label: string;
  /** Proprietà raw del nodo */
  properties: Record<string, unknown>;
  /** Metadati aggiuntivi (validation_status, confidence, etc.) */
  metadata: Record<string, unknown>;
}

/**
 * Arco/relazione nel subgraph.
 */
export interface SubgraphEdge {
  /** ID univoco dell'arco */
  id: string;
  /** ID nodo sorgente */
  source: string;
  /** ID nodo destinazione */
  target: string;
  /** Tipo relazione (DISCIPLINA, CITA, etc.) */
  type: string;
  /** Proprietà dell'arco */
  properties: Record<string, unknown>;
}

/**
 * Metadati della query subgraph.
 */
export interface SubgraphMetadata {
  root_urn: string;
  depth: number;
  query_time_ms: number;
  filtered_by_types: boolean;
  entity_types_filter?: string[];
  relation_types_filter?: string[];
}

/**
 * Response completa dell'API subgraph.
 */
export interface SubgraphResponse {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
  metadata: SubgraphMetadata;
}

/**
 * Parametri per la richiesta subgraph.
 */
export interface SubgraphRequest {
  /** URN del nodo radice da cui espandere il subgraph */
  root_urn: string;
  /** Profondità di esplorazione (default 2) */
  depth?: number;
  /** Numero massimo di nodi (default 50) */
  max_nodes?: number;
  /** Filtra per tipi di entità */
  entity_types?: string[];
  /** Filtra per tipi di relazione */
  relation_types?: string[];
  /** Include metadati extra nei nodi */
  include_metadata?: boolean;
}

/**
 * Content type per Graph Tab nel Workspace.
 * Usato per visualizzare il Knowledge Graph Explorer in una tab.
 */
export interface GraphTabContent {
  /** Tipo content identificatore */
  type: 'graph';
  /** ID univoco content */
  id: string;
  /** URN del nodo radice */
  rootUrn: string;
  /** Profondità di esplorazione (1-3) */
  depth?: number;
  /** Nodo attualmente selezionato (opzionale) */
  selectedNode?: SubgraphNode | null;
}

// =============================================================================
// ISSUE REPORTING (RLCF Feedback Loop)
// =============================================================================

/**
 * Tipi di issue segnalabili per entità e relazioni.
 *
 * ERRORI: Problemi fattuali che richiedono correzione
 * SUGGERIMENTI: Miglioramenti proposti
 */
export type IssueType =
  // Errori fattuali
  | 'factual_error'      // Dati errati
  | 'wrong_relation'     // Relazione sbagliata
  | 'wrong_type'         // Tipo entità errato
  | 'duplicate'          // Nodo duplicato
  | 'outdated'           // Info non aggiornata
  // Suggerimenti
  | 'missing_relation'   // Manca relazione
  | 'incomplete'         // Proprietà mancanti
  | 'improve_label'      // Label migliorabile
  | 'other';             // Altro

/**
 * Gravità dell'issue segnalata.
 */
export type IssueSeverity = 'low' | 'medium' | 'high';

/**
 * Stato di una issue.
 */
export type IssueStatus =
  | 'open'              // Aperta, in attesa di voti
  | 'threshold_reached' // Soglia raggiunta, entity in needs_revision
  | 'dismissed'         // Dismissata (troppi downvote)
  | 'resolved';         // Risolta manualmente

/**
 * Request per segnalare un problema su un'entità.
 */
export interface ReportIssueRequest {
  entity_id: string;
  issue_type: IssueType;
  severity: IssueSeverity;
  description: string;
  user_id: string;
}

/**
 * Response dopo creazione issue.
 */
export interface ReportIssueResponse {
  success: boolean;
  issue_id: string;
  entity_id: string;
  status: 'created' | 'merged';
  message: string;
  merged_with?: string;
}

/**
 * Request per votare su una issue esistente.
 */
export interface VoteIssueRequest {
  issue_id: string;
  vote: 'upvote' | 'downvote';
  comment?: string;
  user_id: string;
}

/**
 * Response dopo voto su issue.
 */
export interface VoteIssueResponse {
  success: boolean;
  issue_id: string;
  new_status: IssueStatus;
  upvote_score: number;
  downvote_score: number;
  votes_count: number;
  entity_reopened: boolean;
  message: string;
}

/**
 * Dettagli dell'entità dal Knowledge Graph per contestualizzare l'issue.
 */
export interface EntityDetailsForIssue {
  // Per nodi
  label?: string;
  node_type?: string;
  urn?: string;
  ambito?: string;
  properties?: Record<string, unknown>;

  // Per relazioni
  is_relation: boolean;
  relation_type?: string;
  source_label?: string;
  source_type?: string;
  target_label?: string;
  target_type?: string;
}

/**
 * Dati di una issue per visualizzazione.
 */
export interface EntityIssue {
  issue_id: string;
  entity_id: string;
  entity_type?: string;
  issue_type: IssueType;
  severity: IssueSeverity;
  description: string;
  status: IssueStatus;
  reported_by: string;
  reporter_authority: number;
  upvote_score: number;
  downvote_score: number;
  votes_count: number;
  created_at: string;
  resolved_at?: string;
  resolution_notes?: string;
  entity_details?: EntityDetailsForIssue;
}

/**
 * Response con lista issues per un'entità.
 */
export interface GetEntityIssuesResponse {
  entity_id: string;
  issues: EntityIssue[];
  open_count: number;
  total_count: number;
}

/**
 * Request per recuperare issue aperte (moderatori).
 */
export interface OpenIssuesRequest {
  status?: IssueStatus;
  severity?: IssueSeverity;
  issue_type?: IssueType;
  offset?: number;
  limit?: number;
}

/**
 * Response con lista issue aperte.
 */
export interface OpenIssuesResponse {
  issues: EntityIssue[];
  total: number;
  has_more: boolean;
}

/**
 * Mappatura tipi issue per UI.
 */
export const ISSUE_TYPE_LABELS: Record<IssueType, { label: string; category: 'error' | 'suggestion' }> = {
  factual_error: { label: 'Errore fattuale', category: 'error' },
  wrong_relation: { label: 'Relazione sbagliata', category: 'error' },
  wrong_type: { label: 'Tipo entità errato', category: 'error' },
  duplicate: { label: 'Nodo duplicato', category: 'error' },
  outdated: { label: 'Info non aggiornata', category: 'error' },
  missing_relation: { label: 'Manca una relazione', category: 'suggestion' },
  incomplete: { label: 'Informazioni incomplete', category: 'suggestion' },
  improve_label: { label: 'Label migliorabile', category: 'suggestion' },
  other: { label: 'Altro', category: 'suggestion' },
};

/**
 * Mappatura gravità per UI.
 */
export const ISSUE_SEVERITY_LABELS: Record<IssueSeverity, { label: string; color: string }> = {
  low: { label: 'Bassa', color: 'text-blue-600' },
  medium: { label: 'Media', color: 'text-amber-600' },
  high: { label: 'Alta', color: 'text-red-600' },
};

// =============================================================================
// DOSSIER TRAINING SET (R5)
// =============================================================================

/**
 * Articolo nel dossier per export training set.
 */
export interface DossierArticleData {
  urn?: string;
  tipo_atto: string;
  numero_atto?: string;
  numero_articolo: string;
  data?: string;
  article_text?: string;
  user_status?: 'unread' | 'reading' | 'important' | 'done';
}

// =============================================================================
// NER FEEDBACK (Citation Correction)
// =============================================================================

/**
 * Dati parsed di una citazione rilevata.
 */
export interface ParsedCitationData {
  tipo_atto: string;
  numero_atto?: string;
  anno?: string;
  articoli: string[];
  confidence?: number;
}

/**
 * Riferimento normativo corretto dall'utente.
 */
export interface CorrectNormReference {
  tipo_atto: string;
  numero_atto?: string;
  anno?: string;
  articoli: string[];
}

/**
 * Request per feedback NER su citazione.
 *
 * NOTA: start_offset e end_offset sono opzionali perché
 * il contesto viene estratto tramite DOM walking, che è
 * più affidabile degli offset HTML (che non corrispondono
 * agli offset nel testo plain).
 */
export interface NERFeedbackRequest {
  article_urn: string;
  user_id: string;
  selected_text: string;
  /** @deprecated Usare context_window invece */
  start_offset?: number;
  /** @deprecated Usare context_window invece */
  end_offset?: number;
  context_window: string;
  feedback_type: 'correction' | 'confirmation' | 'annotation';
  original_parsed?: ParsedCitationData;
  correct_reference: CorrectNormReference;
  confidence_before?: number;
  source: 'citation_preview' | 'selection_popup';
}

/**
 * Response dopo feedback NER.
 */
export interface NERFeedbackResponse {
  success: boolean;
  feedback_id: string;
  buffer_size: number;
  training_ready: boolean;
  training_triggered: boolean;
  patterns_updated: number;
  message: string;
}

/**
 * Request semplificata per conferma citazione corretta.
 */
export interface NERConfirmRequest {
  article_urn: string;
  text: string;
  parsed: ParsedCitationData;
  user_id: string;
}

/**
 * Sessione Q&A collegata al dossier.
 */
export interface DossierQASessionData {
  trace_id: string;
  query: string;
  synthesis: string;
  mode: string;
  experts_used: string[];
  confidence: number;
  feedback?: {
    inline_rating?: number;
    retrieval_score?: number;
    reasoning_score?: number;
    synthesis_score?: number;
    preferred_expert?: string;
  };
  created_at: string;
}

/**
 * Annotazione utente su un articolo.
 */
export interface DossierAnnotationData {
  article_urn?: string;
  article_numero: string;
  user_note?: string;
  highlight_text?: string;
  annotation_type: string;
  created_at?: string;
}

/**
 * Request per export dossier come training set.
 */
export interface DossierTrainingSetExportRequest {
  dossier_id: string;
  user_id: string;
  include_qa_sessions?: boolean;
  include_annotations?: boolean;
  include_article_text?: boolean;
  format?: 'json';
}

/**
 * Request per export completo con dati dossier.
 */
export interface DossierTrainingSetExportFullRequest {
  dossier_title: string;
  dossier_description?: string;
  dossier_tags?: string[];
  articles: DossierArticleData[];
  user_id: string;
  include_qa_sessions?: boolean;
}

/**
 * Response con training set esportato.
 */
export interface DossierTrainingSetExportResponse {
  training_set_id: string;
  dossier_id: string;
  dossier_title: string;
  dossier_description?: string;
  dossier_tags: string[];
  articles: DossierArticleData[];
  qa_sessions: DossierQASessionData[];
  annotations: DossierAnnotationData[];
  exported_at: string;
  exported_by: string;
  articles_count: number;
  qa_sessions_count: number;
  annotations_count: number;
  completed_articles: number;
  avg_qa_confidence: number;
}

/**
 * Request per caricare training set nel buffer RLCF.
 */
export interface LoadDossierTrainingRequest {
  training_set: DossierTrainingSetExportResponse;
  priority_boost?: number;  // 0.0 - 1.0
}

/**
 * Response dopo caricamento training set.
 */
export interface LoadDossierTrainingResponse {
  success: boolean;
  experiences_added: number;
  buffer_size: number;
  training_ready: boolean;
  message: string;
}

// =============================================================================
// NER TRAINING FEEDBACK (Citation Recognition)
// =============================================================================

/**
 * Dati di una citazione normativa parsed automaticamente.
 */
export interface ParsedCitationData {
  /** Tipo atto (es. "codice civile", "legge") */
  actType?: string;
  /** Numero atto (es. "241") */
  actNumber?: string;
  /** Anno o data (es. "1990", "1990-08-07") */
  date?: string;
  /** Articoli citati (es. ["3", "4", "5"]) */
  articles?: string[];
}

/**
 * Riferimento normativo corretto fornito dall'utente.
 */
export interface CorrectNormReference {
  /** Tipo atto normalizzato */
  tipo_atto: string;
  /** Numero atto (opzionale) */
  numero_atto?: string;
  /** Anno (opzionale) */
  anno?: string;
  /** Articoli citati */
  articoli: string[];
}

// =============================================================================
// CITATION EXPORT
// =============================================================================

export type CitationFormat = 'italian_legal' | 'bibtex' | 'plain_text' | 'json';

export interface CitationExportRequest {
  trace_id?: string;
  sources?: Array<{
    article_urn: string;
    expert?: string;
    relevance?: number;
    title?: string;
    source_type?: string;
  }>;
  format: CitationFormat;
  include_query_summary?: boolean;
  include_attribution?: boolean;
}

export interface CitationExportResponse {
  success: boolean;
  format: CitationFormat;
  filename: string;
  download_url: string;
  citations_count: number;
  alis_version: string;
  generated_at: string;
  message?: string;
}

// =============================================================================
// API ERROR
// =============================================================================

export interface MerltApiError {
  status: number;
  message: string;
  detail?: string;
}
