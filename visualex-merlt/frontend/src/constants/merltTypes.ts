/**
 * MERL-T Types Constants
 * ======================
 *
 * Tipi completi di entità e relazioni del Knowledge Graph MERL-T.
 * Allineati a: merlt/pipeline/enrichment/models.py
 *
 * - 27 EntityType
 * - 65 RelationType (11 categorie)
 */

import type { EntityType, RelationType } from '../types/merlt';

// =============================================================================
// ENTITY TYPES (27 tipi)
// =============================================================================

export interface EntityTypeOption {
  value: EntityType;
  label: string;
  description: string;
  category: string;
}

export const ENTITY_TYPE_OPTIONS: EntityTypeOption[] = [
  // ─────────────────────────────────────────────────────────────────────────
  // FONTI NORMATIVE
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'norma', label: 'Norma', description: 'Disposizione normativa', category: 'Fonti Normative' },
  { value: 'versione', label: 'Versione', description: 'Versione temporale di una norma', category: 'Fonti Normative' },
  { value: 'direttiva_ue', label: 'Direttiva UE', description: 'Direttiva europea', category: 'Fonti Normative' },
  { value: 'regolamento_ue', label: 'Regolamento UE', description: 'Regolamento europeo', category: 'Fonti Normative' },

  // ─────────────────────────────────────────────────────────────────────────
  // STRUTTURA TESTUALE
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'comma', label: 'Comma', description: 'Suddivisione di articolo', category: 'Struttura Testuale' },
  { value: 'lettera', label: 'Lettera', description: 'Suddivisione di comma', category: 'Struttura Testuale' },
  { value: 'numero', label: 'Numero', description: 'Suddivisione di lettera', category: 'Struttura Testuale' },
  { value: 'definizione_legale', label: 'Definizione', description: 'Definizione contenuta nella norma', category: 'Struttura Testuale' },

  // ─────────────────────────────────────────────────────────────────────────
  // GIURISPRUDENZA E DOTTRINA
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'atto_giudiziario', label: 'Atto Giudiziario', description: 'Sentenza, ordinanza, decreto', category: 'Giurisprudenza' },
  { value: 'caso', label: 'Caso', description: 'Caso giuridico concreto', category: 'Giurisprudenza' },
  { value: 'dottrina', label: 'Dottrina', description: 'Opinione dottrinale', category: 'Giurisprudenza' },
  { value: 'precedente', label: 'Precedente', description: 'Massima giurisprudenziale', category: 'Giurisprudenza' },
  { value: 'brocardo', label: 'Brocardo', description: 'Massima latina (Pacta sunt servanda, etc.)', category: 'Giurisprudenza' },

  // ─────────────────────────────────────────────────────────────────────────
  // SOGGETTI E RUOLI
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'soggetto_giuridico', label: 'Soggetto', description: 'Persona fisica o giuridica', category: 'Soggetti' },
  { value: 'ruolo_giuridico', label: 'Ruolo', description: 'Ruolo ricoperto (creditore, debitore, etc.)', category: 'Soggetti' },
  { value: 'organo', label: 'Organo', description: 'Organo istituzionale', category: 'Soggetti' },

  // ─────────────────────────────────────────────────────────────────────────
  // CONCETTI GIURIDICI
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'concetto', label: 'Concetto', description: 'Concetto giuridico astratto', category: 'Concetti' },
  { value: 'principio', label: 'Principio', description: 'Principio fondamentale del diritto', category: 'Concetti' },
  { value: 'diritto_soggettivo', label: 'Diritto Soggettivo', description: 'Diritto soggettivo tutelato', category: 'Concetti' },
  { value: 'interesse_legittimo', label: 'Interesse Legittimo', description: 'Interesse legittimo', category: 'Concetti' },
  { value: 'responsabilita', label: 'Responsabilità', description: 'Tipo di responsabilità', category: 'Concetti' },

  // ─────────────────────────────────────────────────────────────────────────
  // DINAMICHE
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'fatto_giuridico', label: 'Fatto Giuridico', description: 'Evento che produce effetti giuridici', category: 'Dinamiche' },
  { value: 'procedura', label: 'Procedura', description: 'Sequenza di atti', category: 'Dinamiche' },
  { value: 'sanzione', label: 'Sanzione', description: 'Conseguenza per violazione', category: 'Dinamiche' },
  { value: 'termine', label: 'Termine', description: 'Scadenza temporale', category: 'Dinamiche' },

  // ─────────────────────────────────────────────────────────────────────────
  // LOGICA E REASONING
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'regola', label: 'Regola', description: 'Regola logica giuridica', category: 'Logica' },
  { value: 'proposizione', label: 'Proposizione', description: 'Proposizione normativa', category: 'Logica' },
  { value: 'modalita_giuridica', label: 'Modalità', description: 'Modalità deontica (permesso, obbligo, divieto)', category: 'Logica' },
];

// =============================================================================
// RELATION TYPES (65 tipi in 11 categorie)
// =============================================================================

export interface RelationTypeOption {
  value: RelationType;
  label: string;
  description: string;
  category: string;
}

export const RELATION_TYPE_OPTIONS: RelationTypeOption[] = [
  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI STRUTTURALI (5)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'CONTIENE', label: 'Contiene', description: 'Norma contiene Comma, Comma contiene Lettera', category: 'Strutturali' },
  { value: 'PARTE_DI', label: 'Parte di', description: 'Inverso di contiene', category: 'Strutturali' },
  { value: 'VERSIONE_PRECEDENTE', label: 'Versione precedente', description: 'Versione → Versione precedente', category: 'Strutturali' },
  { value: 'VERSIONE_SUCCESSIVA', label: 'Versione successiva', description: 'Versione → Versione successiva', category: 'Strutturali' },
  { value: 'HA_VERSIONE', label: 'Ha versione', description: 'Norma → Versione', category: 'Strutturali' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI DI MODIFICA (9)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'SOSTITUISCE', label: 'Sostituisce', description: 'Sostituzione testuale completa', category: 'Modifica' },
  { value: 'INSERISCE', label: 'Inserisce', description: 'Aggiunta senza rimozione', category: 'Modifica' },
  { value: 'ABROGA_TOTALMENTE', label: 'Abroga totalmente', description: 'Abrogazione completa', category: 'Modifica' },
  { value: 'ABROGA_PARZIALMENTE', label: 'Abroga parzialmente', description: 'Abrogazione di clausole', category: 'Modifica' },
  { value: 'SOSPENDE', label: 'Sospende', description: 'Sospensione temporanea efficacia', category: 'Modifica' },
  { value: 'PROROGA', label: 'Proroga', description: 'Estensione termini/validità', category: 'Modifica' },
  { value: 'INTEGRA', label: 'Integra', description: 'Integrazione senza sostituzione', category: 'Modifica' },
  { value: 'DEROGA_A', label: 'Deroga a', description: 'Eccezione senza modifica testo', category: 'Modifica' },
  { value: 'CONSOLIDA', label: 'Consolida', description: 'Testo unico da norme sparse', category: 'Modifica' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI SEMANTICHE (7)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'DISCIPLINA', label: 'Disciplina', description: 'Norma disciplina Concetto', category: 'Semantiche' },
  { value: 'APPLICA', label: 'Applica', description: 'Brocardo/Principio si applica ad Articolo', category: 'Semantiche' },
  { value: 'APPLICA_A', label: 'Si applica a', description: 'Norma si applica a Soggetto', category: 'Semantiche' },
  { value: 'DEFINISCE', label: 'Definisce', description: 'Norma definisce termine legale', category: 'Semantiche' },
  { value: 'PREVEDE_SANZIONE', label: 'Prevede sanzione', description: 'Norma prevede Sanzione', category: 'Semantiche' },
  { value: 'STABILISCE_TERMINE', label: 'Stabilisce termine', description: 'Norma stabilisce Termine', category: 'Semantiche' },
  { value: 'PREVEDE', label: 'Prevede', description: 'Norma prevede Procedura', category: 'Semantiche' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI DI DIPENDENZA (3)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'DIPENDE_DA', label: 'Dipende da', description: 'Dipendenza logica tra norme', category: 'Dipendenza' },
  { value: 'PRESUPPONE', label: 'Presuppone', description: 'Prerequisito implicito', category: 'Dipendenza' },
  { value: 'SPECIES', label: 'È tipo di', description: 'Relazione gerarchica is-a', category: 'Dipendenza' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI CITAZIONE/INTERPRETAZIONE (3)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'CITA', label: 'Cita', description: 'Citazione esplicita', category: 'Citazione' },
  { value: 'INTERPRETA', label: 'Interpreta', description: 'Interpretazione giudiziaria/dottrinale', category: 'Citazione' },
  { value: 'COMMENTA', label: 'Commenta', description: 'Commento dottrinale', category: 'Citazione' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI EUROPEE (3)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'ATTUA', label: 'Attua', description: 'Norma nazionale attua direttiva UE', category: 'Europee' },
  { value: 'RECEPISCE', label: 'Recepisce', description: 'Recepimento specifico direttiva', category: 'Europee' },
  { value: 'CONFORME_A', label: 'Conforme a', description: 'Conformità a standard UE', category: 'Europee' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI ISTITUZIONALI (3)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'EMESSO_DA', label: 'Emesso da', description: 'Atto emesso da Organo', category: 'Istituzionali' },
  { value: 'HA_COMPETENZA_SU', label: 'Ha competenza su', description: 'Organo ha competenza su materia', category: 'Istituzionali' },
  { value: 'GERARCHICAMENTE_SUPERIORE', label: 'Gerarchicamente superiore', description: 'Gerarchia organi', category: 'Istituzionali' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI CASE-BASED (3)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'RIGUARDA', label: 'Riguarda', description: 'Atto riguarda Soggetto/Caso', category: 'Case-Based' },
  { value: 'APPLICA_NORMA_A_CASO', label: 'Applica norma a caso', description: 'Applicazione giudiziaria', category: 'Case-Based' },
  { value: 'PRECEDENTE_DI', label: 'Precedente di', description: 'Precedente giurisprudenziale', category: 'Case-Based' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI CLASSIFICAZIONE (2)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'FONTE', label: 'Fonte', description: 'Norma ha fonte documento/codice', category: 'Classificazione' },
  { value: 'CLASSIFICA_IN', label: 'Classifica in', description: 'Classificazione tematica (EuroVoc)', category: 'Classificazione' },

  // ─────────────────────────────────────────────────────────────────────────
  // RELAZIONI LKIF - MODALITÀ E REASONING (28)
  // ─────────────────────────────────────────────────────────────────────────
  { value: 'IMPONE', label: 'Impone', description: 'Norma impone obbligo/divieto/permesso', category: 'Modalità' },
  { value: 'CONFERISCE', label: 'Conferisce', description: 'Norma conferisce diritto/potere', category: 'Modalità' },
  { value: 'TITOLARE_DI', label: 'Titolare di', description: 'Soggetto titolare di diritto/obbligo', category: 'Modalità' },
  { value: 'RIVESTE_RUOLO', label: 'Riveste ruolo', description: 'Soggetto assume ruolo giuridico', category: 'Modalità' },
  { value: 'ATTRIBUISCE_RESPONSABILITA', label: 'Attribuisce responsabilità', description: 'Attribuzione responsabilità', category: 'Modalità' },
  { value: 'RESPONSABILE_PER', label: 'Responsabile per', description: 'Soggetto responsabile per', category: 'Modalità' },
  { value: 'ESPRIME_PRINCIPIO', label: 'Esprime principio', description: 'Norma esprime principio', category: 'Principi' },
  { value: 'CONFORMA_A', label: 'Conforma a', description: 'Conformità a principio', category: 'Principi' },
  { value: 'DEROGA_PRINCIPIO', label: 'Deroga principio', description: 'Deroga eccezionale a principio', category: 'Principi' },
  { value: 'BILANCIA_CON', label: 'Bilancia con', description: 'Bilanciamento tra principi', category: 'Principi' },
  { value: 'PRODUCE_EFFETTO', label: 'Produce effetto', description: 'Fatto produce effetto giuridico', category: 'Effetti' },
  { value: 'PRESUPPOSTO_DI', label: 'Presupposto di', description: 'Fatto è presupposto per effetto', category: 'Effetti' },
  { value: 'COSTITUTIVO_DI', label: 'Costitutivo di', description: 'Fatto costituisce rapporto/status', category: 'Effetti' },
  { value: 'ESTINGUE', label: 'Estingue', description: 'Fatto estingue diritto/obbligo', category: 'Effetti' },
  { value: 'MODIFICA_EFFICACIA', label: 'Modifica efficacia', description: 'Fatto modifica efficacia', category: 'Effetti' },
  { value: 'APPLICA_REGOLA', label: 'Applica regola', description: 'Atto giudiziario applica regola', category: 'Reasoning' },
  { value: 'IMPLICA', label: 'Implica', description: 'Implicazione logica', category: 'Reasoning' },
  { value: 'CONTRADICE', label: 'Contraddizione tra proposizioni', category: 'Reasoning' },
  { value: 'GIUSTIFICA', label: 'Giustifica', description: 'Giustificazione/reasoning', category: 'Reasoning' },
  { value: 'LIMITA', label: 'Limita', description: 'Limitazione di diritti/poteri', category: 'Limitazioni' },
  { value: 'TUTELA', label: 'Tutela', description: 'Norma/Procedura tutela diritto', category: 'Tutela' },
  { value: 'VIOLA', label: 'Viola', description: 'Fatto viola norma/diritto', category: 'Violazioni' },
  { value: 'COMPATIBILE_CON', label: 'Compatibile con', description: 'Compatibilità tra norme/principi', category: 'Compatibilità' },
  { value: 'INCOMPATIBILE_CON', label: 'Incompatibile con', description: 'Incompatibilità', category: 'Compatibilità' },
  { value: 'SPECIFICA', label: 'Specifica', description: 'Specificazione astratto → concreto', category: 'Specificazione' },
  { value: 'ESEMPLIFICA', label: 'Esemplifica', description: 'Caso esemplifica concetto', category: 'Specificazione' },
  { value: 'CORRELATO', label: 'Correlato', description: 'Relazione generica', category: 'Generiche' },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Raggruppa le opzioni per categoria
 */
export function groupByCategory<T extends { category: string }>(options: T[]): Record<string, T[]> {
  return options.reduce((acc, opt) => {
    if (!acc[opt.category]) acc[opt.category] = [];
    acc[opt.category].push(opt);
    return acc;
  }, {} as Record<string, T[]>);
}

/**
 * Mappa EntityType -> Label per visualizzazione
 */
export const ENTITY_TYPE_LABELS: Record<string, string> = Object.fromEntries(
  ENTITY_TYPE_OPTIONS.map(opt => [opt.value, opt.label])
);

/**
 * Mappa RelationType -> Label per visualizzazione
 */
export const RELATION_TYPE_LABELS: Record<string, string> = Object.fromEntries(
  RELATION_TYPE_OPTIONS.map(opt => [opt.value, opt.label])
);
