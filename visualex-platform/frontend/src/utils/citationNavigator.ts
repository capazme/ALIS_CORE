/**
 * Citation Navigator - Utilities for navigating to citations and detecting external citations.
 */

import type { ParsedCitationData } from './citationMatcher';

/** Well-known act types that are searchable on Normattiva */
const NORMATTIVA_ACT_TYPES = new Set([
  'legge', 'decreto legge', 'decreto legislativo',
  'decreto del presidente della repubblica', 'regio decreto',
  'codice civile', 'codice penale',
  'codice di procedura civile', 'codice di procedura penale',
  'costituzione', 'preleggi',
  'codice della strada', 'codice della navigazione',
  "codice dell'amministrazione digitale",
  'codice antimafia', 'norme in materia ambientale',
  'codice delle assicurazioni private',
  'codice dei beni culturali e del paesaggio',
  'codice delle comunicazioni elettroniche',
  "codice della crisi d'impresa e dell'insolvenza",
  'codice dei contratti pubblici', 'codice del consumo',
  'codice della protezione civile',
  'codice di giustizia contabile',
  'codice della nautica da diporto',
  "codice dell'ordinamento militare",
  'codice del processo amministrativo',
  'codice in materia di protezione dei dati personali',
  'codice postale e delle telecomunicazioni',
  'codice della proprietà industriale',
  'codice delle pari opportunità',
  'codice del processo tributario',
  'codice del Terzo settore', 'codice del turismo',
  "disposizioni per l'attuazione del Codice civile e disposizioni transitorie",
  "disposizioni per l'attuazione del Codice di procedura civile e disposizioni transitorie",
  "disposizioni per l'attuazione del Codice di procedura penale",
  'disposizioni attuative',
]);

/** EU act types that link to EUR-Lex */
const EU_ACT_TYPES = new Set([
  'Regolamento UE', 'Direttiva UE', 'TUE', 'TFUE', 'CDFUE',
]);

export type CitationType = 'internal' | 'external-eu' | 'external-unknown';

/**
 * Determines the citation type based on the parsed citation data.
 */
export function getCitationType(parsed: ParsedCitationData): CitationType {
  const actType = parsed.act_type.toLowerCase();

  if (EU_ACT_TYPES.has(parsed.act_type) || actType.includes('regolamento ue') || actType.includes('direttiva ue')) {
    return 'external-eu';
  }

  if (NORMATTIVA_ACT_TYPES.has(actType)) {
    return 'internal';
  }

  return 'external-unknown';
}

/**
 * Check if a citation is external (not navigable in-app).
 */
export function isExternalCitation(parsed: ParsedCitationData): boolean {
  return getCitationType(parsed) !== 'internal';
}

/**
 * Get external URL for a citation (EUR-Lex, etc.).
 */
export function getExternalUrl(parsed: ParsedCitationData): string | null {
  const type = getCitationType(parsed);

  if (type === 'external-eu') {
    // EUR-Lex search URL
    const query = encodeURIComponent(`${parsed.act_type} ${parsed.act_number || ''} ${parsed.date || ''} Art. ${parsed.article}`);
    return `https://eur-lex.europa.eu/search.html?textScope=ti-te&qid=&DTS_DOM=ALL&type=advanced&text=${query}`;
  }

  return null;
}

/**
 * Navigate to a citation. Returns true if handled in-app, false if external.
 */
export function navigateToCitation(
  parsed: ParsedCitationData,
  callbacks: {
    onInternalNavigate: (parsed: ParsedCitationData) => void;
    onExternalNavigate: (url: string) => void;
  }
): boolean {
  const type = getCitationType(parsed);

  if (type === 'internal') {
    callbacks.onInternalNavigate(parsed);
    return true;
  }

  const url = getExternalUrl(parsed);
  if (url) {
    callbacks.onExternalNavigate(url);
    return false;
  }

  // Fallback: try internal navigation
  callbacks.onInternalNavigate(parsed);
  return true;
}
