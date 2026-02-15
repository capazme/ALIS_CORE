/**
 * citationParser - Parse Italian legal citation strings into structured data.
 *
 * Handles patterns like:
 *   "art. 2043 c.c."
 *   "art. 12, comma 2, d.lgs. 196/2003"
 *   "artt. 1 e 2 Cost."
 */

export interface ParsedCitation {
  articleNumber: string | null;
  comma: string | null;
  actType: string | null;
  actNumber: string | null;
  actYear: string | null;
  actLabel: string | null;
  raw: string;
}

const ARTICLE_REGEX = /\bart(?:t|icol[oi])?\.\s*(\d+(?:\s*[-/]\s*\w+)?)/i;
const COMMA_REGEX = /\bcomma\s+(\d+)/i;
const ACT_REGEX = /\b(d\.?\s*lgs|d\.?\s*l|d\.?\s*p\.?\s*r|l|r\.?\s*d|c\.?\s*c|c\.?\s*p|c\.?\s*p\.?\s*c|cost|t\.?\s*u)\.?\s*(?:n\.\s*)?(\d+)?(?:\s*[/]\s*(\d{4}))?\b/i;

const ACT_LABELS: Record<string, string> = {
  'c.c': 'Codice Civile',
  'cc': 'Codice Civile',
  'c.p': 'Codice Penale',
  'cp': 'Codice Penale',
  'c.p.c': 'Codice di Procedura Civile',
  'cpc': 'Codice di Procedura Civile',
  'cost': 'Costituzione',
  'd.lgs': 'Decreto Legislativo',
  'dlgs': 'Decreto Legislativo',
  'd.l': 'Decreto Legge',
  'dl': 'Decreto Legge',
  'l': 'Legge',
  'r.d': 'Regio Decreto',
  'rd': 'Regio Decreto',
  'd.p.r': 'Decreto del Presidente della Repubblica',
  'dpr': 'Decreto del Presidente della Repubblica',
  't.u': 'Testo Unico',
  'tu': 'Testo Unico',
};

/**
 * Parse a legal citation string into its constituent parts.
 */
export function parseLegalCitation(query: string): ParsedCitation {
  const result: ParsedCitation = {
    articleNumber: null,
    comma: null,
    actType: null,
    actNumber: null,
    actYear: null,
    actLabel: null,
    raw: query,
  };

  const articleMatch = query.match(ARTICLE_REGEX);
  if (articleMatch) {
    result.articleNumber = articleMatch[1].trim();
  }

  const commaMatch = query.match(COMMA_REGEX);
  if (commaMatch) {
    result.comma = commaMatch[1];
  }

  const actMatch = query.match(ACT_REGEX);
  if (actMatch) {
    const rawType = actMatch[1].replace(/\s+/g, '').replace(/\./g, '').toLowerCase();
    result.actType = actMatch[1];
    result.actNumber = actMatch[2] || null;
    result.actYear = actMatch[3] || null;
    result.actLabel = ACT_LABELS[rawType] || null;
  }

  return result;
}

/**
 * Check if a search query is specific enough to attempt resolution.
 */
export function isSearchReady(query: string): boolean {
  if (!query || query.trim().length < 3) return false;
  const parsed = parseLegalCitation(query);
  // At least an article + act type, or just an act type with a number
  return !!(parsed.articleNumber && parsed.actType) || !!(parsed.actType && parsed.actNumber);
}

/**
 * Format a parsed citation back into a human-readable string.
 */
export function formatParsedCitation(parsed: ParsedCitation): string {
  const parts: string[] = [];

  if (parsed.articleNumber) {
    parts.push(`Art. ${parsed.articleNumber}`);
  }

  if (parsed.comma) {
    parts.push(`comma ${parsed.comma}`);
  }

  if (parsed.actLabel) {
    parts.push(parsed.actLabel);
    if (parsed.actNumber) {
      parts.push(`n. ${parsed.actNumber}`);
    }
    if (parsed.actYear) {
      parts.push(`/${parsed.actYear}`);
    }
  } else if (parsed.actType) {
    parts.push(parsed.actType);
    if (parsed.actNumber) parts.push(parsed.actNumber);
    if (parsed.actYear) parts.push(`/${parsed.actYear}`);
  }

  return parts.join(' ') || parsed.raw;
}
