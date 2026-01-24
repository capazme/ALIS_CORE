import type { SearchParams } from '../types';

/**
 * Parses a Normattiva URL and extracts search parameters
 *
 * Normattiva URL formats:
 * - https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:legge:1990-08-07;241~art1
 * - https://www.normattiva.it/atto/caricaDettaglioAtto?atto.dataPubblicazioneGazzetta=1990-08-07&atto.codiceRedazionale=090G0294
 * - https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:codice.civile:1942-03-16;262~art2043
 *
 * URN format: urn:nir:{authority}:{tipo_atto}:{data};{numero}~art{articolo}
 */

// Map URN act types to our internal format
const URN_ACT_TYPE_MAP: Record<string, string> = {
  'legge': 'legge',
  'decreto.legge': 'decreto legge',
  'decreto.legislativo': 'decreto legislativo',
  'decreto.del.presidente.della.repubblica': 'd.p.r.',
  'regio.decreto': 'regio decreto',
  'codice.civile': 'codice civile',
  'codice.penale': 'codice penale',
  'codice.di.procedura.civile': 'codice di procedura civile',
  'codice.di.procedura.penale': 'codice di procedura penale',
  'costituzione': 'costituzione',
};

/**
 * Mapping URN patterns to readable code names.
 * Based on NORMATTIVA_URN_CODICI from backend map.py
 *
 * Key: "tipo_atto:data;numero" pattern from URN
 * Value: Human-readable code name
 */
const URN_TO_CODICE_MAP: Record<string, string> = {
  // Codici fondamentali
  'regio.decreto:1942-03-16;262': 'Codice Civile',
  'regio.decreto:1942-03-16;262:1': 'Preleggi',
  'regio.decreto:1942-03-16;262:2': 'Codice Civile',
  'regio.decreto:1930-10-19;1398': 'Codice Penale',
  'regio.decreto:1930-10-19;1398:1': 'Codice Penale',
  'regio.decreto:1940-10-28;1443': 'Codice di Procedura Civile',
  'regio.decreto:1940-10-28;1443:1': 'Codice di Procedura Civile',
  'decreto.del.presidente.della.repubblica:1988-09-22;447': 'Codice di Procedura Penale',
  'regio.decreto:1942-03-30;318:1': 'Disp. Att. C.C.',
  'regio.decreto:1941-08-25;1368:1': 'Disp. Att. C.P.C.',
  'regio.decreto:1942-03-30;327:1': 'Codice della Navigazione',

  // Codici settoriali
  'decreto.legislativo:1992-04-30;285': 'Codice della Strada',
  'decreto.legislativo:1992-12-31;546': 'Codice del Processo Tributario',
  'decreto.legislativo:2003-06-30;196': 'Codice Privacy',
  'decreto.legislativo:2003-08-01;259': 'Codice Comunicazioni Elettroniche',
  'decreto.legislativo:2004-01-22;42': 'Codice Beni Culturali',
  'decreto.legislativo:2005-02-10;30': 'Codice Proprietà Industriale',
  'decreto.legislativo:2005-03-07;82': 'CAD',
  'decreto.legislativo:2005-07-18;171': 'Codice Nautica da Diporto',
  'decreto.legislativo:2005-09-06;206': 'Codice del Consumo',
  'decreto.legislativo:2005-09-07;209': 'Codice Assicurazioni',
  'decreto.legislativo:2006-04-03;152': 'Codice Ambiente',
  'decreto.legislativo:2006-04-11;198': 'Codice Pari Opportunità',
  'decreto.legislativo:2010-03-15;66': 'Codice Ordinamento Militare',
  'decreto.legislativo:2010-07-02;104': 'Codice Processo Amministrativo',
  'decreto.legislativo:2010-07-02;104:2': 'Codice Processo Amministrativo',
  'decreto.legislativo:2011-05-23;79': 'Codice del Turismo',
  'decreto.legislativo:2011-09-06;159': 'Codice Antimafia',
  'decreto.legislativo:2016-08-26;174:1': 'Codice Giustizia Contabile',
  'decreto.legislativo:2017-07-03;117': 'Codice Terzo Settore',
  'decreto.legislativo:2018-01-02;1': 'Codice Protezione Civile',
  'decreto.legislativo:2019-01-12;14': 'Codice Crisi d\'Impresa',
  'decreto.legislativo:2023-03-31;36': 'Codice Contratti Pubblici',
};

/**
 * Try to match URN to a known codice name
 */
function matchUrnToCodice(urnPart: string): string | null {
  // Direct match
  if (URN_TO_CODICE_MAP[urnPart]) {
    return URN_TO_CODICE_MAP[urnPart];
  }

  // Try without allegato suffix (e.g., ":2" at the end)
  const withoutAllegato = urnPart.replace(/:\d+$/, '');
  if (URN_TO_CODICE_MAP[withoutAllegato]) {
    return URN_TO_CODICE_MAP[withoutAllegato];
  }

  // Try matching by date;number pattern
  for (const [pattern, name] of Object.entries(URN_TO_CODICE_MAP)) {
    if (urnPart.includes(pattern.split(':')[1])) { // Match by date;number
      return name;
    }
  }

  return null;
}

interface ParseResult {
  success: boolean;
  params?: Partial<SearchParams>;
  error?: string;
}

/**
 * Parse a Normattiva URN string
 * Example: urn:nir:stato:legge:1990-08-07;241~art1
 */
function parseURN(urn: string): ParseResult {
  try {
    // Remove "urn:nir:" prefix
    const withoutPrefix = urn.replace(/^urn:nir:/i, '');

    // Split by : to get parts
    // Format: {authority}:{tipo_atto}:{data};{numero}~art{articolo}
    const parts = withoutPrefix.split(':');

    if (parts.length < 3) {
      return { success: false, error: 'URN non valido: formato non riconosciuto' };
    }

    // Authority is first part (e.g., "stato")
    // Act type is second part (e.g., "legge" or "codice.civile")
    const actTypePart = parts[1];

    // Last part contains date, number, and article
    // Format: {data};{numero}~art{articolo} or {data};{numero}~art{articolo}-com{comma}
    const lastPart = parts.slice(2).join(':'); // Rejoin in case there were extra colons

    // Extract date (before semicolon or tilde)
    let date = '';
    let actNumber = '';
    let article = '';

    // Try to match different patterns
    const fullMatch = lastPart.match(/^(\d{4}-\d{2}-\d{2});?(\d+)?(?:~art(\d+[a-z]?))?/i);

    if (fullMatch) {
      date = fullMatch[1] || '';
      actNumber = fullMatch[2] || '';
      article = fullMatch[3] || '1';
    } else {
      // Try simpler pattern for codici
      const simpleMatch = lastPart.match(/^(\d{4}-\d{2}-\d{2});?(\d+)?/);
      if (simpleMatch) {
        date = simpleMatch[1] || '';
        actNumber = simpleMatch[2] || '';
      }

      // Extract article from ~art{number}
      const articleMatch = lastPart.match(/~art(\d+[a-z]?)/i);
      if (articleMatch) {
        article = articleMatch[1];
      }
    }

    // Map act type
    const actType = URN_ACT_TYPE_MAP[actTypePart.toLowerCase()] || actTypePart.replace(/\./g, ' ');

    return {
      success: true,
      params: {
        act_type: actType,
        date: date,
        act_number: actNumber,
        article: article || '1',
        version: 'vigente',
        show_brocardi_info: true
      }
    };
  } catch (e) {
    return { success: false, error: 'Errore nel parsing dell\'URN' };
  }
}

/**
 * Parse a full Normattiva URL
 */
export function parseNormattivaUrl(url: string): ParseResult {
  try {
    // Clean the URL
    const cleanUrl = url.trim();

    // Check if it's a Normattiva URL
    if (!cleanUrl.includes('normattiva.it')) {
      return { success: false, error: 'URL non riconosciuto come Normattiva' };
    }

    // Try to extract URN from the URL
    // Pattern 1: uri-res/N2Ls?{urn}
    const urnMatch = cleanUrl.match(/uri-res\/N2Ls\?(.+?)(?:&|$)/i);
    if (urnMatch) {
      const urn = decodeURIComponent(urnMatch[1]);
      return parseURN(urn);
    }

    // Pattern 2: Check for urn parameter
    const urlObj = new URL(cleanUrl);
    const urnParam = urlObj.searchParams.get('urn');
    if (urnParam) {
      return parseURN(urnParam);
    }

    // Pattern 3: Try to find URN anywhere in the URL
    const urnInUrl = cleanUrl.match(/urn:nir:[^\s&]+/i);
    if (urnInUrl) {
      return parseURN(urnInUrl[0]);
    }

    // Pattern 4: caricaDettaglioAtto format
    // https://www.normattiva.it/atto/caricaDettaglioAtto?atto.dataPubblicazioneGazzetta=1990-08-07&...
    if (cleanUrl.includes('caricaDettaglioAtto')) {
      const dateMatch = cleanUrl.match(/dataPubblicazioneGazzetta=(\d{4}-\d{2}-\d{2})/);
      // This format is harder to parse fully, return partial
      if (dateMatch) {
        return {
          success: true,
          params: {
            date: dateMatch[1],
            article: '1',
            version: 'vigente',
            show_brocardi_info: true
          }
        };
      }
    }

    return { success: false, error: 'Impossibile estrarre i parametri dall\'URL' };
  } catch (e) {
    return { success: false, error: 'URL non valido' };
  }
}

/**
 * Generate a label from search params
 */
export function generateLabelFromParams(params: Partial<SearchParams>): string {
  const parts: string[] = [];

  if (params.article) {
    parts.push(`Art. ${params.article}`);
  }

  if (params.act_type) {
    // Abbreviate common act types
    const abbrev: Record<string, string> = {
      // Fonti Primarie
      'costituzione': 'Cost.',
      'legge': 'L.',
      'decreto legge': 'D.L.',
      'decreto legislativo': 'D.Lgs.',
      'decreto del presidente della repubblica': 'D.P.R.',
      'regio decreto': 'R.D.',
      // Codici Fondamentali
      'codice civile': 'C.C.',
      'codice penale': 'C.P.',
      'codice di procedura civile': 'C.P.C.',
      'codice di procedura penale': 'C.P.P.',
      'preleggi': 'Prel.',
      "disposizioni per l'attuazione del codice civile e disposizioni transitorie": 'Disp. Att. C.C.',
      "disposizioni per l'attuazione del codice di procedura civile e disposizioni transitorie": 'Disp. Att. C.P.C.',
      // Codici Settoriali
      'codice della strada': 'C.d.S.',
      'codice della navigazione': 'Cod. Nav.',
      'codice del consumo': 'Cod. Cons.',
      'codice in materia di protezione dei dati personali': 'Cod. Privacy',
      'norme in materia ambientale': 'Cod. Amb.',
      'codice dei contratti pubblici': 'Cod. Appalti',
      'codice dei beni culturali e del paesaggio': 'Cod. Beni Cult.',
      'codice delle assicurazioni private': 'Cod. Ass.',
      'codice del processo tributario': 'C.P.Tr.',
      'codice del processo amministrativo': 'C.P.A.',
      "codice dell'amministrazione digitale": 'CAD',
      'codice della proprietà industriale': 'C.P.I.',
      'codice delle comunicazioni elettroniche': 'CCE',
      'codice delle pari opportunità': 'CPO',
      "codice dell'ordinamento militare": 'COM',
      'codice del turismo': 'Cod. Tur.',
      'codice antimafia': 'Cod. Antim.',
      'codice di giustizia contabile': 'CGC',
      'codice del terzo settore': 'CTS',
      'codice della protezione civile': 'Cod. Prot. Civ.',
      "codice della crisi d'impresa e dell'insolvenza": 'CCI',
      'codice della nautica da diporto': 'CND',
    };

    const actAbbrev = abbrev[params.act_type.toLowerCase()] || params.act_type;
    parts.push(actAbbrev);
  }

  if (params.act_number) {
    parts.push(`n. ${params.act_number}`);
  }

  if (params.date) {
    // Format date as year only if it's a full date
    const year = params.date.split('-')[0];
    if (year) {
      parts.push(`/${year}`);
    }
  }

  return parts.join(' ') || 'Norma senza titolo';
}

/**
 * Validate search params have minimum required fields
 */
export function validateSearchParams(params: Partial<SearchParams>): boolean {
  return !!(params.act_type && params.article);
}

/**
 * Convert a Normattiva URN or URL to human-readable format
 *
 * Examples:
 * - "urn:nir:stato:regio.decreto:1942-03-16;262:2~art1339" → "Art. 1339 Codice Civile"
 * - "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:legge:1990-08-07;241~art1" → "Art. 1 L. 241/1990"
 */
export function formatUrnToReadable(urnOrUrl: string): string {
  if (!urnOrUrl) return 'Riferimento non disponibile';

  try {
    // Extract URN from URL if needed
    let urn = urnOrUrl;
    if (urnOrUrl.includes('normattiva.it') || urnOrUrl.startsWith('http')) {
      const urnMatch = urnOrUrl.match(/urn:nir:[^\s&]+/i);
      if (urnMatch) {
        urn = decodeURIComponent(urnMatch[0]);
      }
    }

    // Extract article number from URN
    const articleMatch = urn.match(/~art(\d+[a-z]*(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)/i);
    const article = articleMatch ? articleMatch[1] : null;

    // Extract the act pattern (tipo.atto:data;numero) to check against codice map
    // URN format: urn:nir:stato:{tipo.atto}:{data};{numero}[:allegato]~art{articolo}
    const actPatternMatch = urn.match(/urn:nir:[^:]+:([^:]+):(\d{4}-\d{2}-\d{2});(\d+)(?::(\d+))?/i);

    if (actPatternMatch) {
      const [, actType, date, actNumber, allegato] = actPatternMatch;

      // Build the pattern to match against codice map
      let urnPattern = `${actType}:${date};${actNumber}`;
      if (allegato) {
        urnPattern += `:${allegato}`;
      }

      // Check if this matches a known codice
      const codiceName = matchUrnToCodice(urnPattern);

      if (codiceName) {
        // Known codice - use readable name
        return article ? `Art. ${article} ${codiceName}` : codiceName;
      }

      // Not a known codice - use standard format
      const actTypeReadable = URN_ACT_TYPE_MAP[actType.toLowerCase()] || actType.replace(/\./g, ' ');
      const year = date.split('-')[0];

      // Use abbreviations for common act types
      const abbrev: Record<string, string> = {
        'legge': 'L.',
        'decreto legge': 'D.L.',
        'decreto legislativo': 'D.Lgs.',
        'd.p.r.': 'D.P.R.',
        'regio decreto': 'R.D.',
      };

      const actAbbrev = abbrev[actTypeReadable.toLowerCase()] || actTypeReadable;

      if (article) {
        return `Art. ${article} ${actAbbrev} ${actNumber}/${year}`;
      }
      return `${actAbbrev} ${actNumber}/${year}`;
    }

    // Try standard parsing as fallback
    const parseResult = parseNormattivaUrl(urnOrUrl);
    if (parseResult.success && parseResult.params) {
      return generateLabelFromParams(parseResult.params);
    }

    // Final fallback: clean up the string
    return urn
      .replace(/^https?:\/\/www\.normattiva\.it\/uri-res\/N2Ls\?/, '')
      .replace(/^urn:nir:stato:/, '')
      .replace(/\./g, ' ')
      .replace(/~/g, ' art.')
      .replace(/;/g, ' n.')
      .replace(/:(\d)/g, ' $1');

  } catch {
    return urnOrUrl;
  }
}
