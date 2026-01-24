/**
 * Citation Matcher - Rileva tutte le citazioni normative nel testo.
 *
 * Supporta:
 * - Articoli semplici: "art. 5", "articolo 2043"
 * - Con tipo atto: "art. 2043 c.c.", "art. 575 c.p."
 * - Citazioni complete: "legge 241/1990 art. 3", "L. 241/90", "d.lgs. 50/2016"
 * - Articoli multipli: "artt. 1 e 2 c.c.", "artt. 1, 2, 3"
 * - Con comma/lettera: "art. 5, comma 1" (ignora comma, prende articolo)
 */

// Minimal interface for norma context (subset of NormaVisitata)
interface NormaContext {
  tipo_atto: string;
  numero_atto?: string;
  data?: string;
}

export interface CitationMatch {
  text: string;           // Testo originale matchato
  startIndex: number;     // Posizione nel testo
  endIndex: number;
  parsed: ParsedCitationData;
  cacheKey: string;       // Chiave per cache
}

export interface ParsedCitationData {
  act_type: string;
  act_number?: string;
  date?: string;
  article: string;
  confidence: number;
}

// Mappa abbreviazioni → tipo atto normalizzato
// Allineata con NORMATTIVA_SEARCH in visualex_api/tools/map.py
const ABBREVIATION_TO_ACT_TYPE: Record<string, string> = {
  // Leggi
  'l': 'legge',
  'l.': 'legge',
  'legge': 'legge',
  // Decreto legge
  'dl': 'decreto legge',
  'd.l.': 'decreto legge',
  'd.l': 'decreto legge',
  'decreto legge': 'decreto legge',
  // Decreto legislativo
  'dlgs': 'decreto legislativo',
  'd.lgs': 'decreto legislativo',
  'd.lgs.': 'decreto legislativo',
  'd. lgs.': 'decreto legislativo',
  'decreto legislativo': 'decreto legislativo',
  // DPR
  'dpr': 'decreto del presidente della repubblica',
  'd.p.r.': 'decreto del presidente della repubblica',
  'd.p.r': 'decreto del presidente della repubblica',
  // Regio decreto
  'rd': 'regio decreto',
  'r.d.': 'regio decreto',
  'r.d': 'regio decreto',
  'regio decreto': 'regio decreto',
  // EU
  'reg. ue': 'Regolamento UE',
  'regolamento ue': 'Regolamento UE',
  'dir. ue': 'Direttiva UE',
  'direttiva ue': 'Direttiva UE',
};

// Suffissi tipo atto dopo articolo (c.c., c.p., etc.)
// Allineata con NORMATTIVA_SEARCH in visualex_api/tools/map.py
const SUFFIX_TO_ACT_TYPE: Record<string, string> = {
  // Codici principali
  'c.c.': 'codice civile',
  'c.c': 'codice civile',
  'cc': 'codice civile',
  'cod. civ.': 'codice civile',
  'codice civile': 'codice civile',

  'c.p.': 'codice penale',
  'c.p': 'codice penale',
  'cp': 'codice penale',
  'cod. pen.': 'codice penale',
  'codice penale': 'codice penale',

  'c.p.c.': 'codice di procedura civile',
  'c.p.c': 'codice di procedura civile',
  'cpc': 'codice di procedura civile',
  'cod. proc. civ': 'codice di procedura civile',
  'codice di procedura civile': 'codice di procedura civile',

  'c.p.p.': 'codice di procedura penale',
  'c.p.p': 'codice di procedura penale',
  'cpp': 'codice di procedura penale',
  'cod. proc. pen.': 'codice di procedura penale',
  'codice di procedura penale': 'codice di procedura penale',

  // Costituzione e Preleggi
  'cost.': 'costituzione',
  'cost': 'costituzione',
  'costituzione': 'costituzione',
  'prel.': 'preleggi',
  'preleggi': 'preleggi',
  'disp. prel.': 'preleggi',

  // Altri codici (abbreviazioni da NORMATTIVA_SEARCH)
  'cds': 'codice della strada',
  'c.d.s.': 'codice della strada',
  'cod. strada': 'codice della strada',
  'codice della strada': 'codice della strada',

  'cn': 'codice della navigazione',
  'c.n.': 'codice della navigazione',
  'cod. nav.': 'codice della navigazione',
  'codice della navigazione': 'codice della navigazione',

  'cad': "codice dell'amministrazione digitale",
  'cod. amm. dig.': "codice dell'amministrazione digitale",
  "codice dell'amministrazione digitale": "codice dell'amministrazione digitale",

  'cam': 'codice antimafia',
  'cod. antimafia': 'codice antimafia',
  'codice antimafia': 'codice antimafia',

  'camb': 'norme in materia ambientale',
  'norme amb.': 'norme in materia ambientale',
  'norme in materia ambientale': 'norme in materia ambientale',

  'cap': 'codice delle assicurazioni private',
  'cod. ass. priv.': 'codice delle assicurazioni private',
  'codice delle assicurazioni private': 'codice delle assicurazioni private',

  'cbc': 'codice dei beni culturali e del paesaggio',
  'cod. beni cult.': 'codice dei beni culturali e del paesaggio',
  'codice dei beni culturali e del paesaggio': 'codice dei beni culturali e del paesaggio',

  'cce': 'codice delle comunicazioni elettroniche',
  'cod. com. elet.': 'codice delle comunicazioni elettroniche',
  'codice delle comunicazioni elettroniche': 'codice delle comunicazioni elettroniche',

  'cci': "codice della crisi d'impresa e dell'insolvenza",
  'cod. crisi imp.': "codice della crisi d'impresa e dell'insolvenza",
  "codice della crisi d'impresa e dell'insolvenza": "codice della crisi d'impresa e dell'insolvenza",

  'ccp': 'codice dei contratti pubblici',
  'c.c.p': 'codice dei contratti pubblici',
  'cod. contr. pubb.': 'codice dei contratti pubblici',
  'codice dei contratti pubblici': 'codice dei contratti pubblici',

  'cdc': 'codice del consumo',
  'cod. consumo': 'codice del consumo',
  'codice del consumo': 'codice del consumo',

  'cdpc': 'codice della protezione civile',
  'cod. prot. civ.': 'codice della protezione civile',
  'codice della protezione civile': 'codice della protezione civile',

  'cgco': 'codice di giustizia contabile',
  'cod. giust. cont.': 'codice di giustizia contabile',
  'codice di giustizia contabile': 'codice di giustizia contabile',

  'cnd': 'codice della nautica da diporto',
  'cod. naut. diport.': 'codice della nautica da diporto',
  'codice della nautica da diporto': 'codice della nautica da diporto',

  'com': "codice dell'ordinamento militare",
  'cod. ord. mil.': "codice dell'ordinamento militare",
  "codice dell'ordinamento militare": "codice dell'ordinamento militare",

  'cpa': 'codice del processo amministrativo',
  'cod. proc. amm.': 'codice del processo amministrativo',
  'codice del processo amministrativo': 'codice del processo amministrativo',

  'cpd': 'codice in materia di protezione dei dati personali',
  'cod. prot. dati': 'codice in materia di protezione dei dati personali',
  'codice in materia di protezione dei dati personali': 'codice in materia di protezione dei dati personali',

  'cpet': 'codice postale e delle telecomunicazioni',
  'cod. post. telecom.': 'codice postale e delle telecomunicazioni',
  'codice postale e delle telecomunicazioni': 'codice postale e delle telecomunicazioni',

  'cpi': 'codice della proprietà industriale',
  'cod. prop. ind.': 'codice della proprietà industriale',
  'codice della proprietà industriale': 'codice della proprietà industriale',

  'cpo': 'codice delle pari opportunità',
  'cod. pari opp.': 'codice delle pari opportunità',
  'codice delle pari opportunità': 'codice delle pari opportunità',

  'cpt': 'codice del processo tributario',
  'cod. proc. trib.': 'codice del processo tributario',
  'codice del processo tributario': 'codice del processo tributario',

  'cts': 'codice del Terzo settore',
  'cod. ter. sett.': 'codice del Terzo settore',
  'codice del Terzo settore': 'codice del Terzo settore',

  'ctu': 'codice del turismo',
  'cod. turismo': 'codice del turismo',
  'codice del turismo': 'codice del turismo',

  // Disposizioni attuative - ORDINE: più specifici prima!
  'disp. att. c.p.c.': 'disposizioni per l\'attuazione del Codice di procedura civile e disposizioni transitorie',
  'disp att cpc': 'disposizioni per l\'attuazione del Codice di procedura civile e disposizioni transitorie',
  'disp. att. c.c.': 'disposizioni per l\'attuazione del Codice civile e disposizioni transitorie',
  'disp att cc': 'disposizioni per l\'attuazione del Codice civile e disposizioni transitorie',
  'disp. att. c.p.p.': 'disposizioni per l\'attuazione del Codice di procedura penale',
  'disp att cpp': 'disposizioni per l\'attuazione del Codice di procedura penale',
  'disp. att.': 'disposizioni attuative',  // Generic fallback
  'disp att': 'disposizioni attuative',
  'disp.att.': 'disposizioni attuative',

  // Trattati EU
  'tue': 'TUE',
  'tfue': 'TFUE',
  'cdfue': 'CDFUE',
};

// Suffissi articolo (bis, ter, etc.)
const ARTICLE_SUFFIX_PATTERN = '(?:-?\\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?';

// Preposizioni articolate che precedono "articolo" (dell'articolo, dall'articolo, etc.)
const PREPOSITION_PATTERN = "(?:dell?'|dall?'|all?'|nell?'|sull?')?";

// Pattern base per "articolo" con tutte le varianti
const ARTICLE_WORD_PATTERN = `${PREPOSITION_PATTERN}art(?:icol[oi])?t?\\.?`;

// Pattern regex per suffissi codice (abbreviazioni brevi usate dopo articolo)
// Include: c.c., c.p., c.p.c., c.p.p., cost., prel., cds, cad, cam, ccp, cdc, cpa, etc.
// IMPORTANTE: I pattern più lunghi devono venire PRIMA di quelli più corti!
// Es: c.p.c. prima di c.p., altrimenti "c.p.c." matcha solo "c.p."
const CODE_SUFFIX_REGEX_PATTERN =
  // Disposizioni attuative - PIÙ LUNGHI, vanno PRIMA di tutto!
  'disp\\.?\\s*att\\.?\\s*c\\.?\\s*p\\.?\\s*c\\.?|' +  // disp. att. c.p.c.
  'disp\\.?\\s*att\\.?\\s*c\\.?\\s*p\\.?\\s*p\\.?|' +  // disp. att. c.p.p.
  'disp\\.?\\s*att\\.?\\s*c\\.?\\s*c\\.?|' +           // disp. att. c.c.
  'disp\\.?\\s*att\\.?|' +                             // disp. att. (generic)
  // Codici principali con punti - ORDINE: più lunghi prima!
  'c\\.?\\s*p\\.?\\s*c\\.?|c\\.?\\s*p\\.?\\s*p\\.?|' +  // c.p.c., c.p.p. PRIMA
  'c\\.?\\s*c\\.?\\s*p|' +                              // c.c.p (contratti pubblici)
  'c\\.?\\s*d\\.?\\s*s\\.?|c\\.?\\s*n\\.?|' +          // c.d.s., c.n.
  'c\\.?\\s*c\\.?|c\\.?\\s*p\\.?|' +                    // c.c., c.p. DOPO (più corti)
  // Costituzione e preleggi (con lookahead per evitare "costo", "preliminare", etc.)
  '(?:cost|prel)\\.?(?![a-z])|' +
  // Abbreviazioni 2-4 lettere (ordine: più lunghe prima)
  // IMPORTANTE: (?![a-z]) evita match parziali (es: "com" non matcha "comma")
  '(?:camb|cdpc|cgco|cpet)(?![a-z])|' +                // 4 lettere
  '(?:cds|cad|cam|cap|cbc|cce|cci|ccp|cdc|cnd|com|cpa|cpd|cpi|cpo|cpt|cts|ctu)(?![a-z])|' + // 3 lettere
  // Forme con "cod." - più specifiche prima
  'cod\\.?\\s*proc\\.?\\s*(?:civ|pen|amm|trib)\\.?|' +
  'cod\\.?\\s*(?:amm\\.?\\s*dig|crisi\\s*imp)\\.?|' +
  'cod\\.?\\s*(?:civ|pen|nav|strada|consumo|antimafia|turismo)\\.?|' +
  // Trattati EU
  '(?:tue|tfue|cdfue)(?![a-z])';

// Pattern per nomi estesi dei codici (usato con "del/della")
// IMPORTANTE: Pattern più lunghi/specifici PRIMA di quelli più corti
const CODE_FULL_NAME_PATTERN =
  // Disposizioni attuative - PIÙ LUNGHI, vanno PRIMA!
  'disposizioni\\s+(?:per\\s+l\'?)?attuazione\\s+del\\s+codice\\s+di\\s+procedura\\s+civile|' +
  'disposizioni\\s+(?:per\\s+l\'?)?attuazione\\s+del\\s+codice\\s+di\\s+procedura\\s+penale|' +
  'disposizioni\\s+(?:per\\s+l\'?)?attuazione\\s+del\\s+codice\\s+civile|' +
  'disposizioni\\s+attuative|disp\\.?\\s*att\\.?|' +
  // Codici di procedura PRIMA di codice civile/penale
  'codice\\s+di\\s+procedura\\s+civile|codice\\s+di\\s+procedura\\s+penale|' +
  // Poi codice civile/penale
  'codice\\s+civile|codice\\s+penale|' +
  // Altri codici
  'codice\\s+della\\s+strada|codice\\s+della\\s+navigazione|' +
  'codice\\s+del\\s+consumo|codice\\s+del\\s+turismo|' +
  'codice\\s+del\\s+processo\\s+amministrativo|codice\\s+del\\s+processo\\s+tributario|' +
  'codice\\s+dei\\s+contratti\\s+pubblici|codice\\s+dei\\s+beni\\s+culturali|' +
  'codice\\s+delle\\s+assicurazioni|codice\\s+delle\\s+comunicazioni|' +
  'codice\\s+della\\s+crisi|codice\\s+antimafia|' +
  "codice\\s+dell'amministrazione\\s+digitale|codice\\s+dell'ordinamento\\s+militare|" +
  'costituzione|preleggi|' +
  'norme\\s+in\\s+materia\\s+ambientale';

// =============================================================================
// CONTEXT-AWARE NORM DETECTION
// =============================================================================

interface NormMention {
  actType: string;
  actNumber?: string;
  date?: string;
  endIndex: number; // Posizione fine della menzione nel testo
}

/**
 * Cerca menzioni di norme nel testo (senza articolo).
 * Es: "D.Lgs. 31 ottobre 2024, n. 164", "legge 241/1990", "il codice penale"
 */
function findNormMentions(text: string): NormMention[] {
  const mentions: NormMention[] = [];

  // Pattern 1: "D.Lgs. 31 ottobre 2024, n. 164" - formato con data completa
  const dateFormatRegex = new RegExp(
    '(' +
      'legge|l\\.|' +
      'decreto\\s+legge|d\\.?\\s*l\\.?|dl|' +
      'decreto\\s+legislativo|d\\.?\\s*lgs\\.?|dlgs|' +
      'd\\.?\\s*p\\.?\\s*r\\.?|dpr|' +
      'regio\\s+decreto|r\\.?\\s*d\\.?|rd' +
    ')' +
    '\\s+\\d{1,2}\\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\\s+(\\d{4})\\s*,?\\s*n\\.?\\s*(\\d+)',
    'gi'
  );

  let match;
  while ((match = dateFormatRegex.exec(text)) !== null) {
    const actType = normalizeActType(match[1]);
    const year = match[2]; // Anno dalla data (es. "2024")
    const actNumber = match[3]; // Numero atto (es. "164")

    mentions.push({
      actType,
      actNumber,
      date: year,
      endIndex: match.index + match[0].length,
    });
  }

  // Pattern 2: "D.Lgs. n. 164/2024" o "legge 241/1990" - formato numero/anno
  const slashFormatRegex = new RegExp(
    '(' +
      'legge|l\\.|' +
      'decreto\\s+legge|d\\.?\\s*l\\.?|dl|' +
      'decreto\\s+legislativo|d\\.?\\s*lgs\\.?|dlgs|' +
      'd\\.?\\s*p\\.?\\s*r\\.?|dpr|' +
      'regio\\s+decreto|r\\.?\\s*d\\.?|rd' +
    ')' +
    '\\s+(?:n\\.?\\s*)?(\\d+)\\s*[/\\\\]\\s*(\\d{2,4})',
    'gi'
  );

  while ((match = slashFormatRegex.exec(text)) !== null) {
    const actType = normalizeActType(match[1]);
    const actNumber = match[2];
    const year = normalizeYear(match[3]);

    // Evita duplicati se già matchato dal pattern precedente
    const alreadyExists = mentions.some(m =>
      m.actType === actType && m.actNumber === actNumber && Math.abs(m.endIndex - (match!.index + match![0].length)) < 50
    );
    if (!alreadyExists) {
      mentions.push({
        actType,
        actNumber,
        date: year,
        endIndex: match.index + match[0].length,
      });
    }
  }

  // Pattern 3: Menzioni semplici di codici "il codice civile", "il codice penale", "la Costituzione"
  const codeRegex = /(?:il\s+|la\s+)?(codice\s+(?:civile|penale|di\s+procedura\s+civile|di\s+procedura\s+penale|della\s+strada|della\s+navigazione)|costituzione|preleggi)/gi;

  while ((match = codeRegex.exec(text)) !== null) {
    mentions.push({
      actType: normalizeActType(match[1]),
      endIndex: match.index + match[0].length,
    });
  }

  return mentions;
}

/**
 * Trova la norma più vicina che precede una posizione nel testo.
 * Cerca entro un limite di caratteri (default 300).
 */
function findPrecedingNorm(
  mentions: NormMention[],
  position: number,
  maxDistance: number = 300
): NormMention | null {
  let closest: NormMention | null = null;
  let closestDistance = Infinity;

  for (const mention of mentions) {
    // La menzione deve PRECEDERE la posizione
    if (mention.endIndex <= position) {
      const distance = position - mention.endIndex;
      if (distance <= maxDistance && distance < closestDistance) {
        closest = mention;
        closestDistance = distance;
      }
    }
  }

  return closest;
}

/**
 * Normalizza il tipo atto
 */
function normalizeActType(input: string): string {
  const normalized = input.toLowerCase().replace(/\s+/g, ' ').trim();

  // Prima cerca match esatto
  if (ABBREVIATION_TO_ACT_TYPE[normalized]) {
    return ABBREVIATION_TO_ACT_TYPE[normalized];
  }
  if (SUFFIX_TO_ACT_TYPE[normalized]) {
    return SUFFIX_TO_ACT_TYPE[normalized];
  }

  // Poi cerca match parziale per abbreviazioni con punti
  const withoutSpaces = normalized.replace(/\s/g, '');
  for (const [abbr, actType] of Object.entries(ABBREVIATION_TO_ACT_TYPE)) {
    if (abbr.replace(/\s/g, '') === withoutSpaces) {
      return actType;
    }
  }
  for (const [suffix, actType] of Object.entries(SUFFIX_TO_ACT_TYPE)) {
    if (suffix.replace(/\s/g, '') === withoutSpaces) {
      return actType;
    }
  }

  return normalized;
}

/**
 * Converte anno a 2 cifre in 4 cifre
 */
function normalizeYear(year: string): string {
  if (year.length === 2) {
    const num = parseInt(year);
    return num > 50 ? `19${year}` : `20${year}`;
  }
  return year;
}

/**
 * Genera una chiave di cache univoca per la citazione
 */
function generateCacheKey(parsed: ParsedCitationData): string {
  const parts = [
    parsed.act_type.toLowerCase().replace(/\s+/g, '-'),
    parsed.article,
  ];
  if (parsed.act_number) parts.push(parsed.act_number);
  if (parsed.date) parts.push(parsed.date);
  return parts.join('::');
}

/**
 * Opzioni per l'estrazione delle citazioni
 */
export interface ExtractCitationsOptions {
  /** Norma di default (usata se non si trova contesto) */
  defaultNorma?: NormaContext;
  /** Se true, estrae anche menzioni di norme senza articolo (mostra art. 1) */
  includeNormsWithoutArticle?: boolean;
  /** Distanza massima per il context-aware detection (default: 300 chars) */
  maxContextDistance?: number;
}

/**
 * Estrae tutte le citazioni normative dal testo.
 *
 * Usa context-aware detection per risolvere citazioni ambigue:
 * - "D.Lgs. 164/2024 ... art. 7" → art. 7 del D.Lgs. 164/2024 (non della norma corrente)
 *
 * @param text Testo da analizzare
 * @param defaultNormaOrOptions Norma di default o opzioni complete
 */
export function extractCitations(
  text: string,
  defaultNormaOrOptions?: NormaContext | ExtractCitationsOptions
): CitationMatch[] {
  if (!text || typeof text !== 'string') return [];

  // Normalizza opzioni
  const options: ExtractCitationsOptions = defaultNormaOrOptions && 'tipo_atto' in defaultNormaOrOptions
    ? { defaultNorma: defaultNormaOrOptions }
    : (defaultNormaOrOptions as ExtractCitationsOptions) || {};

  const { defaultNorma, includeNormsWithoutArticle = false, maxContextDistance = 300 } = options;

  // Rimuovi tag HTML per il matching (ma mantieni posizioni)
  const cleanText = text.replace(/<[^>]+>/g, (match) => ' '.repeat(match.length));

  const matches: CitationMatch[] = [];
  const usedRanges: Array<[number, number]> = [];

  // Pre-calcola le menzioni di norme per context-aware detection
  const normMentions = findNormMentions(cleanText);

  // Helper per verificare se una posizione è già usata
  const isOverlapping = (start: number, end: number): boolean => {
    return usedRanges.some(([s, e]) =>
      (start >= s && start < e) || (end > s && end <= e) || (start <= s && end >= e)
    );
  };

  // Helper per aggiungere un match
  const addMatch = (
    matchText: string,
    startIndex: number,
    endIndex: number,
    parsed: ParsedCitationData
  ) => {
    if (isOverlapping(startIndex, endIndex)) return;

    matches.push({
      text: matchText,
      startIndex,
      endIndex,
      parsed,
      cacheKey: generateCacheKey(parsed),
    });
    usedRanges.push([startIndex, endIndex]);
  };

  // ============================================
  // PATTERN 1: Citazioni complete con numero/anno
  // Es: "legge 241/1990", "L. 241/90", "d.lgs. 50/2016 art. 3"
  // ============================================
  const fullCitationRegex = new RegExp(
    // Tipo atto (legge, L., d.lgs., decreto legislativo, etc.)
    '(' +
      'legge|l\\.|' +
      'decreto\\s+legge|d\\.?\\s*l\\.?|dl|' +
      'decreto\\s+legislativo|d\\.?\\s*lgs\\.?|dlgs|' +
      'd\\.?\\s*p\\.?\\s*r\\.?|dpr|' +
      'regio\\s+decreto|r\\.?\\s*d\\.?|rd|' +
      'reg(?:olamento)?\\.?\\s+ue|' +
      'dir(?:ettiva)?\\.?\\s+ue' +
    ')' +
    // Spazio e numero/anno
    '\\s+(?:n\\.?\\s*)?(\\d+)\\s*[/\\\\]\\s*(\\d{2,4})' +
    // Articolo opzionale
    `(?:\\s*,?\\s*${ARTICLE_WORD_PATTERN}\\s*(\\d+${ARTICLE_SUFFIX_PATTERN}))?`,
    'gi'
  );

  let match;
  while ((match = fullCitationRegex.exec(cleanText)) !== null) {
    const actTypeMatch = match[1];
    const actNumber = match[2];
    const year = normalizeYear(match[3]);
    const article = match[4];

    // Se non c'è articolo, skip (non possiamo fare preview di tutta la legge)
    if (!article) continue;

    addMatch(match[0], match.index, match.index + match[0].length, {
      act_type: normalizeActType(actTypeMatch),
      act_number: actNumber,
      date: year,
      article: article.replace(/\s+/g, ''),
      confidence: 0.95,
    });
  }

  // ============================================
  // PATTERN 2: Articoli multipli con suffisso
  // Es: "artt. 1 e 2 c.c.", "artt. 1, 2, 3 c.p.", "artt. 179, 237, 288, 292 c.p.c."
  // Estrae TUTTI gli articoli dalla lista, non solo il primo
  // ============================================
  {
    const multiArticleRegex = new RegExp(
      `(${PREPOSITION_PATTERN}artt?\\.?\\s+)` +  // Gruppo 1: prefisso "artt. "
      // Cattura tutta la lista di numeri
      '(\\d+' + ARTICLE_SUFFIX_PATTERN + '(?:\\s*[,e]\\s*\\d+' + ARTICLE_SUFFIX_PATTERN + ')*)' +  // Gruppo 2: numeri
      '\\s+' +
      // Suffisso tipo atto (usa pattern completo)
      `(${CODE_SUFFIX_REGEX_PATTERN})`,  // Gruppo 3: tipo atto
      'gi'
    );

    while ((match = multiArticleRegex.exec(cleanText)) !== null) {
      const fullMatchStart = match.index;
      const fullMatchEnd = match.index + match[0].length;

      // Verifica che l'intero range non sia già usato
      if (isOverlapping(fullMatchStart, fullMatchEnd)) continue;

      const prefix = match[1];  // "artt. " o "degli artt. "
      const articlesGroup = match[2];  // "179, 237, 288, 292"
      const actTypeSuffix = match[3];  // "c.p.c."
      const actType = normalizeActType(actTypeSuffix);

      // Estrai TUTTI i numeri dalla lista
      const numberRegex = /(\d+(?:-?\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)/gi;
      let numMatch;

      // Posizione base dopo il prefisso
      const numbersStartIndex = fullMatchStart + prefix.length;

      while ((numMatch = numberRegex.exec(articlesGroup)) !== null) {
        const articleNum = numMatch[1];
        const numStartInGroup = numMatch.index;
        const numEndInGroup = numMatch.index + numMatch[0].length;

        // Calcola posizione assoluta nel testo
        const absStart = numbersStartIndex + numStartInGroup;
        const absEnd = numbersStartIndex + numEndInGroup;

        // Verifica che questo specifico numero non sia già coperto
        if (!isOverlapping(absStart, absEnd)) {
          addMatch(articleNum, absStart, absEnd, {
            act_type: actType,
            article: articleNum.replace(/\s+/g, ''),
            confidence: 0.85,
          });
        }
      }
    }
  }

  // ============================================
  // PATTERN 3: Articolo singolo con suffisso tipo atto
  // Es: "art. 2043 c.c.", "articolo 575 c.p.", "art. 5, comma 1, c.c."
  // ============================================
  const articleWithSuffixRegex = new RegExp(
    `${ARTICLE_WORD_PATTERN}\\s+` +
    '(\\d+' + ARTICLE_SUFFIX_PATTERN + ')' +
    // Ignora comma/lettera opzionali
    '(?:\\s*,?\\s*(?:comma|co\\.|lett\\.?)\\s*[\\d\\w]+)*' +
    // Separatore e suffisso tipo atto (usa pattern completo)
    '\\s*,?\\s*' +
    `(${CODE_SUFFIX_REGEX_PATTERN})`,
    'gi'
  );

  while ((match = articleWithSuffixRegex.exec(cleanText)) !== null) {
    if (isOverlapping(match.index, match.index + match[0].length)) continue;

    const article = match[1];
    const actTypeSuffix = match[2];

    addMatch(match[0], match.index, match.index + match[0].length, {
      act_type: normalizeActType(actTypeSuffix),
      article: article.replace(/\s+/g, ''),
      confidence: 0.9,
    });
  }

  // ============================================
  // PATTERN 3bis: Articolo con "del/della/delle" + codice
  // Es: "art. 702 ter del c.p.c.", "articolo 2043 del codice civile"
  //     "e art. 121 delle disp. att." (con congiunzione)
  // PRIORITÀ ALTA: Il riferimento POST-articolo sovrascrive quelli precedenti
  // ============================================
  const articleWithDelCodeRegex = new RegExp(
    // Congiunzione opzionale (e/ed) prima dell'articolo
    '(?:e(?:d)?\\s+)?' +
    `${ARTICLE_WORD_PATTERN}\\s+` +
    '(\\d+' + ARTICLE_SUFFIX_PATTERN + ')' +
    // Ignora comma/lettera opzionali
    '(?:\\s*,?\\s*(?:comma|co\\.|lett\\.?)\\s*[\\d\\w]+)*' +
    // "del/della/dello/delle" + codice (abbreviato o esteso)
    '\\s+del(?:la|lo|le)?\\s+' +
    // Usa entrambi i pattern: abbreviazioni E nomi estesi
    `(${CODE_SUFFIX_REGEX_PATTERN}|${CODE_FULL_NAME_PATTERN})`,
    'gi'
  );

  while ((match = articleWithDelCodeRegex.exec(cleanText)) !== null) {
    if (isOverlapping(match.index, match.index + match[0].length)) continue;

    const article = match[1];
    const actTypeSuffix = match[2];

    addMatch(match[0], match.index, match.index + match[0].length, {
      act_type: normalizeActType(actTypeSuffix),
      article: article.replace(/\s+/g, ''),
      confidence: 0.95, // Alta confidence: riferimento esplicito post-articolo
    });
  }

  // ============================================
  // PATTERN 4: Articoli multipli senza suffisso
  // Es: "articoli 8 e 9", "artt. 1, 2 e 3", "degli articoli 8 e 9"
  // CONTEXT-AWARE: Cerca norma precedente nel testo, altrimenti usa defaultNorma
  // ============================================
  {
    const multiArticleNoSuffixRegex = new RegExp(
      `(${PREPOSITION_PATTERN}(?:artt?\\.?|articol[oi])\\s+)` +  // Gruppo 1: prefisso
      // Cattura tutto il gruppo di numeri: "8 e 9" o "1, 2 e 3"
      '(\\d+' + ARTICLE_SUFFIX_PATTERN + '(?:\\s*[,e]\\s*\\d+' + ARTICLE_SUFFIX_PATTERN + ')+)' +  // Gruppo 2: numeri
      // Negative lookahead per non matchare se c'è un suffisso tipo atto dopo
      // Include sia suffisso diretto (c.c.) che con "del" (del c.p.c.)
      '(?!\\s*,?\\s*(?:c\\.?\\s*c|c\\.?\\s*p|cost|prel|cod))' +
      '(?!\\s+del(?:la|lo)?\\s+(?:c\\.?\\s*[cpn]|cod|cost|prel|codice|costituzione))',
      'gi'
    );

    while ((match = multiArticleNoSuffixRegex.exec(cleanText)) !== null) {
      const fullMatchStart = match.index;
      const fullMatchEnd = match.index + match[0].length;

      // Verifica che l'intero range non sia già usato
      if (isOverlapping(fullMatchStart, fullMatchEnd)) continue;

      // CONTEXT-AWARE: Cerca norma precedente nel testo
      const precedingNorm = findPrecedingNorm(normMentions, fullMatchStart, maxContextDistance);

      // Determina la norma da usare: precedente nel contesto > default
      const targetNorma = precedingNorm
        ? { tipo_atto: precedingNorm.actType, numero_atto: precedingNorm.actNumber, data: precedingNorm.date }
        : defaultNorma;

      // Skip se non abbiamo una norma da associare
      if (!targetNorma?.tipo_atto) continue;

      // Confidence più alta se usiamo contesto rilevato
      const baseConfidence = precedingNorm ? 0.85 : 0.75;

      const prefix = match[1];  // "articoli " o "degli articoli "
      const articlesGroup = match[2];  // "8 e 9"

      // Trova la posizione di ogni numero all'interno del testo originale
      const numberRegex = /(\d+(?:-?\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)/gi;
      let numMatch;

      // La posizione base è dopo il prefisso
      const numbersStartIndex = fullMatchStart + prefix.length;

      while ((numMatch = numberRegex.exec(articlesGroup)) !== null) {
        const articleNum = numMatch[1];
        const numStartInGroup = numMatch.index;
        const numEndInGroup = numMatch.index + numMatch[0].length;

        // Calcola posizione assoluta nel testo
        const absStart = numbersStartIndex + numStartInGroup;
        const absEnd = numbersStartIndex + numEndInGroup;

        // Verifica che questo specifico numero non sia già coperto
        if (!isOverlapping(absStart, absEnd)) {
          addMatch(articleNum, absStart, absEnd, {
            act_type: targetNorma.tipo_atto,
            act_number: targetNorma.numero_atto,
            date: targetNorma.data,
            article: articleNum.replace(/\s+/g, ''),
            confidence: baseConfidence,
          });
        }
      }
    }
  }

  // ============================================
  // PATTERN 5: Articolo semplice
  // Es: "art. 5", "articolo 123", "art. 5, comma 1"
  // CONTEXT-AWARE: Cerca norma precedente nel testo, altrimenti usa defaultNorma
  // ============================================
  {
    const simpleArticleRegex = new RegExp(
      `${ARTICLE_WORD_PATTERN}\\s+` +
      '(\\d+' + ARTICLE_SUFFIX_PATTERN + ')' +
      // Ignora comma/lettera opzionali (ma non catturare suffisso tipo atto)
      '(?:\\s*,?\\s*(?:comma|co\\.|lett\\.?)\\s*[\\d\\w]+)*' +
      // Negative lookahead per non matchare se c'è un suffisso tipo atto dopo
      // Include sia suffisso diretto (c.c.) che con "del" (del c.p.c.)
      '(?!\\s*,?\\s*(?:c\\.?\\s*c|c\\.?\\s*p|cost|prel|cod))' +
      '(?!\\s+del(?:la|lo)?\\s+(?:c\\.?\\s*[cpn]|cod|cost|prel|codice|costituzione))',
      'gi'
    );

    while ((match = simpleArticleRegex.exec(cleanText)) !== null) {
      if (isOverlapping(match.index, match.index + match[0].length)) continue;

      // CONTEXT-AWARE: Cerca norma precedente nel testo
      const precedingNorm = findPrecedingNorm(normMentions, match.index, maxContextDistance);

      // Determina la norma da usare: precedente nel contesto > default
      const targetNorma = precedingNorm
        ? { tipo_atto: precedingNorm.actType, numero_atto: precedingNorm.actNumber, data: precedingNorm.date }
        : defaultNorma;

      // Skip se non abbiamo una norma da associare
      if (!targetNorma?.tipo_atto) continue;

      // Confidence più alta se usiamo contesto rilevato
      const confidence = precedingNorm ? 0.8 : 0.7;

      const article = match[1];

      addMatch(match[0], match.index, match.index + match[0].length, {
        act_type: targetNorma.tipo_atto,
        act_number: targetNorma.numero_atto,
        date: targetNorma.data,
        article: article.replace(/\s+/g, ''),
        confidence,
      });
    }
  }

  // ============================================
  // PATTERN 6: Menzioni di norme senza articolo (opzionale)
  // Es: "legge 241/1990", "D.Lgs. 50/2016", "il codice civile"
  // Se abilitato, crea citazione all'art. 1
  // ============================================
  if (includeNormsWithoutArticle) {
    for (const mention of normMentions) {
      // Trova la posizione di inizio della menzione (endIndex - lunghezza stimata)
      // Approssimazione: cerca all'indietro
      const searchStart = Math.max(0, mention.endIndex - 100);
      const searchArea = cleanText.substring(searchStart, mention.endIndex);

      // Regex per trovare l'inizio della menzione
      const normStartRegex = new RegExp(
        '(?:' +
          'legge|l\\.|' +
          'decreto\\s+legge|d\\.?\\s*l\\.?|dl|' +
          'decreto\\s+legislativo|d\\.?\\s*lgs\\.?|dlgs|' +
          'd\\.?\\s*p\\.?\\s*r\\.?|dpr|' +
          'regio\\s+decreto|r\\.?\\s*d\\.?|rd|' +
          'il\\s+|la\\s+' +
        ')' +
        '.*$',
        'i'
      );

      const startMatch = normStartRegex.exec(searchArea);
      if (!startMatch) continue;

      const startIndex = searchStart + startMatch.index;
      const endIndex = mention.endIndex;

      // Skip se già coperto da altri pattern
      if (isOverlapping(startIndex, endIndex)) continue;

      const matchedText = cleanText.substring(startIndex, endIndex);

      addMatch(matchedText, startIndex, endIndex, {
        act_type: mention.actType,
        act_number: mention.actNumber,
        date: mention.date,
        article: '1', // Default all'art. 1
        confidence: 0.6, // Confidence bassa perché non c'è articolo esplicito
      });
    }
  }

  // Ordina per posizione
  return matches.sort((a, b) => a.startIndex - b.startIndex);
}

/**
 * Serializza i dati della citazione per l'attributo data-citation
 */
export function serializeCitation(parsed: ParsedCitationData): string {
  return JSON.stringify(parsed);
}

/**
 * Deserializza i dati della citazione dall'attributo data-citation
 */
export function deserializeCitation(data: string): ParsedCitationData | null {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

/**
 * Formatta una citazione per la visualizzazione
 */
export function formatCitationLabel(parsed: ParsedCitationData): string {
  // Mappa tipo atto → abbreviazione per label
  // Allineata con BROCARDI_SEARCH in visualex_api/tools/map.py
  const shortNames: Record<string, string> = {
    // Leggi e decreti
    'legge': 'L.',
    'decreto legge': 'D.L.',
    'decreto legislativo': 'D.Lgs.',
    'decreto del presidente della repubblica': 'D.P.R.',
    'regio decreto': 'R.D.',
    // Codici principali
    'codice civile': 'C.C.',
    'codice penale': 'C.P.',
    'codice di procedura civile': 'C.P.C.',
    'codice di procedura penale': 'C.P.P.',
    'costituzione': 'Cost.',
    'preleggi': 'Prel.',
    // Altri codici
    'codice della strada': 'C.d.S.',
    'codice della navigazione': 'C.N.',
    "codice dell'amministrazione digitale": 'CAD',
    'codice antimafia': 'C.A.M.',
    'norme in materia ambientale': 'C.Amb.',
    'codice delle assicurazioni private': 'C.Ass.',
    'codice dei beni culturali e del paesaggio': 'C.B.C.',
    'codice delle comunicazioni elettroniche': 'C.C.E.',
    "codice della crisi d'impresa e dell'insolvenza": 'C.C.I.',
    'codice dei contratti pubblici': 'C.C.P.',
    'codice del consumo': 'C.d.C.',
    'codice della protezione civile': 'C.P.Civ.',
    'codice di giustizia contabile': 'C.G.C.',
    'codice della nautica da diporto': 'C.N.D.',
    "codice dell'ordinamento militare": 'C.O.M.',
    'codice del processo amministrativo': 'C.P.A.',
    'codice in materia di protezione dei dati personali': 'C.P.D.',
    'codice postale e delle telecomunicazioni': 'C.P.T.',
    'codice della proprietà industriale': 'C.P.I.',
    'codice delle pari opportunità': 'C.P.O.',
    'codice del processo tributario': 'C.Proc.Trib.',
    'codice del Terzo settore': 'C.T.S.',
    'codice del turismo': 'C.Tur.',
    // Disposizioni attuative
    "disposizioni per l'attuazione del Codice civile e disposizioni transitorie": 'Disp. Att. C.C.',
    "disposizioni per l'attuazione del Codice di procedura civile e disposizioni transitorie": 'Disp. Att. C.P.C.',
    "disposizioni per l'attuazione del Codice di procedura penale": 'Disp. Att. C.P.P.',
    'disposizioni attuative': 'Disp. Att.',
    // Trattati EU
    'TUE': 'TUE',
    'TFUE': 'TFUE',
    'CDFUE': 'CDFUE',
    'Regolamento UE': 'Reg. UE',
    'Direttiva UE': 'Dir. UE',
  };

  // Atti "unici" che non necessitano di numero/anno nel label (codici, costituzione, etc.)
  const uniqueActs = new Set([
    'codice civile',
    'codice penale',
    'codice di procedura civile',
    'codice di procedura penale',
    'costituzione',
    'codice della strada',
    'codice della navigazione',
    "codice dell'amministrazione digitale",
    'codice antimafia',
    'norme in materia ambientale',
    'codice delle assicurazioni private',
    'codice dei beni culturali e del paesaggio',
    'codice delle comunicazioni elettroniche',
    "codice della crisi d'impresa e dell'insolvenza",
    'codice dei contratti pubblici',
    'codice del consumo',
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
    'codice del Terzo settore',
    'codice del turismo',
    'preleggi',
    "disposizioni per l'attuazione del Codice civile e disposizioni transitorie",
    "disposizioni per l'attuazione del Codice di procedura civile e disposizioni transitorie",
    "disposizioni per l'attuazione del Codice di procedura penale",
    'disposizioni attuative',
    'TUE',
    'TFUE',
    'CDFUE',
  ]);

  const parts: string[] = [`Art. ${parsed.article}`];
  parts.push(shortNames[parsed.act_type] || parsed.act_type);

  // Mostra numero/anno solo per atti non unici (leggi, decreti, etc.)
  if (parsed.act_number && parsed.date && !uniqueActs.has(parsed.act_type)) {
    parts.push(`${parsed.act_number}/${parsed.date}`);
  }

  return parts.join(' ');
}

/**
 * Applica il wrapping delle citazioni al testo HTML.
 * Restituisce il testo con le citazioni wrappate in span.
 */
export function wrapCitationsInHtml(
  html: string,
  defaultNormaOrOptions?: NormaContext | ExtractCitationsOptions
): string {
  const citations = extractCitations(html, defaultNormaOrOptions);

  if (citations.length === 0) return html;

  // Processa dal fondo per non invalidare gli indici
  let result = html;
  for (let i = citations.length - 1; i >= 0; i--) {
    const citation = citations[i];
    const serialized = serializeCitation(citation.parsed);
    const escaped = serialized.replace(/"/g, '&quot;');

    const before = result.substring(0, citation.startIndex);
    const after = result.substring(citation.endIndex);
    const matchText = result.substring(citation.startIndex, citation.endIndex);

    result = before +
      `<span class="citation-hover" data-citation="${escaped}" data-cache-key="${citation.cacheKey}">${matchText}</span>` +
      after;
  }

  return result;
}
