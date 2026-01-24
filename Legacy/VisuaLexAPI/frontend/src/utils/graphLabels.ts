/**
 * Graph Label Utilities
 * =====================
 *
 * Funzioni per creare label intelligenti e abbreviate per i nodi del Knowledge Graph.
 * Migliora la leggibilità riducendo il clutter visivo.
 */

import type { SubgraphNode } from '../types/merlt';

/**
 * Estrae una label compatta per un articolo dal suo URN o label.
 *
 * @example
 * "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:codice.civile:1942~art1218"
 * → "Art. 1218 c.c."
 *
 * "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:codice.penale:1930~art52"
 * → "Art. 52 c.p."
 */
export function extractArticleLabel(urnOrLabel: string): string {
  // Pattern per articolo nel URN
  const artMatch = urnOrLabel.match(/~art(\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)/i);

  if (artMatch) {
    const articleNum = artMatch[1];

    // Determina il codice dalla parte dell'URN
    if (urnOrLabel.includes('codice.civile') || urnOrLabel.includes('regio.decreto:1942-03-16;262')) {
      return `Art. ${articleNum} c.c.`;
    }
    if (urnOrLabel.includes('codice.penale') || urnOrLabel.includes('regio.decreto:1930-10-19;1398')) {
      return `Art. ${articleNum} c.p.`;
    }
    if (urnOrLabel.includes('codice.procedura.civile')) {
      return `Art. ${articleNum} c.p.c.`;
    }
    if (urnOrLabel.includes('codice.procedura.penale')) {
      return `Art. ${articleNum} c.p.p.`;
    }
    if (urnOrLabel.includes('costituzione')) {
      return `Art. ${articleNum} Cost.`;
    }

    // Default: solo articolo
    return `Art. ${articleNum}`;
  }

  // Fallback: prova a estrarre numero dalla label
  const numMatch = urnOrLabel.match(/(?:art(?:icolo)?\.?\s*)?(\d+)/i);
  if (numMatch) {
    return `Art. ${numMatch[1]}`;
  }

  return truncateWithEllipsis(urnOrLabel, 15);
}

/**
 * Abbrevia riferimenti a sentenze di Cassazione e altre corti.
 *
 * @example
 * "Cassazione civile 12345/2020" → "Cass. 12345/2020"
 * "Cassazione penale Sez. Un. 9999/2019" → "Cass. SU 9999/2019"
 */
export function abbreviateCase(label: string): string {
  // Pattern: numero/anno
  const caseMatch = label.match(/(\d+)\/(\d{4})/);
  if (!caseMatch) {
    return truncateWithEllipsis(label, 18);
  }

  const caseNum = caseMatch[1];
  const year = caseMatch[2];

  // Sezioni Unite
  if (label.toLowerCase().includes('sez') && label.toLowerCase().includes('un')) {
    return `Cass. SU ${caseNum}/${year}`;
  }

  // Cassazione generica
  if (label.toLowerCase().includes('cassazione')) {
    return `Cass. ${caseNum}/${year}`;
  }

  // Corte Costituzionale
  if (label.toLowerCase().includes('costituzionale')) {
    return `C.Cost. ${caseNum}/${year}`;
  }

  // TAR
  if (label.toLowerCase().includes('tar')) {
    return `TAR ${caseNum}/${year}`;
  }

  // Consiglio di Stato
  if (label.toLowerCase().includes('consiglio') && label.toLowerCase().includes('stato')) {
    return `CdS ${caseNum}/${year}`;
  }

  // Default: solo numero/anno
  return `${caseNum}/${year}`;
}

/**
 * Tronca una stringa aggiungendo ellipsis se necessario.
 *
 * @param text - Testo da troncare
 * @param maxLength - Lunghezza massima (default 18)
 */
export function truncateWithEllipsis(text: string, maxLength: number = 18): string {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3).trim() + '...';
}

/**
 * Genera una smart label per un nodo del grafo basandosi sul tipo.
 *
 * @param node - Nodo del subgraph
 * @returns Label ottimizzata per la visualizzazione
 */
export function getSmartLabel(node: SubgraphNode): string {
  const { type, label, urn } = node;

  // Norme e articoli
  if (type === 'Norma' || type === 'Comma' || type === 'Versione') {
    return extractArticleLabel(urn || label);
  }

  // Giurisprudenza (Cassazione, sentenze, etc.)
  if (type === 'Precedente' || type === 'Dottrina' || type === 'Brocardo') {
    // Check se contiene riferimento a sentenza
    if (label.includes('/') && /\d{4}/.test(label)) {
      return abbreviateCase(label);
    }
    // Brocardi: mantieni nome breve
    if (type === 'Brocardo') {
      return truncateWithEllipsis(label, 25);
    }
    return truncateWithEllipsis(label, 20);
  }

  // Concetti ed entità
  if (type === 'Concetto' || type === 'Entity' || type === 'Principio') {
    return truncateWithEllipsis(label, 20);
  }

  // Soggetti e ruoli
  if (type === 'Soggetto' || type === 'Ruolo') {
    return truncateWithEllipsis(label, 15);
  }

  // Default: troncamento generico
  return truncateWithEllipsis(label, 15);
}

/**
 * Formatta un URN completo in una forma leggibile.
 * Usato principalmente per tooltip e panel dettagli.
 *
 * @example
 * "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:codice.civile:1942~art1218"
 * → "Codice Civile, Art. 1218"
 */
export function formatUrnReadable(urn: string): string {
  if (!urn) return '';

  // Estrai codice
  let codeName = 'Normativa';
  if (urn.includes('codice.civile') || urn.includes('regio.decreto:1942-03-16;262')) {
    codeName = 'Codice Civile';
  } else if (urn.includes('codice.penale') || urn.includes('regio.decreto:1930-10-19;1398')) {
    codeName = 'Codice Penale';
  } else if (urn.includes('codice.procedura.civile')) {
    codeName = 'Cod. Proc. Civile';
  } else if (urn.includes('codice.procedura.penale')) {
    codeName = 'Cod. Proc. Penale';
  } else if (urn.includes('costituzione')) {
    codeName = 'Costituzione';
  }

  // Estrai articolo
  const artMatch = urn.match(/~art(\d+(?:bis|ter|quater|quinquies)?)/i);
  if (artMatch) {
    return `${codeName}, Art. ${artMatch[1]}`;
  }

  return codeName;
}
