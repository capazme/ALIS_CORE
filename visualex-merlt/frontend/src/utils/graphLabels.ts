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
 */
export function extractArticleLabel(urnOrLabel: string): string {
  const artMatch = urnOrLabel.match(/~art(\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)/i);

  if (artMatch) {
    const articleNum = artMatch[1];

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

    return `Art. ${articleNum}`;
  }

  const numMatch = urnOrLabel.match(/(?:art(?:icolo)?\.?\s*)?(\d+)/i);
  if (numMatch) {
    return `Art. ${numMatch[1]}`;
  }

  return truncateWithEllipsis(urnOrLabel, 15);
}

/**
 * Abbrevia riferimenti a sentenze di Cassazione e altre corti.
 */
export function abbreviateCase(label: string): string {
  const caseMatch = label.match(/(\d+)\/(\d{4})/);
  if (!caseMatch) {
    return truncateWithEllipsis(label, 18);
  }

  const caseNum = caseMatch[1];
  const year = caseMatch[2];

  if (label.toLowerCase().includes('sez') && label.toLowerCase().includes('un')) {
    return `Cass. SU ${caseNum}/${year}`;
  }

  if (label.toLowerCase().includes('cassazione')) {
    return `Cass. ${caseNum}/${year}`;
  }

  if (label.toLowerCase().includes('costituzionale')) {
    return `C.Cost. ${caseNum}/${year}`;
  }

  if (label.toLowerCase().includes('tar')) {
    return `TAR ${caseNum}/${year}`;
  }

  if (label.toLowerCase().includes('consiglio') && label.toLowerCase().includes('stato')) {
    return `CdS ${caseNum}/${year}`;
  }

  return `${caseNum}/${year}`;
}

/**
 * Tronca una stringa aggiungendo ellipsis se necessario.
 */
export function truncateWithEllipsis(text: string, maxLength: number = 18): string {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3).trim() + '...';
}

/**
 * Genera una smart label per un nodo del grafo basandosi sul tipo.
 */
export function getSmartLabel(node: SubgraphNode): string {
  const { type, label, urn } = node;

  if (type === 'Norma' || type === 'Comma' || type === 'Versione') {
    return extractArticleLabel(urn || label);
  }

  if (type === 'Precedente' || type === 'Dottrina' || type === 'Brocardo') {
    if (label.includes('/') && /\d{4}/.test(label)) {
      return abbreviateCase(label);
    }
    if (type === 'Brocardo') {
      return truncateWithEllipsis(label, 25);
    }
    return truncateWithEllipsis(label, 20);
  }

  if (type === 'Concetto' || type === 'Entity' || type === 'Principio') {
    return truncateWithEllipsis(label, 20);
  }

  if (type === 'Soggetto' || type === 'Ruolo') {
    return truncateWithEllipsis(label, 15);
  }

  return truncateWithEllipsis(label, 15);
}

/**
 * Formatta un URN completo in una forma leggibile.
 */
export function formatUrnReadable(urn: string): string {
  if (!urn) return '';

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

  const artMatch = urn.match(/~art(\d+(?:bis|ter|quater|quinquies)?)/i);
  if (artMatch) {
    return `${codeName}, Art. ${artMatch[1]}`;
  }

  return codeName;
}
