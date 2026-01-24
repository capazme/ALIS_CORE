/**
 * Utility per estrarre contesto testuale dal DOM attorno a un elemento.
 *
 * Usato per NER feedback: quando l'utente corregge una citazione,
 * estraiamo il contesto attorno per il training del modello.
 */

export interface DOMContext {
  before: string;      // Testo prima dell'elemento
  selected: string;    // Testo dell'elemento stesso
  after: string;       // Testo dopo l'elemento
  full: string;        // Contesto completo: before + [selected] + after
}

/**
 * Estrae testo plain da un nodo DOM, escludendo tag HTML.
 */
function getPlainText(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.textContent || '';
  }
  if (node.nodeType === Node.ELEMENT_NODE) {
    const element = node as Element;
    // Salta elementi nascosti o script/style
    if (element.tagName === 'SCRIPT' || element.tagName === 'STYLE') {
      return '';
    }
    // Aggiungi newline per elementi block
    const isBlock = ['P', 'DIV', 'BR', 'LI', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(element.tagName);
    let text = '';
    for (const child of element.childNodes) {
      text += getPlainText(child);
    }
    return isBlock ? text + '\n' : text;
  }
  return '';
}

/**
 * Cammina all'indietro nel DOM per raccogliere testo.
 * Si ferma quando ha raccolto abbastanza caratteri o raggiunge il boundary.
 */
function walkBackward(
  startNode: Node,
  maxChars: number,
  boundaryElement?: Element
): string {
  const textParts: string[] = [];
  let currentNode: Node | null = startNode;
  let collectedLength = 0;

  while (currentNode && collectedLength < maxChars) {
    // Se siamo al boundary, fermati
    if (boundaryElement && currentNode === boundaryElement) {
      break;
    }

    // Prova il fratello precedente
    if (currentNode.previousSibling) {
      currentNode = currentNode.previousSibling;
      // Vai al nodo più a destra (ultimo discendente)
      while (currentNode.lastChild) {
        currentNode = currentNode.lastChild;
      }
    } else {
      // Sali al parent
      currentNode = currentNode.parentNode;
      if (!currentNode || (boundaryElement && currentNode === boundaryElement)) {
        break;
      }
      continue; // Non processare il parent, continua a salire
    }

    // Estrai testo se è un text node
    if (currentNode.nodeType === Node.TEXT_NODE) {
      const text = currentNode.textContent || '';
      textParts.unshift(text);
      collectedLength += text.length;
    }
  }

  // Unisci e tronca se necessario
  let result = textParts.join('');
  if (result.length > maxChars) {
    result = result.slice(-maxChars);
  }
  return result;
}

/**
 * Cammina in avanti nel DOM per raccogliere testo.
 * Si ferma quando ha raccolto abbastanza caratteri o raggiunge il boundary.
 */
function walkForward(
  startNode: Node,
  maxChars: number,
  boundaryElement?: Element
): string {
  const textParts: string[] = [];
  let currentNode: Node | null = startNode;
  let collectedLength = 0;

  while (currentNode && collectedLength < maxChars) {
    // Se siamo al boundary, fermati
    if (boundaryElement && currentNode === boundaryElement) {
      break;
    }

    // Prova il fratello successivo
    if (currentNode.nextSibling) {
      currentNode = currentNode.nextSibling;
      // Vai al nodo più a sinistra (primo discendente)
      while (currentNode.firstChild) {
        currentNode = currentNode.firstChild;
      }
    } else {
      // Sali al parent
      currentNode = currentNode.parentNode;
      if (!currentNode || (boundaryElement && currentNode === boundaryElement)) {
        break;
      }
      continue; // Non processare il parent, continua a salire
    }

    // Estrai testo se è un text node
    if (currentNode.nodeType === Node.TEXT_NODE) {
      const text = currentNode.textContent || '';
      textParts.push(text);
      collectedLength += text.length;
    }
  }

  // Unisci e tronca se necessario
  let result = textParts.join('');
  if (result.length > maxChars) {
    result = result.slice(0, maxChars);
  }
  return result;
}

/**
 * Estrae contesto dal DOM attorno a un elemento.
 *
 * @param element - L'elemento di cui estrarre il contesto (es. span.citation-hover)
 * @param charsBefore - Numero di caratteri da estrarre prima (default 500)
 * @param charsAfter - Numero di caratteri da estrarre dopo (default 500)
 * @param boundarySelector - Selettore CSS per il container che limita la ricerca
 * @returns Oggetto con before, selected, after e full
 *
 * @example
 * const context = getContextFromDOM(citationSpan, 500, 500, '.article-content');
 * // context.before = "...Il decreto legislativo 50/2016 disciplina "
 * // context.selected = "art. 3 e 4"
 * // context.after = " in materia di appalti pubblici..."
 * // context.full = "...disciplina [art. 3 e 4] in materia..."
 */
export function getContextFromDOM(
  element: Element,
  charsBefore: number = 500,
  charsAfter: number = 500,
  boundarySelector?: string
): DOMContext {
  // Trova il boundary element se specificato
  const boundaryElement = boundarySelector
    ? element.closest(boundarySelector) || undefined
    : undefined;

  // Testo dell'elemento stesso
  const selected = element.textContent || '';

  // Cammina all'indietro per il contesto "before"
  const before = walkBackward(element, charsBefore, boundaryElement);

  // Cammina in avanti per il contesto "after"
  const after = walkForward(element, charsAfter, boundaryElement);

  // Pulisci whitespace eccessivo
  const cleanBefore = before.replace(/\s+/g, ' ');
  const cleanAfter = after.replace(/\s+/g, ' ');

  // Costruisci il contesto completo con marcatori
  const full = `${cleanBefore}[${selected}]${cleanAfter}`;

  return {
    before: cleanBefore,
    selected,
    after: cleanAfter,
    full,
  };
}

/**
 * Versione semplificata che restituisce solo la stringa di contesto.
 */
export function getContextString(
  element: Element,
  charsBefore: number = 500,
  charsAfter: number = 500,
  boundarySelector?: string
): string {
  const context = getContextFromDOM(element, charsBefore, charsAfter, boundarySelector);
  return context.full;
}
