/**
 * Tree Transform Utility
 *
 * Transforms flat backend tree data into a hierarchical structure
 * (Codice → Libro → Titolo → Capo → Sezione → Articolo).
 */

export interface HierarchicalNode {
  id: string;
  type: 'libro' | 'titolo' | 'capo' | 'sezione' | 'articolo' | 'group';
  label: string;
  numero?: string;
  children: HierarchicalNode[];
  articleCount: number;
}

// Section hierarchy levels in Italian legal documents
const SECTION_PATTERNS: Array<{
  type: HierarchicalNode['type'];
  regex: RegExp;
}> = [
  { type: 'libro', regex: /^LIBRO\s+/i },
  { type: 'titolo', regex: /^TITOLO\s+/i },
  { type: 'capo', regex: /^CAPO\s+/i },
  { type: 'sezione', regex: /^SEZIONE\s+/i },
];

function detectSectionType(title: string): HierarchicalNode['type'] {
  for (const { type, regex } of SECTION_PATTERNS) {
    if (regex.test(title)) return type;
  }
  return 'group';
}

function getSectionDepth(type: HierarchicalNode['type']): number {
  switch (type) {
    case 'libro': return 0;
    case 'titolo': return 1;
    case 'capo': return 2;
    case 'sezione': return 3;
    case 'group': return 4;
    case 'articolo': return 5;
    default: return 5;
  }
}

/**
 * Checks if a string represents an article number.
 */
function isArticleString(str: string): boolean {
  if (!str || typeof str !== 'string') return false;
  const trimmed = str.trim();
  if (/^(\d+)(?:[-\s]?(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?$/i.test(trimmed)) {
    return true;
  }
  if (/^([IVXLCDM]+)(?:[-\s]?(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?$/i.test(trimmed)) {
    return true;
  }
  return false;
}

/**
 * Build a hierarchical tree from flat backend data.
 *
 * Input: mixed array of section titles (strings) and article items
 * (objects with {numero, allegato} or plain article-number strings).
 *
 * Output: nested HierarchicalNode[] tree.
 */
export function buildHierarchicalTree(
  data: unknown[],
  targetAnnex: string | null = null
): HierarchicalNode[] {
  if (!data || data.length === 0) return [];

  // First pass: build a flat ordered list of sections and articles
  interface FlatItem {
    kind: 'section' | 'article';
    label: string;
    type: HierarchicalNode['type'];
    articleNum?: string;
  }

  const flat: FlatItem[] = [];

  for (const item of data) {
    if (typeof item === 'string') {
      if (isArticleString(item)) {
        // Legacy string article (dispositivo only)
        if (targetAnnex === null) {
          flat.push({ kind: 'article', label: `Art. ${item}`, type: 'articolo', articleNum: item });
        }
      } else {
        flat.push({ kind: 'section', label: item, type: detectSectionType(item) });
      }
      continue;
    }

    if (item && typeof item === 'object' && 'numero' in (item as Record<string, unknown>)) {
      const obj = item as { numero: string; allegato?: string | null };
      const itemAnnex = obj.allegato ?? null;
      const annexMatches =
        (itemAnnex === null && targetAnnex === null) ||
        (itemAnnex !== null && targetAnnex !== null && String(itemAnnex) === String(targetAnnex));

      if (annexMatches) {
        flat.push({ kind: 'article', label: `Art. ${obj.numero}`, type: 'articolo', articleNum: obj.numero });
      }
    }
  }

  if (flat.length === 0) return [];

  // Check if we have any section structure
  const hasSections = flat.some(f => f.kind === 'section');

  if (!hasSections) {
    // No sections - return flat list of article nodes
    return flat
      .filter(f => f.kind === 'article')
      .map((f, i) => ({
        id: `art-${f.articleNum}-${i}`,
        type: 'articolo' as const,
        label: f.label,
        numero: f.articleNum,
        children: [],
        articleCount: 1,
      }));
  }

  // Build tree using a stack-based approach
  const root: HierarchicalNode[] = [];
  const stack: HierarchicalNode[] = [];

  function countArticles(node: HierarchicalNode): number {
    if (node.type === 'articolo') return 1;
    let count = 0;
    for (const child of node.children) {
      count += countArticles(child);
    }
    node.articleCount = count;
    return count;
  }

  let sectionCounter = 0;
  let articleCounter = 0;

  for (const item of flat) {
    if (item.kind === 'section') {
      const node: HierarchicalNode = {
        id: `sec-${sectionCounter++}`,
        type: item.type,
        label: item.label,
        children: [],
        articleCount: 0,
      };

      const depth = getSectionDepth(item.type);

      // Pop stack until we find a parent with lower depth
      while (stack.length > 0) {
        const parentDepth = getSectionDepth(stack[stack.length - 1].type);
        if (parentDepth < depth) break;
        stack.pop();
      }

      if (stack.length > 0) {
        stack[stack.length - 1].children.push(node);
      } else {
        root.push(node);
      }
      stack.push(node);
    } else {
      // Article
      const articleNode: HierarchicalNode = {
        id: `art-${item.articleNum}-${articleCounter++}`,
        type: 'articolo',
        label: item.label,
        numero: item.articleNum,
        children: [],
        articleCount: 1,
      };

      if (stack.length > 0) {
        stack[stack.length - 1].children.push(articleNode);
      } else {
        root.push(articleNode);
      }
    }
  }

  // Recount articles bottom-up
  for (const node of root) {
    countArticles(node);
  }

  return root;
}
