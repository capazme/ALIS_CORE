/**
 * Synthesis Parser - Maps synthesis text to source references.
 *
 * Detects inline references like [1], [2], [3] and maps them to source indices.
 */

export interface SynthesisSegment {
  type: 'text' | 'source-ref';
  content: string;
  sourceIndex?: number; // 0-based index into sources array
}

/**
 * Parse synthesis text and extract source references.
 *
 * Input: "La norma prevede [1] che il soggetto [2] debba..."
 * Output: [
 *   { type: 'text', content: 'La norma prevede ' },
 *   { type: 'source-ref', content: '[1]', sourceIndex: 0 },
 *   { type: 'text', content: ' che il soggetto ' },
 *   { type: 'source-ref', content: '[2]', sourceIndex: 1 },
 *   { type: 'text', content: ' debba...' },
 * ]
 */
export function parseSynthesisWithSources(text: string): SynthesisSegment[] {
  if (!text) return [];

  const segments: SynthesisSegment[] = [];
  const regex = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    // Text before the reference
    if (match.index > lastIndex) {
      segments.push({
        type: 'text',
        content: text.slice(lastIndex, match.index),
      });
    }

    // Source reference (1-based in text, convert to 0-based index)
    const sourceNum = parseInt(match[1], 10);
    const sourceIndex = Math.max(0, sourceNum - 1);
    segments.push({
      type: 'source-ref',
      content: match[0],
      sourceIndex,
    });

    lastIndex = match.index + match[0].length;
  }

  // Remaining text after last reference
  if (lastIndex < text.length) {
    segments.push({
      type: 'text',
      content: text.slice(lastIndex),
    });
  }

  return segments;
}

/**
 * Count unique source references in synthesis text.
 */
export function countSourceReferences(text: string): number {
  const refs = new Set<number>();
  const regex = /\[(\d+)\]/g;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    refs.add(parseInt(match[1], 10));
  }

  return refs.size;
}
