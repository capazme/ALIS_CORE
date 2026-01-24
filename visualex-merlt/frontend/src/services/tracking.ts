/**
 * MERLT Research Tracking
 *
 * Collects anonymized interaction data for research purposes.
 * Only active for users who have opted into MERLT research.
 */

import { getMerltConfig } from './merltInit';

interface TrackingEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: number;
}

const eventQueue: TrackingEvent[] = [];
const BATCH_SIZE = 10;
const FLUSH_INTERVAL = 30000; // 30 seconds

let flushTimer: ReturnType<typeof setInterval> | null = null;

/**
 * Start the tracking flush timer
 */
export function startTracking(): void {
  if (!flushTimer) {
    flushTimer = setInterval(flushEvents, FLUSH_INTERVAL);
  }
}

/**
 * Stop tracking and flush remaining events
 */
export function stopTracking(): void {
  if (flushTimer) {
    clearInterval(flushTimer);
    flushTimer = null;
  }
  flushEvents();
}

/**
 * Track an article view
 */
export function trackArticleView(urn: string, articleId: string, userId?: string): void {
  queueEvent('article:viewed', {
    urn,
    articleId,
    userId: userId ? hashUserId(userId) : null,
  });
}

/**
 * Track a text highlight (potential entity proposal)
 */
export function trackHighlight(
  urn: string,
  text: string,
  startOffset: number,
  endOffset: number
): void {
  queueEvent('text:highlighted', {
    urn,
    textLength: text.length,
    startOffset,
    endOffset,
    // Don't send actual text for privacy
  });
}

/**
 * Track a search query
 */
export function trackSearch(
  query: string,
  filters: Record<string, unknown>,
  resultCount: number
): void {
  queueEvent('search:performed', {
    queryLength: query.length,
    queryWordCount: query.split(/\s+/).length,
    hasFilters: Object.keys(filters).length > 0,
    resultCount,
    // Don't send actual query for privacy
  });
}

/**
 * Queue an event for batch sending
 */
function queueEvent(type: string, data: Record<string, unknown>): void {
  eventQueue.push({
    type,
    data,
    timestamp: Date.now(),
  });

  if (eventQueue.length >= BATCH_SIZE) {
    flushEvents();
  }
}

/**
 * Send queued events to MERLT backend
 */
async function flushEvents(): Promise<void> {
  if (eventQueue.length === 0) return;

  const config = getMerltConfig();
  if (!config) return;

  const events = eventQueue.splice(0, eventQueue.length);

  try {
    const token = await config.getAuthToken();

    await fetch(`${config.apiBaseUrl}/merlt/tracking/events`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ events }),
    });
  } catch (error) {
    console.error('[MERLT] Failed to send tracking events:', error);
    // Re-queue failed events (at the front)
    eventQueue.unshift(...events);
  }
}

/**
 * Hash user ID for privacy
 */
function hashUserId(userId: string): string {
  // Simple hash for anonymization
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    const char = userId.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return `user_${Math.abs(hash).toString(36)}`;
}
