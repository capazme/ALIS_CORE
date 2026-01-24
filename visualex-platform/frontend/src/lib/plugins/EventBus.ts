/**
 * EventBus for Plugin Communication
 *
 * Provides a pub/sub mechanism for plugins to communicate with visualex core
 * and with each other. Events are typed for safety.
 */

import type { PluginEventName, PluginEvents, PluginEventHandler } from './types';

type EventCallback = PluginEventHandler<PluginEventName>;
type UnsubscribeFn = () => void;

class EventBusImpl {
  private listeners = new Map<PluginEventName, Set<EventCallback>>();
  private eventHistory: Array<{ event: PluginEventName; data: unknown; timestamp: number }> = [];
  private readonly maxHistorySize = 100;

  /**
   * Subscribe to an event
   */
  on<T extends PluginEventName>(event: T, handler: PluginEventHandler<T>): UnsubscribeFn {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }

    const handlers = this.listeners.get(event)!;
    handlers.add(handler as EventCallback);

    return () => {
      handlers.delete(handler as EventCallback);
      if (handlers.size === 0) {
        this.listeners.delete(event);
      }
    };
  }

  /**
   * Subscribe to an event, but only trigger once
   */
  once<T extends PluginEventName>(event: T, handler: PluginEventHandler<T>): UnsubscribeFn {
    const unsubscribe = this.on(event, ((data: PluginEvents[T]) => {
      unsubscribe();
      handler(data);
    }) as PluginEventHandler<T>);

    return unsubscribe;
  }

  /**
   * Emit an event to all subscribers
   */
  emit<T extends PluginEventName>(event: T, data: PluginEvents[T]): void {
    // Record in history
    this.eventHistory.push({
      event,
      data,
      timestamp: Date.now(),
    });

    // Trim history if needed
    if (this.eventHistory.length > this.maxHistorySize) {
      this.eventHistory = this.eventHistory.slice(-this.maxHistorySize);
    }

    // Notify listeners
    const handlers = this.listeners.get(event);
    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(data as PluginEvents[PluginEventName]);
        } catch (error) {
          console.error(`[EventBus] Error in handler for event "${event}":`, error);
        }
      });
    }
  }

  /**
   * Get recent event history (useful for debugging)
   */
  getHistory(event?: PluginEventName): typeof this.eventHistory {
    if (event) {
      return this.eventHistory.filter((e) => e.event === event);
    }
    return [...this.eventHistory];
  }

  /**
   * Clear all listeners (useful for testing)
   */
  clear(): void {
    this.listeners.clear();
    this.eventHistory = [];
  }

  /**
   * Get count of listeners for an event
   */
  listenerCount(event: PluginEventName): number {
    return this.listeners.get(event)?.size ?? 0;
  }
}

// Singleton instance
export const EventBus = new EventBusImpl();

// Export type for testing
export type { EventBusImpl };
