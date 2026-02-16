/**
 * Runtime mock for @visualex/platform/lib/plugins
 *
 * The real module is provided by visualex-platform at runtime.
 * This mock allows Vitest to resolve imports during testing.
 */

const listeners = new Map<string, Set<(...args: unknown[]) => void>>();

export const EventBus = {
  emit(event: string, data?: unknown) {
    const handlers = listeners.get(event);
    if (handlers) {
      handlers.forEach((h) => h(data));
    }
  },
  on(event: string, handler: (...args: unknown[]) => void) {
    if (!listeners.has(event)) {
      listeners.set(event, new Set());
    }
    listeners.get(event)!.add(handler);
  },
  off(event: string, handler: (...args: unknown[]) => void) {
    listeners.get(event)?.delete(handler);
  },
};
