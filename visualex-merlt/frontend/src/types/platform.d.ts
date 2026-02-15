/**
 * Type declarations for @visualex/platform/lib/plugins
 *
 * This external module is provided by the visualex-platform host at runtime.
 * These declarations allow TypeScript compilation without the actual package.
 */
declare module '@visualex/platform/lib/plugins' {
  import type { ComponentType } from 'react';

  export interface PluginManifest {
    id: string;
    name: string;
    version: string;
    description?: string;
    requiredFeatures?: string[];
    subscribedEvents?: string[];
    contributedSlots?: string[];
  }

  export interface PluginContext {
    apiBaseUrl: string;
    getAuthToken: () => Promise<string | null>;
    user?: {
      id: string;
      [key: string]: unknown;
    };
  }

  export interface SlotComponent {
    slot: string;
    component: ComponentType<Record<string, unknown>>;
    priority?: number;
  }

  export interface PluginEventMap {
    'article:viewed': { urn: string; articleId: string; userId: string };
    'article:highlighted': {
      urn: string;
      text: string;
      startOffset: number;
      endOffset: number;
    };
    'article:text-selected': {
      urn: string;
      text: string;
      startOffset: number;
      endOffset: number;
      [key: string]: unknown;
    };
    'citation:detected': {
      urn: string;
      text: string;
      parsed?: Record<string, unknown>;
      [key: string]: unknown;
    };
    'search:performed': {
      query: string;
      filters: Record<string, unknown>;
      resultCount: number;
    };
  }

  export type PluginEventHandler<E extends keyof PluginEventMap = keyof PluginEventMap> = (
    data: PluginEventMap[E],
  ) => void;

  export interface EventBusInterface {
    emit<E extends keyof PluginEventMap>(event: E, data: PluginEventMap[E]): void;
    emit(event: string, data?: unknown): void;
    on<E extends keyof PluginEventMap>(event: E, handler: PluginEventHandler<E>): void;
    on(event: string, handler: PluginEventHandler): void;
    off<E extends keyof PluginEventMap>(event: E, handler: PluginEventHandler<E>): void;
    off(event: string, handler: PluginEventHandler): void;
  }

  export type EventBus = EventBusInterface;
  export declare const EventBus: EventBusInterface;

  export type PluginEvents = PluginEventMap;

  export interface SlotProps {
    'article-toolbar': {
      urn: string;
      articleId: string;
    };
    'article-sidebar': {
      urn: string;
      articleId: string;
    };
    'article-content-overlay': {
      urn: string;
      articleId: string;
      contentRef: React.RefObject<HTMLElement>;
    };
    'profile-tabs': {
      userId: string;
    };
    'admin-dashboard': Record<string, unknown>;
    'bulletin-board': {
      userId: string;
    };
    'dossier-actions': {
      dossierId: string;
      userId: string;
      dossier: {
        title: string;
        description: string;
        tags?: string[];
        items: Array<{
          type: string;
          status?: string;
          data: Record<string, unknown>;
        }>;
      };
    };
    'graph-view': {
      rootUrn: string;
      depth: number;
      userId: string;
    };
    context?: Record<string, unknown>;
    className?: string;
  }

  export interface Plugin {
    manifest: PluginManifest;
    initialize(context: PluginContext): Promise<() => void>;
    getSlotComponents(): SlotComponent[];
    getEventHandlers(): Partial<{
      [E in keyof PluginEventMap]: PluginEventHandler<E>;
    }>;
  }
}
