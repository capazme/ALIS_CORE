/**
 * Integration tests for Plugin System
 *
 * Tests dynamic loading, feature flags, and event communication
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { PluginRegistry } from '../PluginRegistry';
import { EventBus } from '../EventBus';
import type { Plugin, PluginManifest, PluginContext } from '../types';

// Mock MERLT plugin for testing
const createMockPlugin = (overrides: Partial<Plugin> = {}): Plugin => {
  const manifest: PluginManifest = {
    id: 'test-merlt',
    name: 'Test MERLT Plugin',
    version: '1.0.0',
    description: 'Test plugin',
    requiredFeatures: ['merlt'],
    subscribedEvents: ['article:viewed'],
    contributedSlots: ['article-sidebar'],
  };

  return {
    manifest,
    initialize: vi.fn().mockResolvedValue(() => {}),
    getSlotComponents: vi.fn().mockReturnValue([]),
    getEventHandlers: vi.fn().mockReturnValue({}),
    ...overrides,
  };
};

describe('PluginRegistry', () => {
  beforeEach(() => {
    // Clear registry state
    PluginRegistry.reset();
    EventBus.clear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Plugin Registration', () => {
    it('registers plugin configuration', () => {
      const mockLoader = vi.fn().mockResolvedValue({ default: createMockPlugin() });

      PluginRegistry.register({
        id: 'merlt',
        enabled: true,
        loader: mockLoader,
      });

      // Plugin should not be loaded yet
      expect(PluginRegistry.isLoaded('merlt')).toBe(false);
    });
  });

  describe('Feature-based Loading', () => {
    it('loads plugin only when user has required features', async () => {
      const mockPlugin = createMockPlugin();
      const mockLoader = vi.fn().mockResolvedValue({ default: mockPlugin });

      PluginRegistry.register({
        id: 'merlt',
        enabled: true,
        loader: mockLoader,
      });

      // Configure with user that has 'merlt' feature
      PluginRegistry.configure({
        user: { id: 'user-1', features: ['merlt'] },
        apiBaseUrl: 'http://localhost:3000',
        getAuthToken: async () => 'test-token',
      });

      await PluginRegistry.loadPlugins();

      expect(mockLoader).toHaveBeenCalled();
      expect(mockPlugin.initialize).toHaveBeenCalled();
      expect(PluginRegistry.isLoaded('merlt')).toBe(true);
    });

    it('does not load plugin when user lacks required features', async () => {
      const mockPlugin = createMockPlugin();
      const mockLoader = vi.fn().mockResolvedValue({ default: mockPlugin });

      PluginRegistry.register({
        id: 'merlt',
        enabled: true,
        loader: mockLoader,
      });

      // Configure with user that does NOT have 'merlt' feature
      PluginRegistry.configure({
        user: { id: 'user-1', features: ['other-feature'] },
        apiBaseUrl: 'http://localhost:3000',
        getAuthToken: async () => 'test-token',
      });

      await PluginRegistry.loadPlugins();

      expect(mockLoader).toHaveBeenCalled();
      expect(mockPlugin.initialize).not.toHaveBeenCalled();
      expect(PluginRegistry.isLoaded('merlt')).toBe(false);
    });

    it('unloads plugin when user logs out', async () => {
      const cleanup = vi.fn();
      const mockPlugin = createMockPlugin({
        initialize: vi.fn().mockResolvedValue(cleanup),
      });
      const mockLoader = vi.fn().mockResolvedValue({ default: mockPlugin });

      PluginRegistry.register({
        id: 'merlt',
        enabled: true,
        loader: mockLoader,
      });

      // First, load with user
      PluginRegistry.configure({
        user: { id: 'user-1', features: ['merlt'] },
        apiBaseUrl: 'http://localhost:3000',
        getAuthToken: async () => 'test-token',
      });
      await PluginRegistry.loadPlugins();
      expect(PluginRegistry.isLoaded('merlt')).toBe(true);

      // Then, simulate logout (no user)
      PluginRegistry.configure({
        user: null,
        apiBaseUrl: 'http://localhost:3000',
        getAuthToken: async () => null,
      });
      await PluginRegistry.loadPlugins();

      expect(cleanup).toHaveBeenCalled();
      expect(PluginRegistry.isLoaded('merlt')).toBe(false);
    });
  });

  describe('Plugin Disabled', () => {
    it('does not load disabled plugins even with features', async () => {
      const mockPlugin = createMockPlugin();
      const mockLoader = vi.fn().mockResolvedValue({ default: mockPlugin });

      PluginRegistry.register({
        id: 'merlt',
        enabled: false, // Plugin disabled globally
        loader: mockLoader,
      });

      PluginRegistry.configure({
        user: { id: 'user-1', features: ['merlt'] },
        apiBaseUrl: 'http://localhost:3000',
        getAuthToken: async () => 'test-token',
      });

      await PluginRegistry.loadPlugins();

      expect(mockPlugin.initialize).not.toHaveBeenCalled();
      expect(PluginRegistry.isLoaded('merlt')).toBe(false);
    });
  });
});

describe('EventBus', () => {
  beforeEach(() => {
    EventBus.clear();
  });

  it('delivers events to subscribers', () => {
    const handler = vi.fn();

    EventBus.on('article:viewed', handler);
    EventBus.emit('article:viewed', { urn: 'urn:test', articleId: 'art-1' });

    expect(handler).toHaveBeenCalledWith({
      urn: 'urn:test',
      articleId: 'art-1',
    });
  });

  it('supports multiple subscribers', () => {
    const handler1 = vi.fn();
    const handler2 = vi.fn();

    EventBus.on('article:viewed', handler1);
    EventBus.on('article:viewed', handler2);
    EventBus.emit('article:viewed', { urn: 'urn:test', articleId: 'art-1' });

    expect(handler1).toHaveBeenCalled();
    expect(handler2).toHaveBeenCalled();
  });

  it('returns unsubscribe function', () => {
    const handler = vi.fn();

    const unsubscribe = EventBus.on('article:viewed', handler);
    unsubscribe();

    EventBus.emit('article:viewed', { urn: 'urn:test', articleId: 'art-1' });

    expect(handler).not.toHaveBeenCalled();
  });

  it('once() triggers handler only once', () => {
    const handler = vi.fn();

    EventBus.once('article:viewed', handler);
    EventBus.emit('article:viewed', { urn: 'urn:test', articleId: 'art-1' });
    EventBus.emit('article:viewed', { urn: 'urn:test', articleId: 'art-2' });

    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('keeps event history', () => {
    EventBus.emit('article:viewed', { urn: 'urn:1', articleId: 'art-1' });
    EventBus.emit('search:performed', { query: 'test', filters: {}, resultCount: 10 });
    EventBus.emit('article:viewed', { urn: 'urn:2', articleId: 'art-2' });

    const history = EventBus.getHistory();
    expect(history).toHaveLength(3);

    const articleHistory = EventBus.getHistory('article:viewed');
    expect(articleHistory).toHaveLength(2);
  });
});

describe('Plugin-EventBus Integration', () => {
  beforeEach(() => {
    PluginRegistry.reset();
    EventBus.clear();
  });

  it('plugin receives events after registration', async () => {
    const eventHandler = vi.fn();
    const mockPlugin = createMockPlugin({
      getEventHandlers: () => ({
        'article:viewed': eventHandler,
      }),
    });
    const mockLoader = vi.fn().mockResolvedValue({ default: mockPlugin });

    PluginRegistry.register({
      id: 'merlt',
      enabled: true,
      loader: mockLoader,
    });

    PluginRegistry.configure({
      user: { id: 'user-1', features: ['merlt'] },
      apiBaseUrl: 'http://localhost:3000',
      getAuthToken: async () => 'test-token',
    });

    await PluginRegistry.loadPlugins();

    // Emit event after plugin is loaded
    EventBus.emit('article:viewed', { urn: 'urn:test', articleId: 'art-1' });

    expect(eventHandler).toHaveBeenCalledWith({
      urn: 'urn:test',
      articleId: 'art-1',
    });
  });
});

describe('Slot Components', () => {
  beforeEach(() => {
    PluginRegistry.reset();
    EventBus.clear();
  });

  it('returns slot components from loaded plugins', async () => {
    const TestComponent = () => null;
    const mockPlugin = createMockPlugin({
      getSlotComponents: () => [
        { slot: 'article-sidebar', component: TestComponent, priority: 100 },
      ],
    });
    const mockLoader = vi.fn().mockResolvedValue({ default: mockPlugin });

    PluginRegistry.register({
      id: 'merlt',
      enabled: true,
      loader: mockLoader,
    });

    PluginRegistry.configure({
      user: { id: 'user-1', features: ['merlt'] },
      apiBaseUrl: 'http://localhost:3000',
      getAuthToken: async () => 'test-token',
    });

    await PluginRegistry.loadPlugins();

    const components = PluginRegistry.getSlotComponents('article-sidebar');
    expect(components).toHaveLength(1);
    expect(components[0].component).toBe(TestComponent);
    expect(components[0].priority).toBe(100);
  });

  it('returns empty array when no plugins loaded', () => {
    const components = PluginRegistry.getSlotComponents('article-sidebar');
    expect(components).toHaveLength(0);
  });
});
