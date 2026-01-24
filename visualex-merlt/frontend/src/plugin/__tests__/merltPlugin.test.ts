/**
 * Tests for MERLT Plugin
 *
 * Verifies the plugin implements the Plugin interface correctly
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { PluginContext } from '@visualex/platform/lib/plugins';

// We'll dynamically import to test the module loading
describe('MERLT Plugin', () => {
  let plugin: Awaited<typeof import('../index')>['default'];
  let mockContext: PluginContext;

  beforeEach(async () => {
    // Reset modules
    vi.resetModules();

    // Import plugin
    const module = await import('../index');
    plugin = module.default;

    // Create mock context
    mockContext = {
      user: { id: 'user-1', features: ['merlt'] },
      apiBaseUrl: 'http://localhost:3000',
      emit: vi.fn(),
      getAuthToken: vi.fn().mockResolvedValue('test-token'),
    };
  });

  describe('Manifest', () => {
    it('has correct id', () => {
      expect(plugin.manifest.id).toBe('merlt');
    });

    it('requires merlt feature', () => {
      expect(plugin.manifest.requiredFeatures).toContain('merlt');
    });

    it('subscribes to article events', () => {
      expect(plugin.manifest.subscribedEvents).toContain('article:viewed');
    });

    it('contributes to article-sidebar slot', () => {
      expect(plugin.manifest.contributedSlots).toContain('article-sidebar');
    });
  });

  describe('initialize()', () => {
    it('returns cleanup function', async () => {
      const cleanup = await plugin.initialize(mockContext);
      expect(typeof cleanup).toBe('function');
    });

    it('can be called multiple times', async () => {
      const cleanup1 = await plugin.initialize(mockContext);
      cleanup1();

      const cleanup2 = await plugin.initialize(mockContext);
      expect(typeof cleanup2).toBe('function');
      cleanup2();
    });
  });

  describe('getSlotComponents()', () => {
    it('returns array of slot components', () => {
      const components = plugin.getSlotComponents();
      expect(Array.isArray(components)).toBe(true);
      expect(components.length).toBeGreaterThan(0);
    });

    it('includes article-sidebar component', () => {
      const components = plugin.getSlotComponents();
      const sidebarComponent = components.find((c) => c.slot === 'article-sidebar');
      expect(sidebarComponent).toBeDefined();
      expect(sidebarComponent?.component).toBeDefined();
    });

    it('components have priority', () => {
      const components = plugin.getSlotComponents();
      components.forEach((c) => {
        expect(typeof c.priority).toBe('number');
      });
    });
  });

  describe('getEventHandlers()', () => {
    it('returns object with event handlers', () => {
      const handlers = plugin.getEventHandlers();
      expect(typeof handlers).toBe('object');
    });

    it('has handler for article:viewed', () => {
      const handlers = plugin.getEventHandlers();
      expect(handlers['article:viewed']).toBeDefined();
      expect(typeof handlers['article:viewed']).toBe('function');
    });

    it('article:viewed handler processes event', async () => {
      await plugin.initialize(mockContext);
      const handlers = plugin.getEventHandlers();

      // Should not throw
      handlers['article:viewed']?.({
        urn: 'urn:nir:stato:codice.civile~art1453',
        articleId: 'art-1',
        userId: 'user-1',
      });
    });
  });
});

describe('Plugin Dynamic Import', () => {
  it('exports default plugin', async () => {
    const module = await import('../index');
    expect(module.default).toBeDefined();
    expect(module.default.manifest).toBeDefined();
    expect(module.default.initialize).toBeDefined();
    expect(module.default.getSlotComponents).toBeDefined();
    expect(module.default.getEventHandlers).toBeDefined();
  });
});
