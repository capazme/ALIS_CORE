/**
 * MERLT Plugin for visualex-platform
 *
 * This is the entry point for the MERLT plugin. It's loaded dynamically
 * by visualex-platform when the user has the 'merlt' feature flag.
 *
 * The plugin provides:
 * - Article analysis sidebar (entity extraction, validation UI)
 * - Event tracking for research data collection
 * - User contribution components
 */

import type {
  Plugin,
  PluginManifest,
  PluginContext,
  PluginEventHandler,
  SlotComponent,
} from '@visualex/platform/lib/plugins';

import { MerltSidebarPanel } from '../components/MerltSidebarPanel';
import { MerltToolbar } from '../components/MerltToolbar';
import { MerltContentOverlay } from '../components/MerltContentOverlay';
import { BulletinBoardSlot } from '../components/slots/BulletinBoardSlot';
import { DossierActionsSlot } from '../components/slots/DossierActionsSlot';
import { GraphViewSlot } from '../components/slots/GraphViewSlot';
import { ProfilePage } from '../components/merlt/profile/ProfilePage';
import { AcademicDashboard } from '../components/merlt/dashboard/AcademicDashboard';
import { initializeMerltServices, shutdownMerltServices } from '../services/merltInit';
import { trackArticleView, trackSearch, trackHighlight } from '../services/tracking';

const manifest: PluginManifest = {
  id: 'merlt',
  name: 'MERLT Research',
  version: '1.0.0',
  description: 'Legal knowledge extraction and validation for research',

  // User must have 'merlt' feature to use this plugin
  requiredFeatures: ['merlt'],

  // Events we listen to
  subscribedEvents: [
    'article:viewed',
    'article:highlighted',
    'article:text-selected',
    'citation:detected',
    'search:performed',
  ],

  // Events we emit (for documentation; emitted via EventBus.emit)
  // 'merlt:source-navigate' - navigates to a source article in the platform

  // UI slots we provide
  contributedSlots: [
    'article-sidebar',
    'article-toolbar',
    'article-content-overlay',
    'bulletin-board',
    'dossier-actions',
    'graph-view',
    'profile-tabs',
    'admin-dashboard',
  ],
};

const merltPlugin: Plugin = {
  manifest,

  async initialize(context: PluginContext): Promise<() => void> {
    // Initialize MERLT backend connection
    await initializeMerltServices({
      apiBaseUrl: context.apiBaseUrl,
      getAuthToken: context.getAuthToken,
      userId: context.user?.id,
    });

    // Return cleanup function
    return () => {
      shutdownMerltServices();
    };
  },

  getSlotComponents(): SlotComponent[] {
    return [
      {
        slot: 'article-sidebar',
        component: MerltSidebarPanel,
        priority: 100, // High priority = rendered first
      },
      {
        slot: 'article-toolbar',
        component: MerltToolbar,
        priority: 50,
      },
      {
        slot: 'article-content-overlay',
        component: MerltContentOverlay,
        priority: 100,
      },
      {
        slot: 'bulletin-board',
        component: BulletinBoardSlot,
        priority: 100,
      },
      {
        slot: 'dossier-actions',
        component: DossierActionsSlot,
        priority: 100,
      },
      {
        slot: 'graph-view',
        component: GraphViewSlot,
        priority: 100,
      },
      {
        slot: 'profile-tabs',
        component: ProfilePage,
        priority: 100,
      },
      {
        slot: 'admin-dashboard',
        component: AcademicDashboard as unknown as SlotComponent['component'],
        priority: 100,
      },
    ];
  },

  getEventHandlers(): Partial<{
    'article:viewed': PluginEventHandler<'article:viewed'>;
    'article:highlighted': PluginEventHandler<'article:highlighted'>;
    'search:performed': PluginEventHandler<'search:performed'>;
  }> {
    return {
      'article:viewed': (data: { urn: string; articleId: string; userId: string }) => {
        // Track for research data collection
        trackArticleView(data.urn, data.articleId, data.userId);
      },

      'article:highlighted': (data: { urn: string; text: string; startOffset: number; endOffset: number }) => {
        // Track text selections for potential entity proposals
        trackHighlight(data.urn, data.text, data.startOffset, data.endOffset);
      },

      'search:performed': (data: { query: string; filters: Record<string, unknown>; resultCount: number }) => {
        // Track search patterns for research
        trackSearch(data.query, data.filters, data.resultCount);
      },
    };
  },
};

export default merltPlugin;
