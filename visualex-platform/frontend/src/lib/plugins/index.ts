/**
 * Plugin System
 *
 * Provides extension points for optional features like MERLT.
 *
 * Usage in visualex-platform:
 *
 * ```tsx
 * // App.tsx
 * import { PluginProvider, PluginSlot } from '@/lib/plugins';
 *
 * const plugins = [
 *   {
 *     id: 'merlt',
 *     enabled: true,
 *     loader: () => import('@visualex/merlt-plugin'),
 *   },
 * ];
 *
 * function App() {
 *   const { user } = useAuth();
 *
 *   return (
 *     <PluginProvider
 *       plugins={plugins}
 *       user={user ? { id: user.id, features: user.features } : null}
 *       apiBaseUrl={import.meta.env.VITE_API_URL}
 *       getAuthToken={getAuthToken}
 *     >
 *       <YourApp />
 *     </PluginProvider>
 *   );
 * }
 *
 * // ArticleView.tsx
 * function ArticleView({ urn, articleId }) {
 *   const { emit } = usePlugins();
 *
 *   useEffect(() => {
 *     emit('article:viewed', { urn, articleId });
 *   }, [urn, articleId, emit]);
 *
 *   return (
 *     <div className="flex">
 *       <main>
 *         <ArticleContent />
 *       </main>
 *       <aside>
 *         <PluginSlot
 *           name="article-sidebar"
 *           props={{ urn, articleId }}
 *           fallback={null}
 *         />
 *       </aside>
 *     </div>
 *   );
 * }
 * ```
 */

// Types
export type {
  Plugin,
  PluginManifest,
  PluginConfig,
  PluginContext,
  PluginEvents,
  PluginEventName,
  PluginEventHandler,
  PluginSlotName,
  SlotProps,
  SlotComponent,
  PluginLoader,
} from './types';

// Core
export { EventBus } from './EventBus';
export { PluginRegistry } from './PluginRegistry';

// React Components
export { PluginSlot } from './PluginSlot';
export type { PluginSlotProps } from './PluginSlot';

export { PluginProvider, usePlugins, usePluginFeature } from './PluginProvider';
export type { PluginProviderProps } from './PluginProvider';
