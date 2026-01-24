/**
 * GraphViewSlot - Plugin slot for workspace graph view
 *
 * Provides the GraphViewContent component that was previously
 * imported directly in WorkspaceTabPanel.
 */

import { GraphViewContent } from '../features/workspace/GraphViewContent';
import type { SlotProps } from '@visualex/platform/lib/plugins';

export function GraphViewSlot({ rootUrn, depth, userId }: SlotProps['graph-view']) {
  return (
    <GraphViewContent
      rootUrn={rootUrn}
      depth={depth}
      userId={userId}
    />
  );
}
