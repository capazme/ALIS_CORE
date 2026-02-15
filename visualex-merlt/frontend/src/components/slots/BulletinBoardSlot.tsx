/**
 * BulletinBoardSlot - Plugin slot for bulletin board / knowledge graph
 *
 * Provides the KnowledgeGraphExplorer and IssueList components
 * that were previously imported directly in BulletinBoardPage.
 */

import { KnowledgeGraphExplorer } from '../features/bulletin/KnowledgeGraphExplorer';
import { IssueList } from '../merlt/IssueList';
import type { SlotProps } from '@visualex/platform/lib/plugins';

export function BulletinBoardSlot({ userId }: SlotProps['bulletin-board']) {
  return (
    <div className="space-y-6">
      <div className="relative">
        <KnowledgeGraphExplorer
          height={650}
          className="w-full"
          userId={userId}
          onArticleClick={(urn) => {
            // TODO: Navigate to article via EventBus
          }}
        />
      </div>
      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <IssueList userId={userId} />
      </div>
    </div>
  );
}
