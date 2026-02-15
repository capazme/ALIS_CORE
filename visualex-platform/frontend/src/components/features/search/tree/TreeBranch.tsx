/**
 * TreeBranch - Recursive container that renders a list of HierarchicalNode at one level.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { TreeNodeItem } from './TreeNodeItem';
import type { HierarchicalNode } from '../../../../utils/treeTransform';

export interface TreeBranchProps {
  nodes: HierarchicalNode[];
  depth?: number;
  isExpanded: (id: string) => boolean;
  onToggle: (id: string) => void;
  onArticleSelect?: (articleNumber: string) => void;
  isArticleLoaded?: (articleNum: string) => boolean;
}

export function TreeBranch({
  nodes,
  depth = 0,
  isExpanded,
  onToggle,
  onArticleSelect,
  isArticleLoaded,
}: TreeBranchProps) {
  if (nodes.length === 0) return null;

  return (
    <div role="group" className="space-y-0.5">
      {nodes.map(node => (
        <div key={node.id}>
          <TreeNodeItem
            node={node}
            isExpanded={isExpanded(node.id)}
            onToggle={() => onToggle(node.id)}
            onArticleSelect={onArticleSelect}
            isArticleLoaded={isArticleLoaded}
            depth={depth}
          />

          {/* Children - animated collapse */}
          <AnimatePresence initial={false}>
            {node.children.length > 0 && isExpanded(node.id) && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2, ease: 'easeInOut' }}
                className="overflow-hidden"
              >
                <TreeBranch
                  nodes={node.children}
                  depth={depth + 1}
                  isExpanded={isExpanded}
                  onToggle={onToggle}
                  onArticleSelect={onArticleSelect}
                  isArticleLoaded={isArticleLoaded}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ))}
    </div>
  );
}
