/**
 * MerltSidebarPanel
 *
 * Rendered in the article-sidebar slot when MERLT plugin is active.
 * Shows extracted entities, validation UI, and contribution options.
 */

import { useState, useEffect } from 'react';
import type { SlotProps } from '@visualex/platform/lib/plugins';
import { EventBus } from '@visualex/platform/lib/plugins';
import { useMerltArticleAnalysis } from '../hooks/useMerltArticleAnalysis';
import { EntityList } from './EntityList';
import { ValidationQueue } from './ValidationQueue';
import { ContributionPanel } from './ContributionPanel';

type Props = SlotProps['article-sidebar'];

type Tab = 'entities' | 'validate' | 'contribute';

export function MerltSidebarPanel({ urn, articleId }: Props): React.ReactElement {
  const [activeTab, setActiveTab] = useState<Tab>('entities');
  const { entities, relations, isLoading, error } = useMerltArticleAnalysis(urn);

  // Subscribe to article:viewed event to update context
  useEffect(() => {
    const handleArticleViewed = (data: { urn: string; articleId: string; userId?: string }) => {
      if (data.urn === urn) {
        console.log('[MERLT] Article viewed:', data);
        // Could trigger refresh or update internal state here
      }
    };

    const unsubscribe = EventBus.on('article:viewed', handleArticleViewed);
    return unsubscribe;
  }, [urn]);

  // Subscribe to article:text-selected for entity proposal flows
  useEffect(() => {
    const handleTextSelected = (data: { urn: string; text: string; startOffset: number; endOffset: number }) => {
      if (data.urn === urn) {
        console.log('[MERLT] Text selected:', data.text);
        // Could highlight related entities in sidebar or show suggestion UI
      }
    };

    const unsubscribe = EventBus.on('article:text-selected', handleTextSelected);
    return unsubscribe;
  }, [urn]);

  if (error) {
    return (
      <div className="p-4 text-red-600 bg-red-50 rounded-lg">
        <p className="font-medium">Errore MERLT</p>
        <p className="text-sm">{error.message}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white border-l border-gray-200">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-900">MERLT Research</h2>
        <p className="text-xs text-gray-500">Contribuisci alla ricerca</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        <TabButton active={activeTab === 'entities'} onClick={() => setActiveTab('entities')}>
          Entità ({entities?.length ?? 0})
        </TabButton>
        <TabButton active={activeTab === 'validate'} onClick={() => setActiveTab('validate')}>
          Valida
        </TabButton>
        <TabButton active={activeTab === 'contribute'} onClick={() => setActiveTab('contribute')}>
          Proponi
        </TabButton>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <>
            {activeTab === 'entities' && <EntityList entities={entities} relations={relations} />}
            {activeTab === 'validate' && <ValidationQueue articleUrn={urn} />}
            {activeTab === 'contribute' && <ContributionPanel articleUrn={urn} />}
          </>
        )}
      </div>
    </div>
  );
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

function TabButton({ active, onClick, children }: TabButtonProps): React.ReactElement {
  return (
    <button
      onClick={onClick}
      className={`
        flex-1 px-3 py-2 text-xs font-medium transition-colors
        ${
          active
            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
            : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
        }
      `}
    >
      {children}
    </button>
  );
}
