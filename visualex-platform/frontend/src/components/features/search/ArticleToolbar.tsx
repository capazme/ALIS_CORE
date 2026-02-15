/**
 * ArticleToolbar - Extracted toolbar from ArticleTabContent.
 * Provides quick actions: quick norms, notes, highlights, copy, more menu.
 */

import { useState, useEffect } from 'react';
import {
  Zap, StickyNote, Highlighter, Copy, MoreHorizontal,
  FolderPlus, Share2, Download, Clock, GitCompare,
  PanelRightOpen
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import { PluginSlot } from '../../../lib/plugins/PluginSlot';

export interface ArticleToolbarProps {
  // Data
  urn: string;
  articleId: string;
  url?: string;
  annotationsCount: number;
  highlightsCount: number;
  hasBrocardi: boolean;

  // Actions
  onAddToQuickNorms: () => void;
  onToggleNotes: () => void;
  onHighlightClick: () => void;
  onCopy: () => void;
  onOpenDossier: () => void;
  onShareLink: () => void;
  onExport: () => void;
  onVersionSearch: () => void;
  onCompare: () => void;
  onOpenStudyMode?: () => void;
  onToggleBrocardi: () => void;

  // State
  showNotes: boolean;
  showHighlightPicker: boolean;
}

export function ArticleToolbar({
  urn,
  articleId,
  url,
  annotationsCount,
  highlightsCount,
  hasBrocardi,
  onAddToQuickNorms,
  onToggleNotes,
  onHighlightClick,
  onCopy,
  onOpenDossier,
  onShareLink,
  onExport,
  onVersionSearch,
  onCompare,
  onOpenStudyMode,
  onToggleBrocardi,
  showNotes,
  showHighlightPicker,
}: ArticleToolbarProps) {
  const [showMoreMenu, setShowMoreMenu] = useState(false);

  // Close menu on Escape key
  useEffect(() => {
    if (!showMoreMenu) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowMoreMenu(false);
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [showMoreMenu]);

  return (
    <div className="flex items-center gap-1">
      {/* Primary buttons */}
      <button
        onClick={onAddToQuickNorms}
        className="p-1.5 rounded-md hover:bg-amber-50 dark:hover:bg-amber-900/20 text-slate-400 hover:text-amber-500 transition-colors"
        title="Aggiungi a norme rapide"
      >
        <Zap size={16} />
      </button>
      <button
        onClick={onToggleNotes}
        className={cn("p-1.5 rounded-md transition-colors relative",
          showNotes || annotationsCount > 0
            ? "bg-primary-50 text-primary-600 dark:bg-primary-900/20 dark:text-primary-400"
            : "hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-primary-500"
        )}
        title="Note Personali"
      >
        <StickyNote size={16} />
        {annotationsCount > 0 && (
          <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-primary-500 text-white text-[9px] rounded-full flex items-center justify-center">
            {annotationsCount}
          </span>
        )}
      </button>
      <button
        data-highlight-button
        onClick={onHighlightClick}
        className={cn(
          "p-1.5 rounded-md transition-colors relative",
          highlightsCount > 0
            ? "bg-purple-50 text-purple-600 dark:bg-purple-900/20 dark:text-purple-400"
            : "hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-purple-500"
        )}
        title="Evidenzia Testo"
      >
        <Highlighter size={16} />
        {highlightsCount > 0 && (
          <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-purple-500 text-white text-[9px] rounded-full flex items-center justify-center">
            {highlightsCount}
          </span>
        )}
      </button>

      {/* Brocardi sidebar toggle */}
      {hasBrocardi && (
        <button
          onClick={onToggleBrocardi}
          className="p-1.5 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-indigo-500 transition-colors"
          title="Approfondimenti Brocardi"
        >
          <PanelRightOpen size={16} />
        </button>
      )}

      {/* Plugin Slot */}
      <PluginSlot
        name="article-toolbar"
        props={{ urn, articleId }}
        className="flex items-center gap-1"
      />

      <button
        onClick={onCopy}
        className="p-1.5 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-emerald-500 transition-colors"
        title="Copia"
      >
        <Copy size={16} />
      </button>

      <div className="w-px h-4 mx-1 bg-slate-200 dark:bg-slate-700" role="separator" />

      {/* More menu */}
      <div className="relative">
        <button
          onClick={() => setShowMoreMenu(!showMoreMenu)}
          aria-haspopup="true"
          aria-expanded={showMoreMenu}
          className={cn(
            "p-1.5 rounded-md transition-colors",
            showMoreMenu
              ? "bg-slate-100 dark:bg-slate-800 text-primary-500"
              : "text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
          )}
          title="Altre azioni"
        >
          <MoreHorizontal size={16} />
        </button>
        {showMoreMenu && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setShowMoreMenu(false)} />
            <div role="menu" className="absolute right-0 mt-2 w-48 py-1 bg-white rounded-lg border border-slate-200 shadow-xl animate-in fade-in zoom-in-95 duration-200 z-50 dark:bg-slate-900 dark:border-slate-700">
              <button
                onClick={() => { onOpenDossier(); setShowMoreMenu(false); }}
                role="menuitem"
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-left text-slate-700 transition-colors hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                <FolderPlus size={14} className="text-slate-400" />
                Aggiungi a dossier
              </button>
              <button
                onClick={() => { onShareLink(); setShowMoreMenu(false); }}
                role="menuitem"
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-left text-slate-700 transition-colors hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                <Share2 size={14} className="text-slate-400" />
                Condividi link
              </button>
              <button
                onClick={() => { onExport(); setShowMoreMenu(false); }}
                role="menuitem"
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-left text-slate-700 transition-colors hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                <Download size={14} className="text-slate-400" />
                Esporta...
              </button>

              <div className="border-t border-slate-200 dark:border-slate-700 my-1" />

              <button
                onClick={() => { onVersionSearch(); setShowMoreMenu(false); }}
                role="menuitem"
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-left text-slate-700 transition-colors hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                <Clock size={14} className="text-slate-400" />
                Cerca versione...
              </button>
              <button
                onClick={() => { onCompare(); setShowMoreMenu(false); }}
                role="menuitem"
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-left text-slate-700 transition-colors hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                <GitCompare size={14} className="text-slate-400" />
                Confronta con...
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
