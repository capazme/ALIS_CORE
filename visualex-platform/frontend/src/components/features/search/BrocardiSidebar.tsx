/**
 * BrocardiSidebar - Sidebar with tabs: Brocardi, Ratio, Spiegazione, Massime, Relazioni.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { X, BookOpen, Lightbulb, FileText, Scale, GitBranch } from 'lucide-react';
import { cn } from '../../../lib/utils';
import type { BrocardiInfo } from '../../../types';
import type { BrocardiTab } from '../../../hooks/useBrocardiSidebar';

export interface BrocardiSidebarProps {
  isOpen: boolean;
  activeTab: BrocardiTab;
  onClose: () => void;
  onSetTab: (tab: BrocardiTab) => void;
  brocardi: BrocardiInfo | null | undefined;
}

const TABS: Array<{ id: BrocardiTab; label: string; icon: React.ReactNode }> = [
  { id: 'brocardi', label: 'Brocardi', icon: <BookOpen size={14} /> },
  { id: 'ratio', label: 'Ratio', icon: <Lightbulb size={14} /> },
  { id: 'spiegazione', label: 'Spiegazione', icon: <FileText size={14} /> },
  { id: 'massime', label: 'Massime', icon: <Scale size={14} /> },
  { id: 'relazioni', label: 'Relazioni', icon: <GitBranch size={14} /> },
];

export function BrocardiSidebar({ isOpen, activeTab, onClose, onSetTab, brocardi }: BrocardiSidebarProps) {
  const hasContent = (tab: BrocardiTab): boolean => {
    if (!brocardi) return false;
    switch (tab) {
      case 'brocardi': return !!(brocardi.Brocardi && brocardi.Brocardi.length > 0);
      case 'ratio': return !!brocardi.Ratio;
      case 'spiegazione': return !!brocardi.Spiegazione;
      case 'massime': return !!(brocardi.Massime && brocardi.Massime.length > 0);
      case 'relazioni': return !!(brocardi.Relazioni && brocardi.Relazioni.length > 0);
      default: return false;
    }
  };

  if (!brocardi) return null;

  const availableTabs = TABS.filter(t => hasContent(t.id));
  if (availableTabs.length === 0) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 320, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="flex-shrink-0 border-l border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden"
        >
          <div className="w-[320px] h-full flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-700">
              <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300">Approfondimenti</h3>
              <button
                onClick={onClose}
                aria-label="Chiudi approfondimenti"
                className="p-1 rounded text-slate-400 transition-colors hover:bg-slate-200 dark:hover:bg-slate-700"
              >
                <X size={16} />
              </button>
            </div>

            {/* Tab strip */}
            <div className="flex border-b border-slate-200 dark:border-slate-700 overflow-x-auto">
              {availableTabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => onSetTab(tab.id)}
                  className={cn(
                    "flex items-center gap-1 px-3 py-2 text-xs font-medium whitespace-nowrap transition-colors",
                    activeTab === tab.id
                      ? "text-primary-600 dark:text-primary-400 border-b-2 border-primary-500 bg-primary-50/50 dark:bg-primary-900/20"
                      : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700/50"
                  )}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.15 }}
                >
                  {activeTab === 'brocardi' && brocardi.Brocardi && (
                    <div className="space-y-3">
                      {brocardi.Brocardi.map((b, i) => (
                        <div key={i} className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 text-sm text-slate-700 dark:text-slate-300 italic">
                          {b}
                        </div>
                      ))}
                    </div>
                  )}

                  {activeTab === 'ratio' && brocardi.Ratio && (
                    <div className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-wrap">
                      {brocardi.Ratio}
                    </div>
                  )}

                  {activeTab === 'spiegazione' && brocardi.Spiegazione && (
                    <div className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-wrap">
                      {brocardi.Spiegazione}
                    </div>
                  )}

                  {activeTab === 'massime' && brocardi.Massime && (
                    <div className="space-y-3">
                      {brocardi.Massime.map((m, i) => (
                        <div key={i} className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 text-sm text-slate-700 dark:text-slate-300">
                          {typeof m === 'string' ? m : m.massima || JSON.stringify(m)}
                        </div>
                      ))}
                    </div>
                  )}

                  {activeTab === 'relazioni' && brocardi.Relazioni && (
                    <div className="space-y-2">
                      {brocardi.Relazioni.map((r, i) => (
                        <div key={i} className="p-2 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 text-sm">
                          <span className="font-medium text-primary-600 dark:text-primary-400">{r.titolo || r.tipo || 'Relazione'}</span>
                          {r.testo && (
                            <p className="text-xs text-slate-500 mt-1">{r.testo}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
