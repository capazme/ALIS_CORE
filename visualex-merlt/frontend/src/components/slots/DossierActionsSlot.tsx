/**
 * DossierActionsSlot - Plugin slot for dossier actions
 *
 * Provides the export training set button that was previously
 * part of DossierPage.
 */

import { useState } from 'react';
import { GraduationCap, Loader2 } from 'lucide-react';
import type { SlotProps } from '@visualex/platform/lib/plugins';
import { exportDossierTrainingSet, loadDossierTrainingSet } from '../../services/merltService';
import type { DossierArticleData } from '../../types/merlt';

export function DossierActionsSlot({ dossierId, userId, dossier }: SlotProps['dossier-actions']) {
  const [isExporting, setIsExporting] = useState(false);

  const handleExportTrainingSet = async () => {
    setIsExporting(true);
    try {
      const articles: DossierArticleData[] = dossier.items
        .filter(item => item.type === 'norma')
        .map(item => ({
          urn: String(item.data.urn || ''),
          tipo_atto: String(item.data.tipo_atto || 'unknown'),
          numero_atto: String(item.data.numero_atto || ''),
          numero_articolo: String(item.data.numero_articolo || ''),
          data: String(item.data.data || ''),
          user_status: (item.status || 'unread') as 'unread' | 'reading' | 'important' | 'done',
        }));

      if (articles.length === 0) {
        alert('Il dossier non contiene articoli normativi da esportare.');
        return;
      }

      const trainingSet = await exportDossierTrainingSet({
        dossier_title: dossier.title,
        dossier_description: dossier.description,
        dossier_tags: dossier.tags || [],
        articles,
        user_id: userId,
        include_qa_sessions: true,
      });

      const loadResult = await loadDossierTrainingSet({
        training_set: trainingSet,
        priority_boost: 0.3,
      });

      const completedCount = articles.filter(a => a.user_status === 'done').length;
      alert(
        `✅ Training Set Esportato!\n\n` +
        `📚 Articoli: ${trainingSet.articles_count}\n` +
        `✓ Completati: ${completedCount}\n` +
        `💬 Sessioni Q&A: ${trainingSet.qa_sessions_count}\n` +
        `📊 Esperienze aggiunte: ${loadResult.experiences_added}\n` +
        `${loadResult.training_ready ? '🎓 Pronto per training!' : '⏳ In attesa di più dati...'}`
      );
    } catch (error) {
      console.error('Error exporting training set:', error);
      alert('Errore durante l\'export del training set. Riprova più tardi.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <button
      type="button"
      onClick={handleExportTrainingSet}
      className="text-emerald-600 hover:bg-emerald-50 dark:hover:bg-emerald-900/20 p-2 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
      aria-label="Esporta Training Set RLCF"
      title="Esporta Training Set RLCF"
      disabled={isExporting}
    >
      {isExporting ? <Loader2 size={18} className="animate-spin" aria-hidden="true" /> : <GraduationCap size={18} aria-hidden="true" />}
    </button>
  );
}
