/**
 * Feedback Panel Component
 *
 * Container component with tabbed interface for all 3 feedback types:
 * - User Feedback (1-5 star rating + comments)
 * - RLCF Expert Feedback (multi-step expert validation)
 * - NER Corrections (entity recognition improvements)
 */

import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@components/ui/Tabs';
import { Badge } from '@components/ui/Badge';
import { useAuthStore } from '@/app/store/auth';
import { UserFeedbackForm } from './UserFeedbackForm';
import { RLCFExpertFeedbackForm } from './RLCFExpertFeedbackForm';
import { NERCorrectionForm } from './NERCorrectionForm';
import { MessageSquarePlus, Brain, Tag, Lock } from 'lucide-react';
// Note: User type from auth store, role check will use string comparison

interface FeedbackPanelProps {
  traceId: string;
  queryText: string;
}

export function FeedbackPanel({ traceId, queryText }: FeedbackPanelProps) {
  const { user } = useAuthStore();
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Check permissions
  const canSubmitUserFeedback = user !== null;
  // For now, all authenticated users can submit expert feedback (TODO: add role check when User type has role field)
  const canSubmitExpertFeedback = user !== null;
  const canSubmitNERCorrections = user !== null;

  const handleFeedbackSuccess = (type: string) => {
    setSuccessMessage(`Feedback ${type} inviato con successo!`);
    setTimeout(() => setSuccessMessage(null), 5000);
  };

  return (
    <div className="space-y-4">
      {/* Success message */}
      {successMessage && (
        <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg text-green-300 text-sm" role="alert">
          {successMessage}
        </div>
      )}

      <Tabs defaultValue="user" className="space-y-6">
        <TabsList className="bg-slate-900 p-1 rounded-lg inline-flex gap-1">
          {/* User Feedback Tab */}
          <TabsTrigger
            value="user"
            className="flex items-center gap-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 min-h-[44px]"
            disabled={!canSubmitUserFeedback}
          >
            <MessageSquarePlus className="w-4 h-4" aria-hidden="true" />
            Feedback Utente
            {!canSubmitUserFeedback && <Lock className="w-3 h-3 ml-1" aria-hidden="true" />}
          </TabsTrigger>

          {/* RLCF Expert Tab */}
          <TabsTrigger
            value="expert"
            className="flex items-center gap-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 min-h-[44px]"
            disabled={!canSubmitExpertFeedback}
          >
            <Brain className="w-4 h-4" aria-hidden="true" />
            RLCF Expert
            {canSubmitExpertFeedback && (
              <Badge variant="outline" className="ml-1 text-purple-300 border-purple-500/40 text-xs">
                Expert
              </Badge>
            )}
            {!canSubmitExpertFeedback && <Lock className="w-3 h-3 ml-1" aria-hidden="true" />}
          </TabsTrigger>

          {/* NER Corrections Tab */}
          <TabsTrigger
            value="ner"
            className="flex items-center gap-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 min-h-[44px]"
            disabled={!canSubmitNERCorrections}
          >
            <Tag className="w-4 h-4" aria-hidden="true" />
            NER Corrections
            {!canSubmitNERCorrections && <Lock className="w-3 h-3 ml-1" aria-hidden="true" />}
          </TabsTrigger>
        </TabsList>

        {/* User Feedback Content */}
        <TabsContent value="user">
          {canSubmitUserFeedback ? (
            <UserFeedbackForm
              traceId={traceId}
              onSuccess={() => handleFeedbackSuccess('Utente')}
            />
          ) : (
            <AuthenticationRequired type="user" />
          )}
        </TabsContent>

        {/* RLCF Expert Feedback Content */}
        <TabsContent value="expert">
          {canSubmitExpertFeedback ? (
            <RLCFExpertFeedbackForm
              traceId={traceId}
              onSuccess={() => handleFeedbackSuccess('Esperto RLCF')}
            />
          ) : (
            <AuthenticationRequired type="expert" />
          )}
        </TabsContent>

        {/* NER Corrections Content */}
        <TabsContent value="ner">
          {canSubmitNERCorrections ? (
            <NERCorrectionForm
              traceId={traceId}
              queryText={queryText}
              onSuccess={() => handleFeedbackSuccess('NER')}
            />
          ) : (
            <AuthenticationRequired type="ner" />
          )}
        </TabsContent>
      </Tabs>

      {/* Info Footer */}
      <div className="pt-6 border-t border-slate-700 text-xs text-slate-500">
        <h4 className="font-semibold text-slate-400 mb-2">Perché il Feedback è Importante?</h4>
        <ul className="space-y-1.5 ml-4">
          <li className="flex items-start gap-2">
            <span className="text-blue-400 mt-0.5">•</span>
            <span>
              <strong className="text-slate-400">Feedback Utente</strong> aiuta a valutare la
              qualità percepita e l'utilità delle risposte per utenti non-esperti
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-400 mt-0.5">•</span>
            <span>
              <strong className="text-slate-400">RLCF Expert</strong> migliora il sistema attraverso
              validazione di esperti legali, influenzando i punteggi di autorità e la qualità
              futura
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-orange-400 mt-0.5">•</span>
            <span>
              <strong className="text-slate-400">NER Corrections</strong> raffina il riconoscimento
              di entità legali (norme, sentenze, concetti) per query future più accurate
            </span>
          </li>
        </ul>
      </div>
    </div>
  );
}

interface AuthenticationRequiredProps {
  type: 'user' | 'expert' | 'ner';
}

function AuthenticationRequired({ type }: AuthenticationRequiredProps) {
  const getMessage = () => {
    switch (type) {
      case 'user':
        return {
          title: 'Autenticazione Richiesta',
          description:
            'Per inviare feedback utente devi essere autenticato. Effettua il login per continuare.',
        };
      case 'expert':
        return {
          title: 'Permessi Esperti Richiesti',
          description:
            'Il feedback RLCF Expert è riservato a utenti con ruolo Esperto, Valutatore o Amministratore. Contatta un amministratore per richiedere i permessi.',
        };
      case 'ner':
        return {
          title: 'Autenticazione Richiesta',
          description:
            'Per inviare correzioni NER devi essere autenticato. Effettua il login per continuare.',
        };
    }
  };

  const { title, description } = getMessage();

  return (
    <div className="p-6 md:p-12 text-center bg-slate-800/50 rounded-lg border border-slate-700">
      <Lock className="w-12 h-12 text-slate-500 mx-auto mb-4" aria-hidden="true" />
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-slate-400 max-w-md mx-auto">{description}</p>
    </div>
  );
}
