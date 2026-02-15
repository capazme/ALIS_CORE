/**
 * ContributionPanel Component
 *
 * Allows users to propose new entities or relations for an article.
 */

import { useState } from 'react';
import { useProposals } from '../hooks/useProposals';

interface ContributionPanelProps {
  articleUrn: string;
  userId?: string;
}

type ProposalType = 'entity' | 'relation';

export function ContributionPanel({ articleUrn, userId }: ContributionPanelProps): React.ReactElement {
  const [type, setType] = useState('entity' as ProposalType);
  const [name, setName] = useState('');
  const [entityType, setEntityType] = useState('CONCEPT');
  const [description, setDescription] = useState('');

  const { submitProposal, isSubmitting } = useProposals(userId);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!name.trim()) return;

    await submitProposal({
      type,
      articleUrn,
      name: name.trim(),
      entityType,
      description: description.trim() || undefined,
    });

    // Reset form
    setName('');
    setDescription('');
  };

  return (
    <div className="p-4">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Type selector */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setType('entity')}
            className={`
              flex-1 px-3 py-2 text-xs font-medium rounded-lg transition-colors
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
              ${type === 'entity' ? 'bg-purple-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'}
            `}
          >
            Entità
          </button>
          <button
            type="button"
            onClick={() => setType('relation')}
            className={`
              flex-1 px-3 py-2 text-xs font-medium rounded-lg transition-colors
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
              ${type === 'relation' ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'}
            `}
          >
            Relazione
          </button>
        </div>

        {type === 'entity' && (
          <>
            {/* Entity name */}
            <div>
              <label htmlFor="name" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                Nome entità
              </label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="es. Risoluzione contrattuale"
                className="
                  w-full px-3 py-2 text-sm
                  border border-slate-300 dark:border-slate-600 rounded-lg
                  bg-white dark:bg-slate-800 text-slate-900 dark:text-white
                  focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
                  outline-none transition-shadow
                "
              />
            </div>

            {/* Entity type */}
            <div>
              <label htmlFor="entityType" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                Tipo
              </label>
              <select
                id="entityType"
                value={entityType}
                onChange={(e) => setEntityType(e.target.value)}
                className="
                  w-full px-3 py-2 text-sm
                  border border-slate-300 dark:border-slate-600 rounded-lg
                  bg-white dark:bg-slate-800 text-slate-900 dark:text-white
                  focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
                  outline-none
                "
              >
                <option value="CONCEPT">Concetto</option>
                <option value="SUBJECT">Soggetto</option>
                <option value="CONDITION">Condizione</option>
                <option value="EFFECT">Effetto</option>
                <option value="PROCEDURE">Procedura</option>
              </select>
            </div>

            {/* Description */}
            <div>
              <label htmlFor="description" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                Descrizione (opzionale)
              </label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Breve descrizione dell'entità..."
                rows={3}
                className="
                  w-full px-3 py-2 text-sm
                  border border-slate-300 dark:border-slate-600 rounded-lg
                  bg-white dark:bg-slate-800 text-slate-900 dark:text-white
                  focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
                  outline-none transition-shadow resize-none
                "
              />
            </div>
          </>
        )}

        {type === 'relation' && (
          <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg text-sm text-slate-500 dark:text-slate-400">
            <p>Per proporre una relazione:</p>
            <ol className="list-decimal list-inside mt-2 space-y-1 text-xs">
              <li>Seleziona un&apos;entità sorgente nella lista</li>
              <li>Seleziona un&apos;entità target</li>
              <li>Scegli il tipo di relazione</li>
            </ol>
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          disabled={isSubmitting || !name.trim()}
          className="
            w-full px-4 py-2 text-sm font-medium
            bg-blue-600 text-white rounded-lg
            hover:bg-blue-700 transition-colors
            disabled:opacity-50 disabled:cursor-not-allowed
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2
          "
        >
          {isSubmitting ? 'Invio in corso...' : 'Proponi'}
        </button>
      </form>
    </div>
  );
}
