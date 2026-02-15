/**
 * EntityList Component
 *
 * Displays extracted entities and relations for an article.
 */

export interface Entity {
  id: string;
  name: string;
  type: string;
  confidence: number;
  position?: { start: number; end: number };
}

export interface Relation {
  id: string;
  sourceId: string;
  targetId: string;
  type: string;
  confidence: number;
}

interface EntityListProps {
  entities: Entity[] | undefined;
  relations: Relation[] | undefined;
}

export function EntityList({ entities, relations }: EntityListProps): React.ReactElement {
  if (!entities || entities.length === 0) {
    return (
      <div className="p-4 text-center">
        <p className="text-slate-500 dark:text-slate-400 text-sm">
          Nessuna entità estratta per questo articolo.
        </p>
        <p className="text-slate-400 dark:text-slate-500 text-xs mt-1">
          Avvia l'estrazione per popolare il Knowledge Graph.
        </p>
      </div>
    );
  }

  const groupedByType = entities.reduce(
    (acc, entity) => {
      if (!acc[entity.type]) {
        acc[entity.type] = [];
      }
      acc[entity.type].push(entity);
      return acc;
    },
    {} as Record<string, Entity[]>
  );

  return (
    <div className="p-2 space-y-3">
      {Object.entries(groupedByType).map(([type, typeEntities]) => (
        <div key={type}>
          <h3 className="px-2 py-1 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">{type}</h3>
          <ul className="space-y-1" role="list">
            {typeEntities.map((entity) => (
              <li
                key={entity.id}
                className="
                  flex items-center justify-between px-2 py-1.5 min-h-[44px]
                  bg-slate-50 dark:bg-slate-800/50 rounded hover:bg-slate-100 dark:hover:bg-slate-700/50
                  cursor-pointer transition-colors
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
                "
                tabIndex={0}
                role="listitem"
              >
                <span className="text-sm text-slate-900 dark:text-slate-100">{entity.name}</span>
                <span
                  className={`
                    text-xs px-1.5 py-0.5 rounded
                    ${entity.confidence > 0.8 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : ''}
                    ${entity.confidence > 0.6 && entity.confidence <= 0.8 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' : ''}
                    ${entity.confidence <= 0.6 ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : ''}
                  `}
                >
                  {Math.round(entity.confidence * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}

      {relations && relations.length > 0 && (
        <div>
          <h3 className="px-2 py-1 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Relazioni</h3>
          <p className="px-2 text-xs text-slate-500 dark:text-slate-400">{relations.length} relazioni trovate</p>
        </div>
      )}

      {(!relations || relations.length === 0) && (
        <div>
          <h3 className="px-2 py-1 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Relazioni</h3>
          <p className="px-2 text-xs text-slate-400 dark:text-slate-500">Nessuna relazione trovata.</p>
        </div>
      )}
    </div>
  );
}
