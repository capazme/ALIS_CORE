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
      <div className="p-4 text-center text-gray-500 text-sm">
        Nessuna entità estratta per questo articolo.
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
          <h3 className="px-2 py-1 text-xs font-semibold text-gray-500 uppercase">{type}</h3>
          <ul className="space-y-1">
            {typeEntities.map((entity) => (
              <li
                key={entity.id}
                className="
                  flex items-center justify-between px-2 py-1.5
                  bg-gray-50 rounded hover:bg-gray-100
                  cursor-pointer transition-colors
                "
              >
                <span className="text-sm text-gray-900">{entity.name}</span>
                <span
                  className={`
                    text-xs px-1.5 py-0.5 rounded
                    ${entity.confidence > 0.8 ? 'bg-green-100 text-green-700' : ''}
                    ${entity.confidence > 0.6 && entity.confidence <= 0.8 ? 'bg-yellow-100 text-yellow-700' : ''}
                    ${entity.confidence <= 0.6 ? 'bg-red-100 text-red-700' : ''}
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
          <h3 className="px-2 py-1 text-xs font-semibold text-gray-500 uppercase">Relazioni</h3>
          <p className="px-2 text-xs text-gray-500">{relations.length} relazioni trovate</p>
        </div>
      )}
    </div>
  );
}
