"""
Graph Schema - MERL-T Knowledge Graph
=====================================

Complete schema implementation following the MERL-T Knowledge Graph specification.

This module defines:
- 26 Node Types organized across 7 categories
- 65 Relation Types organized into 11 semantic categories
- Property definitions for each node and edge type
- Index creation for optimized queries (25 standard + 10 full-text)

Reference: Legacy/MERL-T_alpha/docs/archive/02-methodology/knowledge-graph.md

Standards Compliance:
- LKIF Core Ontology (all 15 modules)
- Akoma Ntoso
- ELI (European Legislation Identifier)
- EuroVoc Thesaurus
- FAIR Principles
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Node types
    "NodeType",
    "NODE_PROPERTIES",
    # Edge types
    "EdgeType",
    "EDGE_PROPERTIES",
    "COMMON_EDGE_PROPERTIES",
    # Direction
    "Direction",
    # Indexes
    "IndexDefinition",
    "INDEXES",
    "FULLTEXT_INDEXES",
    # Schema manager
    "GraphSchema",
]


# =============================================================================
# Node Type Definitions (23 Types)
# =============================================================================


class NodeType(str, Enum):
    """
    Node types in the MERL-T Knowledge Graph.

    Organized into categories:
    - Normative Sources: NORMA, VERSIONE, DIRETTIVA_UE, REGOLAMENTO_UE
    - Text Structure: COMMA, LETTERA, NUMERO, DEFINIZIONE_LEGALE
    - Case Law & Doctrine: ATTO_GIUDIZIARIO, CASO, DOTTRINA
    - Subjects & Roles: SOGGETTO_GIURIDICO, RUOLO_GIURIDICO, ORGANO
    - Legal Concepts: CONCETTO, PRINCIPIO, DIRITTO_SOGGETTIVO, INTERESSE_LEGITTIMO, RESPONSABILITA
    - Dynamics: FATTO_GIURIDICO, PROCEDURA, SANZIONE, TERMINE
    - Logic & Reasoning: REGOLA, PROPOSIZIONE, MODALITA_GIURIDICA
    """

    # === Normative Sources ===
    NORMA = "Norma"  # Legal norm (law, decree, article)
    VERSIONE = "Versione"  # Temporal version for multivigenza
    DIRETTIVA_UE = "DirettivaUE"  # EU Directive
    REGOLAMENTO_UE = "RegolamentoUE"  # EU Regulation

    # === Text Structure ===
    COMMA = "Comma"  # Paragraph/clause within article
    LETTERA = "Lettera"  # Lettered point within comma
    NUMERO = "Numero"  # Numbered sub-point within lettera
    DEFINIZIONE_LEGALE = "DefinizioneLegale"  # Legal definition in norm

    # === Case Law & Doctrine ===
    ATTO_GIUDIZIARIO = "AttoGiudiziario"  # Judicial act/decision
    CASO = "Caso"  # Case/fact pattern
    DOTTRINA = "Dottrina"  # Doctrinal commentary

    # === Subjects & Roles ===
    SOGGETTO_GIURIDICO = "SoggettoGiuridico"  # Legal subject (person, entity)
    RUOLO_GIURIDICO = "RuoloGiuridico"  # Legal role (imputato, creditore)
    ORGANO = "Organo"  # Jurisdictional/administrative body

    # === Legal Concepts ===
    CONCETTO = "Concetto"  # Legal concept (simulazione, buona fede)
    PRINCIPIO = "Principio"  # Legal principle (legalità, uguaglianza)
    DIRITTO_SOGGETTIVO = "DirittoSoggettivo"  # Subjective right
    INTERESSE_LEGITTIMO = "InteresseLegittimo"  # Legitimate interest
    RESPONSABILITA = "Responsabilita"  # Legal responsibility/liability

    # === Dynamics ===
    FATTO_GIURIDICO = "FattoGiuridico"  # Legal fact/event
    PROCEDURA = "Procedura"  # Legal procedure/process
    SANZIONE = "Sanzione"  # Sanction/penalty
    TERMINE = "Termine"  # Term/deadline

    # === Logic & Reasoning ===
    REGOLA = "Regola"  # Logical rule for reasoning
    PROPOSIZIONE = "Proposizione"  # Legal proposition
    MODALITA_GIURIDICA = "ModalitaGiuridica"  # Deontic modality (obbligo, permesso, divieto)


# =============================================================================
# Node Property Definitions
# =============================================================================

NODE_PROPERTIES: Dict[NodeType, List[str]] = {
    # --- Normative Sources ---
    NodeType.NORMA: [
        "node_id", "estremi", "urn", "fonte", "titolo", "descrizione",
        "testo_originale", "testo_vigente", "stato", "efficacia", "versione",
        "data_versione", "data_pubblicazione", "data_entrata_in_vigore",
        "data_abrogazione", "data_cessazione_efficacia", "ambito_territoriale",
        "ambito_di_applicazione", "materie", "classificazione_tematica",
        "revisioni_costituzionali", "note_redazionali", "doi", "licenza",
        # Additional for articles
        "numero_articolo", "rubrica", "allegato",
    ],
    NodeType.VERSIONE: [
        "node_id", "numero_versione", "data_inizio_validita", "data_fine_validita",
        "testo_completo", "descrizione_modifiche", "fonte_modifica", "consolidato",
    ],
    NodeType.DIRETTIVA_UE: [
        "node_id", "estremi", "urn", "tipo", "titolo", "descrizione",
        "data_adozione", "data_pubblicazione_gue", "data_entrata_in_vigore",
        "termine_recepimento", "base_giuridica", "classificazione_tematica",
    ],
    NodeType.REGOLAMENTO_UE: [
        "node_id", "estremi", "urn", "tipo", "titolo", "descrizione",
        "data_adozione", "data_pubblicazione_gue", "data_entrata_in_vigore",
        "base_giuridica", "classificazione_tematica",
    ],

    # --- Text Structure ---
    NodeType.COMMA: [
        "node_id", "urn", "tipo", "posizione", "testo", "testo_originale",
        "ordinamento", "data_versione",
    ],
    NodeType.LETTERA: [
        "node_id", "urn", "tipo", "posizione", "testo", "testo_originale",
        "ordinamento", "data_versione",
    ],
    NodeType.NUMERO: [
        "node_id", "urn", "tipo", "posizione", "testo", "testo_originale",
        "ordinamento", "data_versione",
    ],
    NodeType.DEFINIZIONE_LEGALE: [
        "node_id", "termine", "definizione", "ambito_applicazione",
        "sinonimi", "note",
    ],

    # --- Case Law & Doctrine ---
    NodeType.ATTO_GIUDIZIARIO: [
        "node_id", "estremi", "urn", "descrizione", "organo_emittente",
        "data", "tipologia", "materia", "massima",
    ],
    NodeType.CASO: [
        "node_id", "identificativo", "descrizione", "tipo_controversia",
        "esito", "rilevanza", "data", "parti",
    ],
    NodeType.DOTTRINA: [
        "node_id", "titolo", "autore", "descrizione",
        "data_pubblicazione", "fonte",
    ],

    # --- Subjects & Roles ---
    NodeType.SOGGETTO_GIURIDICO: [
        "node_id", "nome", "tipo", "ruolo", "qualifiche",
    ],
    NodeType.RUOLO_GIURIDICO: [
        "node_id", "nome", "tipo_ruolo", "descrizione",
        "poteri", "doveri", "requisiti", "incompatibilita", "temporaneo",
    ],
    NodeType.ORGANO: [
        "node_id", "nome", "tipo", "livello", "competenza_territoriale",
        "competenza_materia", "sede", "composizione",
    ],

    # --- Legal Concepts ---
    NodeType.CONCETTO: [
        "node_id", "nome", "definizione", "ambito_di_applicazione", "categoria",
    ],
    NodeType.PRINCIPIO: [
        "node_id", "nome", "tipo", "descrizione", "ambito_applicazione",
        "livello", "fonte", "derogabile", "bilanciabile",
    ],
    NodeType.DIRITTO_SOGGETTIVO: [
        "node_id", "nome", "tipo_diritto", "descrizione", "titolare",
        "opponibilita", "rinunciabilita", "trasmissibilita",
        "prescrittibilita", "tutela",
    ],
    NodeType.INTERESSE_LEGITTIMO: [
        "node_id", "tipo", "descrizione", "bene_della_vita",
        "titolare", "qualificazione", "strumenti_tutela",
    ],
    NodeType.RESPONSABILITA: [
        "node_id", "tipo_responsabilita", "descrizione", "fondamento",
        "elementi_costitutivi", "regime_probatorio", "prescrizione", "solidale",
    ],

    # --- Dynamics ---
    NodeType.FATTO_GIURIDICO: [
        "node_id", "tipo_fatto", "descrizione", "volontarieta", "liceita",
        "data_fatto", "luogo", "effetti_giuridici", "rilevanza",
    ],
    NodeType.PROCEDURA: [
        "node_id", "nome", "descrizione", "ambito", "tipologia",
    ],
    NodeType.SANZIONE: [
        "node_id", "tipo", "descrizione", "entita_minima", "entita_massima",
        "modalita_applicazione", "circostanze_aggravanti", "circostanze_attenuanti",
    ],
    NodeType.TERMINE: [
        "node_id", "descrizione", "durata", "tipo", "modalita_calcolo",
        "prorogabile", "conseguenze_mancato_rispetto",
    ],

    # --- Logic & Reasoning ---
    NodeType.REGOLA: [
        "node_id", "nome", "tipo_regola", "premesse", "conseguenze",
        "eccezioni", "forza", "ambito", "formalizzazione",
    ],
    NodeType.PROPOSIZIONE: [
        "node_id", "contenuto", "tipo", "modalita", "valore_verita",
        "contesto", "giustificazione",
    ],
    NodeType.MODALITA_GIURIDICA: [
        "node_id", "tipo_modalita", "descrizione", "soggetto_attivo",
        "soggetto_passivo", "condizioni", "contesto", "intensita", "derogabile",
    ],
}


# =============================================================================
# Edge Type Definitions (65 Types)
# =============================================================================


class EdgeType(str, Enum):
    """
    Edge types (relationships) in the MERL-T Knowledge Graph.

    Organized into 11 categories:
    1. Structural (5): containment, versioning, hierarchy
    2. Modification (9): legal text changes
    3. Semantic (6): meaning and governance
    4. Dependency (3): logical dependencies
    5. Citation & Interpretation (3): references
    6. European (3): EU law integration
    7. Institutional (3): bodies and jurisdiction
    8. Case-based (3): judicial application
    9. Classification (2): taxonomy
    10. LKIF Modalities (28): deontic logic and reasoning
    """

    # === 1. Structural Relations (5) ===
    CONTIENE = "contiene"  # Contains: Norma -> Comma, etc.
    PARTE_DI = "parte_di"  # Part of (inverse of contiene)
    VERSIONE_PRECEDENTE = "versione_precedente"  # Previous version
    VERSIONE_SUCCESSIVA = "versione_successiva"  # Next version
    HA_VERSIONE = "ha_versione"  # Norm has version

    # === 2. Modification Relations (9) ===
    SOSTITUISCE = "sostituisce"  # Replaces text
    INSERISCE = "inserisce"  # Inserts new text
    ABROGA_TOTALMENTE = "abroga_totalmente"  # Total repeal
    ABROGA_PARZIALMENTE = "abroga_parzialmente"  # Partial repeal
    SOSPENDE = "sospende"  # Suspends effectiveness
    PROROGA = "proroga"  # Extends deadline/validity
    INTEGRA = "integra"  # Supplements
    DEROGA_A = "deroga_a"  # Derogates (exception)
    CONSOLIDA = "consolida"  # Consolidates into unified text

    # === 3. Semantic Relations (6) ===
    DISCIPLINA = "disciplina"  # Governs concept
    APPLICA_A = "applica_a"  # Applies to subject
    DEFINISCE = "definisce"  # Defines term
    PREVEDE_SANZIONE = "prevede_sanzione"  # Prescribes sanction
    STABILISCE_TERMINE = "stabilisce_termine"  # Establishes deadline
    PREVEDE = "prevede"  # Provides for procedure

    # === 4. Dependency Relations (3) ===
    DIPENDE_DA = "dipende_da"  # Depends on
    PRESUPPONE = "presuppone"  # Presupposes
    SPECIES = "species"  # Specializes (is-a)

    # === 5. Citation & Interpretation Relations (3) ===
    CITA = "cita"  # Cites
    INTERPRETA = "interpreta"  # Interprets
    COMMENTA = "commenta"  # Comments on

    # === 6. European Relations (3) ===
    ATTUA = "attua"  # Implements EU directive
    RECEPISCE = "recepisce"  # Transposes EU directive
    CONFORME_A = "conforme_a"  # Complies with EU law

    # === 7. Institutional Relations (3) ===
    EMESSO_DA = "emesso_da"  # Issued by body
    HA_COMPETENZA_SU = "ha_competenza_su"  # Has jurisdiction over
    GERARCHICAMENTE_SUPERIORE = "gerarchicamente_superiore"  # Hierarchically superior

    # === 8. Case-based Relations (3) ===
    RIGUARDA = "riguarda"  # Concerns subject/case
    APPLICA_NORMA_A_CASO = "applica_norma_a_caso"  # Applies norm to case
    PRECEDENTE_DI = "precedente_di"  # Precedent for

    # === 9. Classification Relations (2) ===
    FONTE = "fonte"  # Source document
    CLASSIFICA_IN = "classifica_in"  # Classifies in category

    # === 10. LKIF Modality Relations (28) ===
    # Deontic and rights
    IMPONE = "impone"  # Imposes obligation/prohibition
    CONFERISCE = "conferisce"  # Confers right/power
    TITOLARE_DI = "titolare_di"  # Holder of right
    RIVESTE_RUOLO = "riveste_ruolo"  # Plays role

    # Responsibility
    ATTRIBUISCE_RESPONSABILITA = "attribuisce_responsabilita"
    RESPONSABILE_PER = "responsabile_per"

    # Principles
    ESPRIME_PRINCIPIO = "esprime_principio"  # Expresses principle
    CONFORMA_A_PRINCIPIO = "conforma_a_principio"  # Conforms to principle
    DEROGA_PRINCIPIO = "deroga_principio"  # Derogates principle
    BILANCIA_CON = "bilancia_con"  # Balances with

    # Legal effects
    PRODUCE_EFFETTO = "produce_effetto"  # Produces legal effect
    PRESUPPOSTO_DI = "presupposto_di"  # Prerequisite for
    COSTITUTIVO_DI = "costitutivo_di"  # Constitutive of
    ESTINGUE = "estingue"  # Extinguishes
    MODIFICA_EFFICACIA = "modifica_efficacia"  # Modifies efficacy

    # Reasoning
    APPLICA_REGOLA = "applica_regola"  # Applies rule
    IMPLICA = "implica"  # Implies
    CONTRADICE = "contradice"  # Contradicts
    GIUSTIFICA = "giustifica"  # Justifies

    # Rights and limits
    LIMITA = "limita"  # Limits
    TUTELA = "tutela"  # Protects
    VIOLA = "viola"  # Violates

    # Compatibility
    COMPATIBILE_CON = "compatibile_con"  # Compatible with
    INCOMPATIBILE_CON = "incompatibile_con"  # Incompatible with

    # Specification
    SPECIFICA = "specifica"  # Specifies
    ESEMPLIFICA = "esemplifica"  # Exemplifies

    # Causality
    CAUSA_DI = "causa_di"  # Cause of
    CONDIZIONE_DI = "condizione_di"  # Condition for


# =============================================================================
# Edge Property Definitions
# =============================================================================

# Common properties for all edges
COMMON_EDGE_PROPERTIES = [
    "data_decorrenza", "data_cessazione", "fonte_relazione",
    "certezza", "paragrafo_riferimento", "confidence_score",
    "validato_da", "data_validazione",
]

EDGE_PROPERTIES: Dict[EdgeType, List[str]] = {
    # Structural
    EdgeType.CONTIENE: [],
    EdgeType.PARTE_DI: [],
    EdgeType.VERSIONE_PRECEDENTE: [],
    EdgeType.VERSIONE_SUCCESSIVA: [],
    EdgeType.HA_VERSIONE: [],

    # Modification
    EdgeType.SOSTITUISCE: ["testo_modificato", "testo_nuovo", "data_efficacia"],
    EdgeType.INSERISCE: ["testo_inserito", "posizione_inserimento", "data_efficacia"],
    EdgeType.ABROGA_TOTALMENTE: ["data_efficacia", "effetto"],
    EdgeType.ABROGA_PARZIALMENTE: ["parte_abrogata", "data_efficacia"],
    EdgeType.SOSPENDE: ["data_inizio_sospensione", "data_fine_sospensione", "motivo"],
    EdgeType.PROROGA: ["nuova_scadenza", "durata_proroga"],
    EdgeType.INTEGRA: ["contenuto_integrativo", "data_efficacia"],
    EdgeType.DEROGA_A: ["ambito_deroga", "condizioni", "temporanea"],
    EdgeType.CONSOLIDA: ["tipo_consolidamento"],

    # Semantic
    EdgeType.DISCIPLINA: [],
    EdgeType.APPLICA_A: [],
    EdgeType.DEFINISCE: [],
    EdgeType.PREVEDE_SANZIONE: [],
    EdgeType.STABILISCE_TERMINE: [],
    EdgeType.PREVEDE: [],

    # Dependency
    EdgeType.DIPENDE_DA: ["tipo_dipendenza"],
    EdgeType.PRESUPPONE: ["tipo_presupposto"],
    EdgeType.SPECIES: [],

    # Citation
    EdgeType.CITA: ["tipo_citazione"],
    EdgeType.INTERPRETA: ["tipo_interpretazione", "orientamento"],
    EdgeType.COMMENTA: [],

    # European
    EdgeType.ATTUA: ["data_recepimento", "conforme", "note_conformita"],
    EdgeType.RECEPISCE: ["integrale", "parziale", "adeguamento_necessario"],
    EdgeType.CONFORME_A: [],

    # Institutional
    EdgeType.EMESSO_DA: [],
    EdgeType.HA_COMPETENZA_SU: [],
    EdgeType.GERARCHICAMENTE_SUPERIORE: [],

    # Case-based
    EdgeType.RIGUARDA: [],
    EdgeType.APPLICA_NORMA_A_CASO: [],
    EdgeType.PRECEDENTE_DI: ["forza_vincolante"],

    # Classification
    EdgeType.FONTE: [],
    EdgeType.CLASSIFICA_IN: ["schema_classificazione"],

    # LKIF Modalities
    EdgeType.IMPONE: ["condizionale", "condizioni"],
    EdgeType.CONFERISCE: ["beneficiario"],
    EdgeType.TITOLARE_DI: [],
    EdgeType.RIVESTE_RUOLO: ["contesto", "temporaneo"],
    EdgeType.ATTRIBUISCE_RESPONSABILITA: ["soggetto_responsabile"],
    EdgeType.RESPONSABILE_PER: ["fondamento", "grado"],
    EdgeType.ESPRIME_PRINCIPIO: [],
    EdgeType.CONFORMA_A_PRINCIPIO: [],
    EdgeType.DEROGA_PRINCIPIO: ["giustificazione", "ambito_deroga"],
    EdgeType.BILANCIA_CON: ["contesto", "prevalenza"],
    EdgeType.PRODUCE_EFFETTO: ["automatico", "condizioni"],
    EdgeType.PRESUPPOSTO_DI: [],
    EdgeType.COSTITUTIVO_DI: [],
    EdgeType.ESTINGUE: ["modo_estinzione"],
    EdgeType.MODIFICA_EFFICACIA: ["tipo_modifica"],
    EdgeType.APPLICA_REGOLA: ["esplicita"],
    EdgeType.IMPLICA: ["tipo_implicazione"],
    EdgeType.CONTRADICE: ["tipo_contraddizione"],
    EdgeType.GIUSTIFICA: ["tipo_giustificazione"],
    EdgeType.LIMITA: ["tipo_limite", "proporzionale"],
    EdgeType.TUTELA: ["tipo_tutela"],
    EdgeType.VIOLA: ["gravita", "dolosa"],
    EdgeType.COMPATIBILE_CON: [],
    EdgeType.INCOMPATIBILE_CON: ["criterio_risoluzione"],
    EdgeType.SPECIFICA: [],
    EdgeType.ESEMPLIFICA: [],
    EdgeType.CAUSA_DI: ["tipo_causalita"],
    EdgeType.CONDIZIONE_DI: ["tipo_condizione"],
}


# =============================================================================
# Direction Enum for Traversal
# =============================================================================


class Direction(str, Enum):
    """Direction for graph traversal."""

    IN = "in"  # Incoming edges only
    OUT = "out"  # Outgoing edges only
    BOTH = "both"  # Both directions


# =============================================================================
# Index Definitions
# =============================================================================


@dataclass
class IndexDefinition:
    """Definition for a graph index."""

    node_type: NodeType
    property_name: str
    is_unique: bool = False
    is_fulltext: bool = False


# Standard indexes for optimized queries
INDEXES: List[IndexDefinition] = [
    # URN/ID indexes (primary lookup)
    IndexDefinition(NodeType.NORMA, "urn", is_unique=True),
    IndexDefinition(NodeType.NORMA, "node_id", is_unique=True),
    IndexDefinition(NodeType.VERSIONE, "node_id", is_unique=True),
    IndexDefinition(NodeType.COMMA, "urn", is_unique=True),
    IndexDefinition(NodeType.LETTERA, "urn", is_unique=True),
    IndexDefinition(NodeType.NUMERO, "urn", is_unique=True),
    IndexDefinition(NodeType.ATTO_GIUDIZIARIO, "urn", is_unique=True),
    IndexDefinition(NodeType.DIRETTIVA_UE, "urn", is_unique=True),
    IndexDefinition(NodeType.REGOLAMENTO_UE, "urn", is_unique=True),

    # Filtering indexes
    IndexDefinition(NodeType.NORMA, "stato"),
    IndexDefinition(NodeType.NORMA, "data_pubblicazione"),
    IndexDefinition(NodeType.NORMA, "data_entrata_in_vigore"),
    IndexDefinition(NodeType.NORMA, "numero_articolo"),
    IndexDefinition(NodeType.ATTO_GIUDIZIARIO, "organo_emittente"),
    IndexDefinition(NodeType.ATTO_GIUDIZIARIO, "data"),
    IndexDefinition(NodeType.ORGANO, "tipo"),
    IndexDefinition(NodeType.ORGANO, "livello"),

    # Concept/Entity indexes
    IndexDefinition(NodeType.CONCETTO, "node_id", is_unique=True),
    IndexDefinition(NodeType.CONCETTO, "nome"),
    IndexDefinition(NodeType.PRINCIPIO, "node_id", is_unique=True),
    IndexDefinition(NodeType.PRINCIPIO, "nome"),
    IndexDefinition(NodeType.SOGGETTO_GIURIDICO, "node_id", is_unique=True),
    IndexDefinition(NodeType.RUOLO_GIURIDICO, "node_id", is_unique=True),
    IndexDefinition(NodeType.SANZIONE, "node_id", is_unique=True),
    IndexDefinition(NodeType.TERMINE, "node_id", is_unique=True),
    IndexDefinition(NodeType.DEFINIZIONE_LEGALE, "node_id", is_unique=True),
    IndexDefinition(NodeType.DEFINIZIONE_LEGALE, "termine"),
]

# Full-text indexes for search
FULLTEXT_INDEXES: List[IndexDefinition] = [
    IndexDefinition(NodeType.NORMA, "testo_vigente", is_fulltext=True),
    IndexDefinition(NodeType.NORMA, "titolo", is_fulltext=True),
    IndexDefinition(NodeType.COMMA, "testo", is_fulltext=True),
    IndexDefinition(NodeType.LETTERA, "testo", is_fulltext=True),
    IndexDefinition(NodeType.ATTO_GIUDIZIARIO, "massima", is_fulltext=True),
    IndexDefinition(NodeType.ATTO_GIUDIZIARIO, "descrizione", is_fulltext=True),
    IndexDefinition(NodeType.CONCETTO, "definizione", is_fulltext=True),
    IndexDefinition(NodeType.PRINCIPIO, "descrizione", is_fulltext=True),
    IndexDefinition(NodeType.DOTTRINA, "descrizione", is_fulltext=True),
    IndexDefinition(NodeType.DEFINIZIONE_LEGALE, "definizione", is_fulltext=True),
]


# =============================================================================
# Schema Manager
# =============================================================================


@dataclass
class GraphSchema:
    """
    Graph schema manager for MERL-T Knowledge Graph.

    Handles schema validation, index creation, and query building.
    """

    _initialized: bool = field(default=False, repr=False)
    _indexes_created: List[str] = field(default_factory=list, repr=False)

    def get_create_index_queries(self) -> List[str]:
        """
        Generate Cypher queries to create all indexes.

        Returns:
            List of CREATE INDEX Cypher statements
        """
        queries = []
        for idx in INDEXES:
            query = f"CREATE INDEX ON :{idx.node_type.value}({idx.property_name})"
            queries.append(query)
        return queries

    def get_create_fulltext_index_queries(self) -> List[str]:
        """
        Generate queries to create full-text indexes.

        Returns:
            List of CALL statements for full-text index creation
        """
        queries = []
        for idx in FULLTEXT_INDEXES:
            query = (
                f"CALL db.idx.fulltext.createNodeIndex("
                f"'{idx.node_type.value}', '{idx.property_name}')"
            )
            queries.append(query)
        return queries

    def get_node_properties(self, node_type: NodeType) -> List[str]:
        """
        Get property names for a node type.

        Args:
            node_type: The node type enum value

        Returns:
            List of property names
        """
        return NODE_PROPERTIES.get(node_type, [])

    def get_edge_properties(self, edge_type: EdgeType) -> List[str]:
        """
        Get property names for an edge type.

        Args:
            edge_type: The edge type enum value

        Returns:
            List of property names (includes common properties)
        """
        specific = EDGE_PROPERTIES.get(edge_type, [])
        return specific + COMMON_EDGE_PROPERTIES

    def get_all_node_types(self) -> List[NodeType]:
        """Get all node types."""
        return list(NodeType)

    def get_all_edge_types(self) -> List[EdgeType]:
        """Get all edge types."""
        return list(EdgeType)

    def get_node_types_by_category(self) -> Dict[str, List[NodeType]]:
        """
        Get node types organized by category.

        Returns:
            Dict mapping category name to list of node types
        """
        return {
            "Normative Sources": [
                NodeType.NORMA, NodeType.VERSIONE,
                NodeType.DIRETTIVA_UE, NodeType.REGOLAMENTO_UE,
            ],
            "Text Structure": [
                NodeType.COMMA, NodeType.LETTERA, NodeType.NUMERO,
                NodeType.DEFINIZIONE_LEGALE,
            ],
            "Case Law & Doctrine": [
                NodeType.ATTO_GIUDIZIARIO, NodeType.CASO, NodeType.DOTTRINA,
            ],
            "Subjects & Roles": [
                NodeType.SOGGETTO_GIURIDICO, NodeType.RUOLO_GIURIDICO,
                NodeType.ORGANO,
            ],
            "Legal Concepts": [
                NodeType.CONCETTO, NodeType.PRINCIPIO, NodeType.DIRITTO_SOGGETTIVO,
                NodeType.INTERESSE_LEGITTIMO, NodeType.RESPONSABILITA,
            ],
            "Dynamics": [
                NodeType.FATTO_GIURIDICO, NodeType.PROCEDURA,
                NodeType.SANZIONE, NodeType.TERMINE,
            ],
            "Logic & Reasoning": [
                NodeType.REGOLA, NodeType.PROPOSIZIONE, NodeType.MODALITA_GIURIDICA,
            ],
        }

    def get_edge_types_by_category(self) -> Dict[str, List[EdgeType]]:
        """
        Get edge types organized by category.

        Returns:
            Dict mapping category name to list of edge types
        """
        return {
            "Structural": [
                EdgeType.CONTIENE, EdgeType.PARTE_DI, EdgeType.VERSIONE_PRECEDENTE,
                EdgeType.VERSIONE_SUCCESSIVA, EdgeType.HA_VERSIONE,
            ],
            "Modification": [
                EdgeType.SOSTITUISCE, EdgeType.INSERISCE, EdgeType.ABROGA_TOTALMENTE,
                EdgeType.ABROGA_PARZIALMENTE, EdgeType.SOSPENDE, EdgeType.PROROGA,
                EdgeType.INTEGRA, EdgeType.DEROGA_A, EdgeType.CONSOLIDA,
            ],
            "Semantic": [
                EdgeType.DISCIPLINA, EdgeType.APPLICA_A, EdgeType.DEFINISCE,
                EdgeType.PREVEDE_SANZIONE, EdgeType.STABILISCE_TERMINE, EdgeType.PREVEDE,
            ],
            "Dependency": [
                EdgeType.DIPENDE_DA, EdgeType.PRESUPPONE, EdgeType.SPECIES,
            ],
            "Citation & Interpretation": [
                EdgeType.CITA, EdgeType.INTERPRETA, EdgeType.COMMENTA,
            ],
            "European": [
                EdgeType.ATTUA, EdgeType.RECEPISCE, EdgeType.CONFORME_A,
            ],
            "Institutional": [
                EdgeType.EMESSO_DA, EdgeType.HA_COMPETENZA_SU,
                EdgeType.GERARCHICAMENTE_SUPERIORE,
            ],
            "Case-based": [
                EdgeType.RIGUARDA, EdgeType.APPLICA_NORMA_A_CASO, EdgeType.PRECEDENTE_DI,
            ],
            "Classification": [
                EdgeType.FONTE, EdgeType.CLASSIFICA_IN,
            ],
            "LKIF Modalities": [
                EdgeType.IMPONE, EdgeType.CONFERISCE, EdgeType.TITOLARE_DI,
                EdgeType.RIVESTE_RUOLO, EdgeType.ATTRIBUISCE_RESPONSABILITA,
                EdgeType.RESPONSABILE_PER, EdgeType.ESPRIME_PRINCIPIO,
                EdgeType.CONFORMA_A_PRINCIPIO, EdgeType.DEROGA_PRINCIPIO,
                EdgeType.BILANCIA_CON, EdgeType.PRODUCE_EFFETTO,
                EdgeType.PRESUPPOSTO_DI, EdgeType.COSTITUTIVO_DI,
                EdgeType.ESTINGUE, EdgeType.MODIFICA_EFFICACIA,
                EdgeType.APPLICA_REGOLA, EdgeType.IMPLICA, EdgeType.CONTRADICE,
                EdgeType.GIUSTIFICA, EdgeType.LIMITA, EdgeType.TUTELA,
                EdgeType.VIOLA, EdgeType.COMPATIBILE_CON, EdgeType.INCOMPATIBILE_CON,
                EdgeType.SPECIFICA, EdgeType.ESEMPLIFICA, EdgeType.CAUSA_DI,
                EdgeType.CONDIZIONE_DI,
            ],
        }

    def validate_node_data(
        self, node_type: NodeType, data: Dict[str, Any]
    ) -> List[str]:
        """
        Validate node data against schema.

        Args:
            node_type: The node type
            data: Node property data

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        valid_props = self.get_node_properties(node_type)

        # Check for required node_id or urn
        has_id = "node_id" in data and data["node_id"]
        has_urn = "urn" in data and data["urn"]

        if not has_id and not has_urn:
            errors.append(f"{node_type.value} requires 'node_id' or 'urn' property")

        # Warn about unknown properties (not error, for flexibility)
        for prop in data:
            if prop not in valid_props:
                logger.debug(
                    "Property '%s' not in schema for %s (allowed for extensibility)",
                    prop, node_type.value
                )

        return errors

    def build_create_node_query(
        self,
        node_type: NodeType,
        data: Dict[str, Any],
        variable_name: str = "n",
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a CREATE node Cypher query.

        Args:
            node_type: The node type
            data: Node properties
            variable_name: Cypher variable name for the node

        Returns:
            Tuple of (query string, parameters dict)
        """
        # Filter to valid properties only
        valid_props = self.get_node_properties(node_type)
        filtered_data = {k: v for k, v in data.items() if k in valid_props}

        # Build property assignments
        prop_assignments = ", ".join(
            f"{k}: ${k}" for k in filtered_data.keys()
        )

        query = f"CREATE ({variable_name}:{node_type.value} {{{prop_assignments}}})"
        return query, filtered_data

    def build_merge_node_query(
        self,
        node_type: NodeType,
        match_key: str,
        match_value: Any,
        data: Dict[str, Any],
        variable_name: str = "n",
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a MERGE node Cypher query (upsert).

        Args:
            node_type: The node type
            match_key: Property to match on (usually 'urn' or 'node_id')
            match_value: Value to match
            data: Node properties to set
            variable_name: Cypher variable name

        Returns:
            Tuple of (query string, parameters dict)
        """
        valid_props = self.get_node_properties(node_type)
        filtered_data = {k: v for k, v in data.items() if k in valid_props}

        # Build SET assignments
        set_assignments = ", ".join(
            f"{variable_name}.{k} = ${k}" for k in filtered_data.keys()
        )

        params = {**filtered_data, "match_value": match_value}

        query = (
            f"MERGE ({variable_name}:{node_type.value} {{{match_key}: $match_value}}) "
            f"ON CREATE SET {set_assignments} "
            f"ON MATCH SET {set_assignments}"
        )

        return query, params

    def build_create_edge_query(
        self,
        edge_type: EdgeType,
        from_label: NodeType,
        from_key: str,
        from_value: Any,
        to_label: NodeType,
        to_key: str,
        to_value: Any,
        properties: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a CREATE relationship Cypher query.

        Args:
            edge_type: The edge type
            from_label: Source node type
            from_key: Source match property
            from_value: Source match value
            to_label: Target node type
            to_key: Target match property
            to_value: Target match value
            properties: Optional edge properties

        Returns:
            Tuple of (query string, parameters dict)
        """
        valid_props = self.get_edge_properties(edge_type)
        filtered_props = {}
        if properties:
            filtered_props = {k: v for k, v in properties.items() if k in valid_props}

        params = {
            "from_value": from_value,
            "to_value": to_value,
            **filtered_props,
        }

        prop_str = ""
        if filtered_props:
            prop_assignments = ", ".join(f"{k}: ${k}" for k in filtered_props.keys())
            prop_str = f" {{{prop_assignments}}}"

        query = (
            f"MATCH (a:{from_label.value} {{{from_key}: $from_value}}) "
            f"MATCH (b:{to_label.value} {{{to_key}: $to_value}}) "
            f"CREATE (a)-[r:{edge_type.value}{prop_str}]->(b) "
            f"RETURN r"
        )

        return query, params

    def get_schema_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the schema.

        Returns:
            Dict with counts and categories
        """
        return {
            "node_types": len(NodeType),
            "edge_types": len(EdgeType),
            "indexes": len(INDEXES),
            "fulltext_indexes": len(FULLTEXT_INDEXES),
            "node_categories": list(self.get_node_types_by_category().keys()),
            "edge_categories": list(self.get_edge_types_by_category().keys()),
        }
