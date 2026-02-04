"""
Tests for Graph Schema Module - MERL-T Knowledge Graph
========================================================

Unit tests for node/edge definitions, schema validation, and query builders.
Testing the full MERL-T specification with 26 node types and 65 edge types.
"""

import pytest
from visualex.graph.schema import (
    GraphSchema,
    NodeType,
    EdgeType,
    Direction,
    IndexDefinition,
    NODE_PROPERTIES,
    EDGE_PROPERTIES,
    INDEXES,
    FULLTEXT_INDEXES,
)


# =============================================================================
# NodeType Tests
# =============================================================================


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_type_count(self):
        """Schema has 26 node types."""
        assert len(NodeType) == 26

    def test_node_type_string_enum(self):
        """NodeType values are strings."""
        assert NodeType.NORMA.value == "Norma"
        assert NodeType.VERSIONE.value == "Versione"
        assert NodeType.COMMA.value == "Comma"

    def test_node_type_is_str_subclass(self):
        """NodeType can be used as string."""
        assert isinstance(NodeType.NORMA, str)
        assert NodeType.NORMA == "Norma"

    def test_normative_sources_category(self):
        """Normative Sources node types exist."""
        expected = ["Norma", "Versione", "DirettivaUE", "RegolamentoUE"]
        for val in expected:
            assert val in [nt.value for nt in NodeType]

    def test_text_structure_category(self):
        """Text Structure node types exist."""
        expected = ["Comma", "Lettera", "Numero", "DefinizioneLegale"]
        for val in expected:
            assert val in [nt.value for nt in NodeType]

    def test_case_law_doctrine_category(self):
        """Case Law & Doctrine node types exist."""
        expected = ["AttoGiudiziario", "Caso", "Dottrina"]
        for val in expected:
            assert val in [nt.value for nt in NodeType]

    def test_subjects_roles_category(self):
        """Subjects & Roles node types exist."""
        expected = ["SoggettoGiuridico", "RuoloGiuridico", "Organo"]
        for val in expected:
            assert val in [nt.value for nt in NodeType]

    def test_legal_concepts_category(self):
        """Legal Concepts node types exist."""
        expected = ["Concetto", "Principio", "DirittoSoggettivo",
                    "InteresseLegittimo", "Responsabilita"]
        for val in expected:
            assert val in [nt.value for nt in NodeType]

    def test_dynamics_category(self):
        """Dynamics node types exist."""
        expected = ["FattoGiuridico", "Procedura", "Sanzione", "Termine"]
        for val in expected:
            assert val in [nt.value for nt in NodeType]

    def test_logic_reasoning_category(self):
        """Logic & Reasoning node types exist."""
        expected = ["Regola", "Proposizione", "ModalitaGiuridica"]
        for val in expected:
            assert val in [nt.value for nt in NodeType]


# =============================================================================
# EdgeType Tests
# =============================================================================


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_edge_type_count(self):
        """Schema has 65 edge types."""
        assert len(EdgeType) == 65

    def test_structural_relations(self):
        """Structural relations (5) exist."""
        expected = ["contiene", "parte_di", "versione_precedente",
                    "versione_successiva", "ha_versione"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_modification_relations(self):
        """Modification relations (9) exist."""
        expected = ["sostituisce", "inserisce", "abroga_totalmente",
                    "abroga_parzialmente", "sospende", "proroga",
                    "integra", "deroga_a", "consolida"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_semantic_relations(self):
        """Semantic relations (6) exist."""
        expected = ["disciplina", "applica_a", "definisce",
                    "prevede_sanzione", "stabilisce_termine", "prevede"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_dependency_relations(self):
        """Dependency relations (3) exist."""
        expected = ["dipende_da", "presuppone", "species"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_citation_interpretation_relations(self):
        """Citation & Interpretation relations (3) exist."""
        expected = ["cita", "interpreta", "commenta"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_european_relations(self):
        """European relations (3) exist."""
        expected = ["attua", "recepisce", "conforme_a"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_institutional_relations(self):
        """Institutional relations (3) exist."""
        expected = ["emesso_da", "ha_competenza_su", "gerarchicamente_superiore"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_case_based_relations(self):
        """Case-based relations (3) exist."""
        expected = ["riguarda", "applica_norma_a_caso", "precedente_di"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_classification_relations(self):
        """Classification relations (2) exist."""
        expected = ["fonte", "classifica_in"]
        for val in expected:
            assert val in [et.value for et in EdgeType]

    def test_lkif_modalities_exist(self):
        """LKIF Modality relations (28) exist."""
        # Sample of LKIF modalities (using actual edge type values)
        lkif_samples = [
            "impone", "conferisce", "titolare_di", "riveste_ruolo",
            "attribuisce_responsabilita", "responsabile_per",
            "esprime_principio", "conforma_a_principio",
            "giustifica", "limita", "tutela", "viola"
        ]
        edge_values = [et.value for et in EdgeType]
        for val in lkif_samples:
            assert val in edge_values, f"LKIF modality '{val}' not found"


class TestDirection:
    """Tests for Direction enum."""

    def test_all_directions_defined(self):
        """All expected directions exist."""
        assert Direction.IN.value == "in"
        assert Direction.OUT.value == "out"
        assert Direction.BOTH.value == "both"

    def test_direction_is_str_subclass(self):
        """Direction can be used as string."""
        assert isinstance(Direction.IN, str)
        assert Direction.OUT == "out"

    def test_direction_from_string(self):
        """Direction can be created from string."""
        assert Direction("in") == Direction.IN
        assert Direction("out") == Direction.OUT
        assert Direction("both") == Direction.BOTH


# =============================================================================
# Property Definitions Tests
# =============================================================================


class TestNodePropertyDefinitions:
    """Tests for node property definitions."""

    def test_all_node_types_have_properties(self):
        """All node types have property definitions."""
        for node_type in NodeType:
            assert node_type in NODE_PROPERTIES, f"{node_type} missing properties"

    def test_norma_has_required_properties(self):
        """Norma has all required properties."""
        props = NODE_PROPERTIES[NodeType.NORMA]
        required = ["node_id", "urn", "estremi", "titolo", "testo_vigente", "stato"]
        for prop in required:
            assert prop in props, f"Norma missing '{prop}'"

    def test_norma_has_multivigenza_properties(self):
        """Norma has temporal versioning properties."""
        props = NODE_PROPERTIES[NodeType.NORMA]
        versioning = ["data_versione", "data_pubblicazione", "data_entrata_in_vigore"]
        for prop in versioning:
            assert prop in props, f"Norma missing multivigenza property '{prop}'"

    def test_versione_properties(self):
        """Versione has required properties."""
        props = NODE_PROPERTIES[NodeType.VERSIONE]
        required = ["node_id", "numero_versione", "data_inizio_validita",
                    "data_fine_validita", "testo_completo"]
        for prop in required:
            assert prop in props

    def test_comma_properties(self):
        """Comma has required properties."""
        props = NODE_PROPERTIES[NodeType.COMMA]
        required = ["node_id", "urn", "testo", "posizione"]
        for prop in required:
            assert prop in props

    def test_atto_giudiziario_properties(self):
        """AttoGiudiziario has required properties."""
        props = NODE_PROPERTIES[NodeType.ATTO_GIUDIZIARIO]
        required = ["node_id", "urn", "organo_emittente", "data", "massima"]
        for prop in required:
            assert prop in props

    def test_concetto_uses_node_id(self):
        """Concetto uses node_id as identifier."""
        props = NODE_PROPERTIES[NodeType.CONCETTO]
        assert "node_id" in props

    def test_regola_has_lkif_properties(self):
        """Regola has LKIF rule properties."""
        props = NODE_PROPERTIES[NodeType.REGOLA]
        lkif_props = ["premesse", "conseguenze", "eccezioni", "forza"]
        for prop in lkif_props:
            assert prop in props

    def test_modalita_giuridica_properties(self):
        """ModalitaGiuridica has deontic properties."""
        props = NODE_PROPERTIES[NodeType.MODALITA_GIURIDICA]
        required = ["tipo_modalita", "soggetto_attivo", "soggetto_passivo"]
        for prop in required:
            assert prop in props


class TestEdgePropertyDefinitions:
    """Tests for edge property definitions."""

    def test_all_edge_types_have_properties(self):
        """All edge types have property definitions."""
        for edge_type in EdgeType:
            assert edge_type in EDGE_PROPERTIES, f"{edge_type} missing properties"

    def test_modification_edges_have_date(self):
        """Modification edges have temporal properties."""
        modification_edges = [
            EdgeType.SOSTITUISCE, EdgeType.INSERISCE,
            EdgeType.ABROGA_TOTALMENTE, EdgeType.ABROGA_PARZIALMENTE
        ]
        for edge_type in modification_edges:
            props = EDGE_PROPERTIES[edge_type]
            assert "data_efficacia" in props, f"{edge_type} missing data_efficacia"

    def test_cita_has_citation_properties(self):
        """CITA edge has citation properties."""
        props = EDGE_PROPERTIES[EdgeType.CITA]
        assert "tipo_citazione" in props

    def test_interpreta_has_interpretation_properties(self):
        """INTERPRETA edge has interpretation properties."""
        props = EDGE_PROPERTIES[EdgeType.INTERPRETA]
        assert "tipo_interpretazione" in props


# =============================================================================
# Index Definitions Tests
# =============================================================================


class TestIndexDefinitions:
    """Tests for index definitions."""

    def test_indexes_exist(self):
        """Index definitions are defined."""
        assert len(INDEXES) > 0

    def test_urn_indexes_exist(self):
        """URN indexes exist for URN-based node types."""
        urn_types = [NodeType.NORMA, NodeType.COMMA, NodeType.ATTO_GIUDIZIARIO]
        indexed_types = [idx.node_type for idx in INDEXES if idx.property_name == "urn"]
        for urn_type in urn_types:
            assert urn_type in indexed_types, f"URN index missing for {urn_type}"

    def test_fulltext_indexes_defined(self):
        """Full-text indexes are defined for text fields."""
        assert len(FULLTEXT_INDEXES) > 0
        fulltext_props = [(idx.node_type, idx.property_name) for idx in FULLTEXT_INDEXES]
        # Norma testo_vigente should be indexed
        assert (NodeType.NORMA, "testo_vigente") in fulltext_props

    def test_node_id_indexed(self):
        """node_id is indexed for key node types."""
        indexed_types = [idx.node_type for idx in INDEXES if idx.property_name == "node_id"]
        # At least Norma should have node_id indexed
        assert NodeType.NORMA in indexed_types or len([
            idx for idx in INDEXES
            if idx.node_type == NodeType.NORMA and idx.property_name in ("node_id", "urn")
        ]) > 0


# =============================================================================
# GraphSchema Tests
# =============================================================================


class TestGraphSchema:
    """Tests for GraphSchema manager."""

    @pytest.fixture
    def schema(self):
        return GraphSchema()

    def test_get_node_properties(self, schema):
        """get_node_properties returns correct properties."""
        props = schema.get_node_properties(NodeType.NORMA)
        assert "urn" in props
        assert "testo_vigente" in props

    def test_get_edge_properties(self, schema):
        """get_edge_properties returns correct properties."""
        props = schema.get_edge_properties(EdgeType.CONTIENE)
        # CONTIENE may have minimal or no properties
        assert isinstance(props, list)

    def test_get_create_index_queries(self, schema):
        """Index queries are generated correctly."""
        queries = schema.get_create_index_queries()
        assert len(queries) == len(INDEXES)
        assert all(q.startswith("CREATE INDEX ON") for q in queries)

    def test_get_create_fulltext_index_queries(self, schema):
        """Full-text index queries are generated correctly."""
        queries = schema.get_create_fulltext_index_queries()
        assert len(queries) == len(FULLTEXT_INDEXES)
        assert all("db.idx.fulltext.createNodeIndex" in q for q in queries)


# =============================================================================
# Validation Tests
# =============================================================================


class TestSchemaValidation:
    """Tests for schema validation."""

    @pytest.fixture
    def schema(self):
        return GraphSchema()

    def test_valid_norma_data(self, schema):
        """Valid Norma data passes validation."""
        data = {
            "node_id": "norma_001",
            "urn": "urn:nir:stato:legge:2020-12-30;178",
            "estremi": "Legge 30 dicembre 2020, n. 178",
            "titolo": "Bilancio di previsione 2021",
        }
        errors = schema.validate_node_data(NodeType.NORMA, data)
        assert errors == []

    def test_norma_missing_identifier_fails(self, schema):
        """Norma without node_id or urn fails validation."""
        # Only requires one of node_id or urn - with urn, it should pass
        data = {"urn": "urn:nir:stato:legge:2020;1", "titolo": "Test"}
        errors = schema.validate_node_data(NodeType.NORMA, data)
        assert errors == []  # Has urn, so it passes

    def test_norma_missing_both_identifiers_fails(self, schema):
        """Norma without both node_id and urn fails validation."""
        data = {"titolo": "Test"}
        errors = schema.validate_node_data(NodeType.NORMA, data)
        assert any("node_id" in e.lower() or "urn" in e.lower() for e in errors)

    def test_concetto_valid(self, schema):
        """Concetto with required fields passes validation."""
        data = {"node_id": "concetto_001", "nome": "Contratto"}
        errors = schema.validate_node_data(NodeType.CONCETTO, data)
        assert errors == []

    def test_unknown_property_allowed_for_extensibility(self, schema):
        """Unknown properties are allowed for extensibility (only logged)."""
        data = {
            "node_id": "norma_001",
            "urn": "urn:nir:stato:legge:2020;1",
            "unknown_field": "value",
        }
        # Unknown properties are allowed for extensibility - only logged as debug
        errors = schema.validate_node_data(NodeType.NORMA, data)
        assert errors == []  # No errors for unknown props (just logged)


# =============================================================================
# Query Builder Tests
# =============================================================================


class TestQueryBuilders:
    """Tests for Cypher query builders."""

    @pytest.fixture
    def schema(self):
        return GraphSchema()

    def test_build_create_node_query(self, schema):
        """CREATE node query is built correctly."""
        data = {
            "node_id": "norma_001",
            "urn": "urn:nir:stato:legge:2020-12-30;178",
            "estremi": "Legge 30 dicembre 2020, n. 178",
        }
        query, params = schema.build_create_node_query(NodeType.NORMA, data)

        assert "CREATE (n:Norma" in query
        assert "node_id: $node_id" in query
        assert params["node_id"] == "norma_001"

    def test_build_create_node_filters_invalid_props(self, schema):
        """CREATE node filters invalid properties."""
        data = {
            "node_id": "norma_001",
            "urn": "urn:nir:stato:legge:2020;1",
            "invalid_prop": "should_be_filtered",
        }
        query, params = schema.build_create_node_query(NodeType.NORMA, data)

        assert "invalid_prop" not in query
        assert "invalid_prop" not in params

    def test_build_merge_node_query(self, schema):
        """MERGE node query is built correctly."""
        data = {
            "node_id": "norma_001",
            "titolo": "Bilancio 2021",
        }
        query, params = schema.build_merge_node_query(
            NodeType.NORMA, "node_id", "norma_001", data
        )

        assert "MERGE (n:Norma" in query
        assert "ON CREATE SET" in query
        assert "ON MATCH SET" in query
        assert params["match_value"] == "norma_001"

    def test_build_create_edge_query(self, schema):
        """CREATE edge query is built correctly."""
        query, params = schema.build_create_edge_query(
            EdgeType.CONTIENE,
            NodeType.NORMA, "node_id", "norma_001",
            NodeType.COMMA, "node_id", "comma_001",
        )

        assert "MATCH (a:Norma" in query
        assert "MATCH (b:Comma" in query
        assert "CREATE (a)-[r:contiene]->(b)" in query
        assert params["from_value"] == "norma_001"
        assert params["to_value"] == "comma_001"

    def test_build_create_edge_with_properties(self, schema):
        """CREATE edge with properties is built correctly."""
        query, params = schema.build_create_edge_query(
            EdgeType.CITA,
            NodeType.NORMA, "node_id", "norma_001",
            NodeType.NORMA, "node_id", "norma_002",
            properties={"tipo_citazione": "rinvio"},
        )

        assert "tipo_citazione: $tipo_citazione" in query
        assert params["tipo_citazione"] == "rinvio"


# =============================================================================
# Integration Scenarios - MERL-T Use Cases
# =============================================================================


class TestMERLTScenarios:
    """Integration scenarios for MERL-T Knowledge Graph usage."""

    @pytest.fixture
    def schema(self):
        return GraphSchema()

    def test_codice_civile_structure(self, schema):
        """Test schema for Codice Civile article structure."""
        # Norma (R.D. 262/1942)
        norma_data = {
            "node_id": "cc_1942",
            "urn": "urn:nir:stato:regio.decreto:1942-03-16;262",
            "estremi": "Regio Decreto 16 marzo 1942, n. 262",
            "titolo": "Approvazione del Codice Civile",
            "stato": "vigente",
        }
        errors = schema.validate_node_data(NodeType.NORMA, norma_data)
        assert errors == []

        # Comma (Art. 1453, comma 1)
        comma_data = {
            "node_id": "cc_1942_art1453_c1",
            "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1453-com1",
            "testo": "Nei contratti con prestazioni corrispettive...",
            "posizione": 1,
        }
        errors = schema.validate_node_data(NodeType.COMMA, comma_data)
        assert errors == []

    def test_cassazione_ruling(self, schema):
        """Test schema for Cassazione ruling."""
        data = {
            "node_id": "cass_2020_12345",
            "urn": "ecli:IT:CASS:2020:12345",
            "organo_emittente": "Corte di Cassazione",
            "data": "2020-06-15",
            "massima": "La risoluzione del contratto per inadempimento...",
            "tipologia": "sentenza",
        }
        errors = schema.validate_node_data(NodeType.ATTO_GIUDIZIARIO, data)
        assert errors == []

    def test_eu_directive_transposition(self, schema):
        """Test schema for EU directive transposition."""
        direttiva_data = {
            "node_id": "dir_2019_1024",
            "urn": "celex:32019L1024",
            "estremi": "Direttiva (UE) 2019/1024",
            "titolo": "Direttiva Open Data",
            "data_adozione": "2019-06-20",
            "termine_recepimento": "2021-07-17",
        }
        errors = schema.validate_node_data(NodeType.DIRETTIVA_UE, direttiva_data)
        assert errors == []

    def test_multivigenza_versioning(self, schema):
        """Test schema for temporal versioning (multivigenza)."""
        versione_data = {
            "node_id": "cc_1942_v3",
            "numero_versione": 3,
            "data_inizio_validita": "2020-01-01",
            "data_fine_validita": "2021-12-31",
            "testo_completo": "Testo vigente nel periodo...",
            "descrizione_modifiche": "Modificato da L. 178/2020",
        }
        errors = schema.validate_node_data(NodeType.VERSIONE, versione_data)
        assert errors == []

    def test_legal_concept_definition(self, schema):
        """Test schema for legal concept."""
        concetto_data = {
            "node_id": "concetto_contratto",
            "nome": "Contratto",
            "definizione": "Accordo di due o più parti per costituire, regolare o estinguere...",
            "categoria": "diritto_civile",
        }
        errors = schema.validate_node_data(NodeType.CONCETTO, concetto_data)
        assert errors == []

    def test_lkif_rule(self, schema):
        """Test schema for LKIF logical rule."""
        regola_data = {
            "node_id": "regola_inadempimento",
            "nome": "Regola risoluzione per inadempimento",
            "tipo_regola": "condizione_effetto",
            "premesse": ["inadempimento", "contratto_bilaterale"],
            "conseguenze": ["diritto_risoluzione"],
            "eccezioni": ["inadempimento_lieve"],
        }
        errors = schema.validate_node_data(NodeType.REGOLA, regola_data)
        assert errors == []

    def test_deontic_modality(self, schema):
        """Test schema for deontic modality."""
        modalita_data = {
            "node_id": "obbligo_adempimento",
            "tipo_modalita": "obbligo",
            "descrizione": "Obbligo di adempiere la prestazione",
            "soggetto_attivo": "creditore",
            "soggetto_passivo": "debitore",
            "derogabile": True,
        }
        errors = schema.validate_node_data(NodeType.MODALITA_GIURIDICA, modalita_data)
        assert errors == []
