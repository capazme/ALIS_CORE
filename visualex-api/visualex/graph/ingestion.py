"""
Graph Ingestion Pipeline - Norm Node Ingestion
===============================================

Ingest scraped legal norms into the FalkorDB Knowledge Graph.

This module processes NormaVisitata data from Epic 2a scrapers and creates
hierarchical graph nodes following the MERL-T Knowledge Graph specification.

Node Hierarchy:
    Norma (Article)
        -> Comma (Paragraph)
            -> Lettera (a), b), c))
                -> Numero (1), 2), 3))

Reference: Story 2b-2: Norm Node Ingestion
"""

import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Any, Optional, Tuple

from visualex.graph.schema import NodeType, EdgeType

if TYPE_CHECKING:
    from visualex.graph.client import FalkorDBClient
    from visualex.models.norma import NormaVisitata
    from visualex.graph.relations import RelationCreator

__all__ = [
    "NormIngester",
    "ArticleParser",
    "ArticleStructure",
    "CommaStructure",
    "LetteraStructure",
    "NumeroStructure",
    "IngestionResult",
    "BatchResult",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Data Structures for Parsed Article
# =============================================================================


@dataclass
class NumeroStructure:
    """Parsed numero (numbered sub-sub-point) within a lettera."""

    posizione: int  # 1, 2, 3...
    testo: str
    urn: Optional[str] = None


@dataclass
class LetteraStructure:
    """Parsed lettera (lettered sub-point a), b), c)) within a comma."""

    posizione: str  # "a", "b", "c"...
    testo: str
    numeri: List[NumeroStructure] = field(default_factory=list)
    urn: Optional[str] = None


@dataclass
class CommaStructure:
    """Parsed comma (paragraph) within an article."""

    posizione: int  # 1, 2, 3...
    testo: str
    lettere: List[LetteraStructure] = field(default_factory=list)
    urn: Optional[str] = None


@dataclass
class ArticleStructure:
    """Complete parsed structure of an article."""

    urn: str
    numero_articolo: str
    rubrica: Optional[str] = None
    testo_completo: str = ""
    commi: List[CommaStructure] = field(default_factory=list)
    data_versione: Optional[str] = None

    def node_count(self) -> int:
        """Count total nodes that will be created."""
        count = 1  # The article itself
        for comma in self.commi:
            count += 1  # The comma
            for lettera in comma.lettere:
                count += 1  # The lettera
                count += len(lettera.numeri)  # The numeri
        return count


@dataclass
class IngestionResult:
    """Result of ingesting a single article."""

    urn: str
    success: bool
    nodes_created: int = 0
    nodes_updated: int = 0
    edges_created: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "urn": self.urn,
            "success": self.success,
            "nodes_created": self.nodes_created,
            "nodes_updated": self.nodes_updated,
            "edges_created": self.edges_created,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class BatchResult:
    """Result of batch ingestion."""

    total: int
    successful: int
    failed: int
    results: List[IngestionResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.successful / self.total * 100) if self.total > 0 else 0.0

    def get_failed(self) -> List[IngestionResult]:
        """Get list of failed results."""
        return [r for r in self.results if not r.success]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": f"{self.success_rate:.1f}%",
            "duration_seconds": self.duration_seconds,
            "failed_urns": [r.urn for r in self.get_failed()],
        }


# =============================================================================
# Article Parser
# =============================================================================


class ArticleParser:
    """
    Parser for extracting article structure (commi, lettere, numeri) from text.

    Italian legal text structure follows patterns:
    - Commi: numbered paragraphs (1. ... 2. ...) or unnumbered first comma
    - Lettere: lettered points (a) ... b) ... or a. ... b. ...)
    - Numeri: numbered sub-points (1) ... 2) ... or 1. ... 2. ...)
    """

    # Regex patterns for structure detection
    # Comma split: captures number at line start followed by PERIOD only
    # Italian commi use period (1. 2. 3.), not parenthesis
    COMMA_SPLIT_PATTERN = re.compile(r'(?:^|\n)(\d+)\.\s+')

    # Lettera split: captures single letter followed by ) typically
    # Italian lettere use parenthesis: a) b) c)
    LETTERA_SPLIT_PATTERN = re.compile(r'(?:^|\n)\s*([a-z])\)\s*', re.IGNORECASE)

    # Numero split: captures number followed by ) for sub-numbering
    # Italian numeri within lettere use parenthesis: 1) 2) 3)
    NUMERO_SPLIT_PATTERN = re.compile(r'(?:^|\n)\s*(\d+)\)\s*')

    def parse(
        self,
        text: str,
        article_urn: str,
        data_versione: Optional[str] = None,
    ) -> ArticleStructure:
        """
        Parse article text into structured components.

        Args:
            text: Raw article text
            article_urn: Base URN for the article
            data_versione: Version date for nodes

        Returns:
            ArticleStructure with parsed commi, lettere, numeri
        """
        # Extract article number from URN (e.g., "1453" from "~art1453")
        numero_articolo = self._extract_article_number(article_urn)

        structure = ArticleStructure(
            urn=article_urn,
            numero_articolo=numero_articolo or "",
            testo_completo=text,
            data_versione=data_versione,
        )

        if not text or not text.strip():
            logger.debug("Empty article text for %s", article_urn)
            return structure

        # Parse commi
        commi = self._parse_commi(text, article_urn, data_versione)
        structure.commi = commi

        logger.debug(
            "Parsed article %s: %d commi, %d total nodes",
            article_urn,
            len(commi),
            structure.node_count(),
        )

        return structure

    def _extract_article_number(self, urn: str) -> Optional[str]:
        """Extract article number from URN."""
        match = re.search(r'~art(\d+(?:[a-z]*)?)', urn, re.IGNORECASE)
        return match.group(1) if match else None

    def _parse_commi(
        self,
        text: str,
        article_urn: str,
        data_versione: Optional[str],
    ) -> List[CommaStructure]:
        """Parse text into comma structures."""
        commi = []

        # Try to split by numbered paragraphs
        # re.split with capturing group: ['before', '1', 'after1', '2', 'after2', ...]
        parts = self.COMMA_SPLIT_PATTERN.split(text)

        # Check if we actually found numbered commi
        # If split found matches, parts length > 1 and has alternating structure
        has_numbered_commi = len(parts) > 1 and any(
            p.isdigit() for p in parts[1::2] if p
        )

        if not has_numbered_commi:
            # No numbered commas found - treat entire text as single comma
            if text.strip():
                comma = CommaStructure(
                    posizione=1,
                    testo=text.strip(),
                    urn=self._build_comma_urn(article_urn, 1),
                )
                comma.lettere = self._parse_lettere(
                    text, comma.urn, data_versione
                )
                commi.append(comma)
        else:
            # parts alternates: [text_before, num1, text1, num2, text2, ...]
            i = 0
            # Handle text before first number (if any) as implicit comma 0/preamble
            # We skip preamble for now, or could add as comma 0
            if parts[0].strip():
                # Could be unnumbered intro, skip for numbered articles
                pass
            i = 1  # Start from first number

            # Process numbered commas
            while i < len(parts):
                if i < len(parts) and parts[i].isdigit():
                    num_str = parts[i]
                    text_part = parts[i + 1] if i + 1 < len(parts) else ""

                    try:
                        posizione = int(num_str)
                    except ValueError:
                        i += 2
                        continue

                    comma = CommaStructure(
                        posizione=posizione,
                        testo=text_part.strip(),
                        urn=self._build_comma_urn(article_urn, posizione),
                    )
                    comma.lettere = self._parse_lettere(
                        text_part, comma.urn, data_versione
                    )
                    commi.append(comma)
                i += 2

        return commi

    def _parse_lettere(
        self,
        text: str,
        comma_urn: str,
        data_versione: Optional[str],
    ) -> List[LetteraStructure]:
        """Parse text into lettera structures."""
        lettere = []

        if not text:
            return lettere

        # Split by lettera markers
        parts = self.LETTERA_SPLIT_PATTERN.split(text)

        # Check if we found lettere
        has_lettere = len(parts) > 1 and any(
            len(p) == 1 and p.isalpha() for p in parts[1::2] if p
        )

        if not has_lettere:
            return lettere

        # parts: [text_before, 'a', text_a, 'b', text_b, ...]
        i = 1  # Skip text before first lettera
        while i < len(parts):
            if i < len(parts) and len(parts[i]) == 1 and parts[i].isalpha():
                lettera_char = parts[i].lower()
                text_part = parts[i + 1] if i + 1 < len(parts) else ""

                lettera = LetteraStructure(
                    posizione=lettera_char,
                    testo=text_part.strip(),
                    urn=self._build_lettera_urn(comma_urn, lettera_char),
                )
                lettera.numeri = self._parse_numeri(
                    text_part, lettera.urn, data_versione
                )
                lettere.append(lettera)
            i += 2

        return lettere

    def _parse_numeri(
        self,
        text: str,
        lettera_urn: str,
        data_versione: Optional[str],
    ) -> List[NumeroStructure]:
        """Parse text into numero structures."""
        numeri = []

        if not text:
            return numeri

        # Split by numero markers
        parts = self.NUMERO_SPLIT_PATTERN.split(text)

        # Check if we found numeri
        has_numeri = len(parts) > 1 and any(
            p.isdigit() for p in parts[1::2] if p
        )

        if not has_numeri:
            return numeri

        # parts: [text_before, '1', text_1, '2', text_2, ...]
        i = 1  # Skip text before first numero
        while i < len(parts):
            if i < len(parts) and parts[i].isdigit():
                num_str = parts[i]
                text_part = parts[i + 1] if i + 1 < len(parts) else ""

                try:
                    posizione = int(num_str)
                except ValueError:
                    i += 2
                    continue

                numero = NumeroStructure(
                    posizione=posizione,
                    testo=text_part.strip(),
                    urn=self._build_numero_urn(lettera_urn, posizione),
                )
                numeri.append(numero)
            i += 2

        return numeri

    def _build_comma_urn(self, article_urn: str, posizione: int) -> str:
        """Build URN for comma node."""
        # Format: base_urn-com1, base_urn-com2, etc.
        return f"{article_urn}-com{posizione}"

    def _build_lettera_urn(self, comma_urn: str, lettera: str) -> str:
        """Build URN for lettera node."""
        # Format: comma_urn-leta, comma_urn-letb, etc.
        return f"{comma_urn}-let{lettera}"

    def _build_numero_urn(self, lettera_urn: str, posizione: int) -> str:
        """Build URN for numero node."""
        # Format: lettera_urn-num1, lettera_urn-num2, etc.
        return f"{lettera_urn}-num{posizione}"


# =============================================================================
# Norm Ingester
# =============================================================================


class NormIngester:
    """
    Ingest norms into FalkorDB Knowledge Graph.

    Handles MERGE (upsert) operations to ensure idempotent re-ingestion,
    creates hierarchical node structure, and supports batch processing.
    """

    BATCH_SIZE = 100  # Articles per transaction batch
    PROGRESS_LOG_INTERVAL = 50  # Log progress every N articles

    def __init__(
        self,
        client: "FalkorDBClient",
        extract_relations: bool = True,
    ):
        """
        Initialize ingester.

        Args:
            client: Connected FalkorDBClient instance
            extract_relations: If True, extract and create citation/modification
                              edges during ingestion (requires relations module)
        """
        self.client = client
        self.parser = ArticleParser()
        self.extract_relations = extract_relations
        self._relation_creator: Optional["RelationCreator"] = None

        if extract_relations:
            # Lazy import to avoid circular dependency
            from visualex.graph.relations import RelationCreator
            self._relation_creator = RelationCreator(client)

    async def ingest_article(
        self,
        norma_visitata: "NormaVisitata",
        article_text: str,
        brocardi_info: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """
        Ingest a single article with its structure.

        Args:
            norma_visitata: NormaVisitata instance with article metadata
            article_text: Full article text to parse
            brocardi_info: Optional Brocardi enrichment data

        Returns:
            IngestionResult with counts and status
        """
        urn = norma_visitata.urn

        try:
            # Parse article structure
            structure = self.parser.parse(
                text=article_text,
                article_urn=urn,
                data_versione=norma_visitata.data_versione,
            )

            nodes_created = 0
            nodes_updated = 0
            edges_created = 0

            # Create/update Norma (article) node
            norma_result = await self._merge_norma_node(norma_visitata, structure)
            if norma_result:
                if norma_result.get("created"):
                    nodes_created += 1
                else:
                    nodes_updated += 1
                # Count Versione node if created
                if norma_result.get("versione_created"):
                    nodes_created += 1
                    edges_created += 1  # ha_versione edge

            # Create structure nodes (Comma -> Lettera -> Numero)
            struct_result = await self._ingest_structure(structure, urn)
            nodes_created += struct_result.get("nodes_created", 0)
            edges_created += struct_result.get("edges_created", 0)

            # Add Brocardi enrichment if available
            if brocardi_info:
                enrich_result = await self._ingest_brocardi_enrichment(
                    urn, brocardi_info
                )
                nodes_created += enrich_result.get("nodes_created", 0)
                edges_created += enrich_result.get("edges_created", 0)

            # Extract and create citation/modification relations (T5: integration)
            relations_created = 0
            if self._relation_creator and article_text:
                try:
                    rel_result = await self._relation_creator.create_relations_from_text(
                        source_urn=urn,
                        text=article_text,
                    )
                    relations_created = (
                        rel_result.get("citations_created", 0) +
                        rel_result.get("modifications_created", 0) +
                        rel_result.get("jurisprudence_created", 0)
                    )
                    edges_created += relations_created

                    # Process Brocardi relations if available
                    if brocardi_info:
                        brocardi_rel_result = await self._relation_creator.process_brocardi_relations(
                            article_urn=urn,
                            brocardi_info=brocardi_info,
                        )
                        edges_created += brocardi_rel_result.get("interpreta_edges", 0)
                except Exception as e:
                    logger.warning("Relation extraction failed for %s: %s", urn, e)

            logger.info(
                "Ingested article %s: %d nodes, %d edges (incl. %d relations)",
                urn, nodes_created, edges_created, relations_created
            )

            return IngestionResult(
                urn=urn,
                success=True,
                nodes_created=nodes_created,
                nodes_updated=nodes_updated,
                edges_created=edges_created,
            )

        except Exception as e:
            logger.error("Failed to ingest article %s: %s", urn, e)
            return IngestionResult(
                urn=urn,
                success=False,
                error=str(e),
            )

    async def ingest_batch(
        self,
        articles: List[Tuple["NormaVisitata", str, Optional[Dict[str, Any]]]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """
        Ingest multiple articles with batch processing.

        Args:
            articles: List of tuples (NormaVisitata, article_text, brocardi_info)
            progress_callback: Optional callback(current, total, urn) for progress

        Returns:
            BatchResult with summary and individual results
        """
        start_time = time.time()
        results: List[IngestionResult] = []
        successful = 0
        failed = 0
        total = len(articles)

        logger.info("Starting batch ingestion of %d articles", total)

        for i, (norma_visitata, article_text, brocardi_info) in enumerate(articles):
            try:
                result = await self.ingest_article(
                    norma_visitata, article_text, brocardi_info
                )
                results.append(result)

                if result.success:
                    successful += 1
                else:
                    failed += 1

                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total, norma_visitata.urn)

                # Log progress periodically
                if (i + 1) % self.PROGRESS_LOG_INTERVAL == 0:
                    logger.info(
                        "Progress: %d/%d articles (%.1f%%)",
                        i + 1, total, (i + 1) / total * 100
                    )

            except Exception as e:
                # Isolate failure - continue with next article
                logger.error(
                    "Batch item %d failed (URN: %s): %s",
                    i, norma_visitata.urn if norma_visitata else "unknown", e
                )
                results.append(IngestionResult(
                    urn=norma_visitata.urn if norma_visitata else "unknown",
                    success=False,
                    error=str(e),
                ))
                failed += 1

        duration = time.time() - start_time

        logger.info(
            "Batch ingestion complete: %d/%d successful (%.1f%%) in %.2fs",
            successful, total, successful / total * 100 if total > 0 else 0, duration
        )

        return BatchResult(
            total=total,
            successful=successful,
            failed=failed,
            results=results,
            duration_seconds=duration,
        )

    async def _merge_norma_node(
        self,
        norma_visitata: "NormaVisitata",
        structure: ArticleStructure,
    ) -> Dict[str, Any]:
        """
        Create or update Norma node and track versioning.

        Returns:
            Dict with 'node', 'created' (bool), 'versione_created' (bool)
        """
        urn = norma_visitata.urn
        data_versione = structure.data_versione or datetime.now().isoformat()[:10]

        # Check if node already exists to track created vs updated
        existing = await self.client.get_node_by_urn(NodeType.NORMA, urn)
        is_new = existing is None

        data = {
            "urn": urn,
            "numero_articolo": structure.numero_articolo,
            "rubrica": structure.rubrica,
            "testo_vigente": structure.testo_completo,
            "data_versione": data_versione,
            "stato": "vigente",
            # From Norma base
            "fonte": norma_visitata.norma.tipo_atto_str,
            "estremi": str(norma_visitata.norma),
        }

        # Filter None values
        data = {k: v for k, v in data.items() if v is not None}

        node_result = await self.client.merge_node(
            NodeType.NORMA,
            match_key="urn",
            match_value=urn,
            data=data,
        )

        result = {
            "node": node_result,
            "created": is_new,
            "versione_created": False,
        }

        # Create Versione node for temporal tracking (AC2)
        if not is_new and existing:
            # Check if version date changed - create new Versione
            old_version_date = None
            if isinstance(existing.get("n"), dict):
                props = existing["n"].get("properties", {})
                old_version_date = props.get("data_versione")

            if old_version_date and old_version_date != data_versione:
                versione_result = await self._create_versione_node(
                    urn, old_version_date, data_versione
                )
                result["versione_created"] = versione_result is not None

        return result

    async def _create_versione_node(
        self,
        norma_urn: str,
        data_inizio: str,
        data_fine: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a Versione node for temporal tracking.

        Links previous version to new version via ha_versione edge.
        """
        versione_id = f"ver_{norma_urn.replace(':', '_')}_{data_inizio}"

        versione_data = {
            "node_id": versione_id,
            "data_inizio_validita": data_inizio,
            "data_fine_validita": data_fine,
            "descrizione_modifiche": f"Versione dal {data_inizio} al {data_fine}",
        }

        result = await self.client.merge_node(
            NodeType.VERSIONE,
            match_key="node_id",
            match_value=versione_id,
            data=versione_data,
        )

        # Create edge Norma -[ha_versione]-> Versione
        if result:
            await self.client.create_edge(
                EdgeType.HA_VERSIONE,
                NodeType.NORMA, norma_urn,
                NodeType.VERSIONE, versione_id,
                from_key="urn",
                to_key="node_id",
            )

        return result

    async def _ingest_structure(
        self,
        structure: ArticleStructure,
        parent_urn: str,
    ) -> Dict[str, int]:
        """Ingest hierarchical structure (commi -> lettere -> numeri)."""
        nodes_created = 0
        edges_created = 0

        for comma in structure.commi:
            # Create Comma node
            comma_data = {
                "urn": comma.urn,
                "posizione": comma.posizione,
                "testo": comma.testo,
                "data_versione": structure.data_versione,
            }
            comma_data = {k: v for k, v in comma_data.items() if v is not None}

            await self.client.merge_node(
                NodeType.COMMA,
                match_key="urn",
                match_value=comma.urn,
                data=comma_data,
            )
            nodes_created += 1

            # Create edge Norma -[contiene]-> Comma
            await self.client.create_edge(
                EdgeType.CONTIENE,
                NodeType.NORMA, parent_urn,
                NodeType.COMMA, comma.urn,
            )
            edges_created += 1

            # Process lettere
            for lettera in comma.lettere:
                lettera_data = {
                    "urn": lettera.urn,
                    "posizione": lettera.posizione,
                    "testo": lettera.testo,
                    "data_versione": structure.data_versione,
                }
                lettera_data = {k: v for k, v in lettera_data.items() if v is not None}

                await self.client.merge_node(
                    NodeType.LETTERA,
                    match_key="urn",
                    match_value=lettera.urn,
                    data=lettera_data,
                )
                nodes_created += 1

                # Edge Comma -[contiene]-> Lettera
                await self.client.create_edge(
                    EdgeType.CONTIENE,
                    NodeType.COMMA, comma.urn,
                    NodeType.LETTERA, lettera.urn,
                )
                edges_created += 1

                # Process numeri
                for numero in lettera.numeri:
                    numero_data = {
                        "urn": numero.urn,
                        "posizione": numero.posizione,
                        "testo": numero.testo,
                        "data_versione": structure.data_versione,
                    }
                    numero_data = {k: v for k, v in numero_data.items() if v is not None}

                    await self.client.merge_node(
                        NodeType.NUMERO,
                        match_key="urn",
                        match_value=numero.urn,
                        data=numero_data,
                    )
                    nodes_created += 1

                    # Edge Lettera -[contiene]-> Numero
                    await self.client.create_edge(
                        EdgeType.CONTIENE,
                        NodeType.LETTERA, lettera.urn,
                        NodeType.NUMERO, numero.urn,
                    )
                    edges_created += 1

        return {
            "nodes_created": nodes_created,
            "edges_created": edges_created,
        }

    async def _ingest_brocardi_enrichment(
        self,
        article_urn: str,
        brocardi_info: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Ingest Brocardi enrichment data.

        Creates Dottrina nodes for ratio/spiegazione and
        AttoGiudiziario nodes for massime.
        """
        nodes_created = 0
        edges_created = 0

        # Create Dottrina node for ratio/spiegazione
        if brocardi_info.get("ratio") or brocardi_info.get("spiegazione"):
            dottrina_node_id = f"brocardi_{article_urn.replace(':', '_')}"
            dottrina_data = {
                "node_id": dottrina_node_id,
                "titolo": f"Commento Brocardi - {article_urn}",
                "descrizione": brocardi_info.get("spiegazione", ""),
                "fonte": "Brocardi.it",
            }

            if brocardi_info.get("ratio"):
                dottrina_data["titolo"] = brocardi_info["ratio"]

            dottrina_data = {k: v for k, v in dottrina_data.items() if v}

            await self.client.merge_node(
                NodeType.DOTTRINA,
                match_key="node_id",
                match_value=dottrina_node_id,
                data=dottrina_data,
            )
            nodes_created += 1

            # Edge Dottrina -[commenta]-> Norma
            await self.client.create_edge(
                EdgeType.COMMENTA,
                NodeType.DOTTRINA, dottrina_node_id,
                NodeType.NORMA, article_urn,
                from_key="node_id",
                to_key="urn",
            )
            edges_created += 1

        # Create AttoGiudiziario nodes for massime
        massime = brocardi_info.get("massime", [])
        for i, massima in enumerate(massime):
            if not massima:
                continue

            atto_node_id = f"massima_{article_urn.replace(':', '_')}_{i}"
            atto_data = {
                "node_id": atto_node_id,
                "massima": massima.get("testo", str(massima)) if isinstance(massima, dict) else str(massima),
                "fonte": "Brocardi.it",
                "tipologia": "massima",
            }

            if isinstance(massima, dict):
                if massima.get("corte"):
                    atto_data["organo_emittente"] = massima["corte"]
                if massima.get("data"):
                    atto_data["data"] = massima["data"]
                if massima.get("numero"):
                    atto_data["estremi"] = massima["numero"]

            atto_data = {k: v for k, v in atto_data.items() if v}

            await self.client.merge_node(
                NodeType.ATTO_GIUDIZIARIO,
                match_key="node_id",
                match_value=atto_node_id,
                data=atto_data,
            )
            nodes_created += 1

            # Edge AttoGiudiziario -[interpreta]-> Norma
            await self.client.create_edge(
                EdgeType.INTERPRETA,
                NodeType.ATTO_GIUDIZIARIO, atto_node_id,
                NodeType.NORMA, article_urn,
                from_key="node_id",
                to_key="urn",
            )
            edges_created += 1

        return {
            "nodes_created": nodes_created,
            "edges_created": edges_created,
        }
