"""
Admin Operations Module
=======================

Administrative operations for the Knowledge Graph, including manual ingestion.

Provides:
- IngestService: Manual ingestion trigger for norms
- Audit logging for admin operations
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Any, Optional

from visualex.models.norma import Norma, NormaVisitata
from visualex.graph.ingestion import NormIngester, IngestionResult
from visualex.utils.map import NORMATTIVA_URN_CODICI, extract_codice_details

if TYPE_CHECKING:
    from visualex.graph.client import FalkorDBClient
    from visualex.scrapers.normattiva import NormattivaScraper
    from visualex.scrapers.brocardi import BrocardiScraper

__all__ = [
    "IngestService",
    "IngestRequest",
    "IngestJobResult",
]

logger = logging.getLogger(__name__)

# Maximum articles per ingest request to prevent system overload
MAX_ARTICLES_PER_REQUEST = 100


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class IngestRequest:
    """Request for ingestion operation."""

    # Single URN mode
    urn: Optional[str] = None

    # Range/list mode
    act_type: Optional[str] = None
    article_range: Optional[str] = None  # e.g., "1470-1490"
    articles: Optional[List[str]] = None  # e.g., ["1453", "1454"]

    # Options
    include_brocardi: bool = True
    force_refresh: bool = False  # TODO: Implement cache bypass when True

    def validate(self) -> Optional[str]:
        """
        Validate request parameters.

        Returns:
            None if valid, error message string if invalid.
        """
        if self.urn:
            # Single URN mode - valid
            return None
        if self.act_type:
            if self.article_range or self.articles:
                return None
            return "act_type requires article_range or articles"
        return "Either urn or act_type with articles must be provided"


@dataclass
class IngestJobResult:
    """Result of an ingestion job."""

    job_id: str
    status: str  # "completed", "partial", "failed"
    total: int
    succeeded: int
    failed: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "results": self.results,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# Ingest Service
# =============================================================================


class IngestService:
    """
    Service for manual ingestion of norms into the Knowledge Graph.

    Supports:
    - Single article ingestion by URN
    - Range ingestion (e.g., articles 1470-1490)
    - List ingestion (specific articles)
    """

    def __init__(
        self,
        client: "FalkorDBClient",
        normattiva_scraper: "NormattivaScraper",
        brocardi_scraper: Optional["BrocardiScraper"] = None,
    ):
        """
        Initialize IngestService.

        Args:
            client: FalkorDB client for graph operations
            normattiva_scraper: Scraper for fetching article text
            brocardi_scraper: Optional scraper for Brocardi enrichment
        """
        self.client = client
        self.normattiva = normattiva_scraper
        self.brocardi = brocardi_scraper
        self.ingester = NormIngester(client, extract_relations=True)

    async def ingest(self, request: IngestRequest) -> IngestJobResult:
        """
        Execute ingestion based on request.

        Args:
            request: IngestRequest with URN, range, or article list

        Returns:
            IngestJobResult with detailed results
        """
        job_id = str(uuid.uuid4())
        start_time = datetime.now()

        logger.info(
            "Starting ingest job %s: urn=%s, act_type=%s, range=%s, articles=%s",
            job_id,
            request.urn,
            request.act_type,
            request.article_range,
            request.articles,
        )

        # Determine articles to ingest
        if request.urn:
            articles_to_ingest = [request.urn]
        else:
            articles_to_ingest = self._resolve_articles(
                request.act_type,
                request.article_range,
                request.articles,
            )

        total = len(articles_to_ingest)
        results = []
        succeeded = 0
        failed = 0

        # Process each article
        for i, article_spec in enumerate(articles_to_ingest):
            try:
                result = await self._ingest_single(
                    article_spec,
                    request.act_type,
                    request.include_brocardi,
                )
                results.append(result.to_dict() if hasattr(result, 'to_dict') else result)
                if result.success:
                    succeeded += 1
                else:
                    failed += 1

                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    logger.info(
                        "Ingest job %s progress: %d/%d (success=%d, failed=%d)",
                        job_id, i + 1, total, succeeded, failed,
                    )

            except Exception as e:
                logger.error(
                    "Error ingesting %s in job %s: %s",
                    article_spec, job_id, str(e),
                )
                failed += 1
                error_result = IngestionResult(
                    urn=article_spec,
                    success=False,
                    error=str(e),
                )
                results.append(error_result.to_dict())

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Determine status
        if failed == 0:
            status = "completed"
        elif succeeded == 0:
            status = "failed"
        else:
            status = "partial"

        # Log audit entry
        self._log_audit(job_id, status, total, succeeded, failed, duration)

        return IngestJobResult(
            job_id=job_id,
            status=status,
            total=total,
            succeeded=succeeded,
            failed=failed,
            results=results,
            completed_at=end_time.isoformat(),
            duration_seconds=round(duration, 2),
        )

    async def ingest_by_urn(self, urn: str, include_brocardi: bool = True) -> IngestionResult:
        """
        Ingest a single article by URN.

        Args:
            urn: Full URN (e.g., "urn:nir:stato:regio.decreto:1942-03-16;262~art1453")
            include_brocardi: Whether to fetch Brocardi enrichment

        Returns:
            IngestionResult with ingestion details
        """
        return await self._ingest_single(urn, None, include_brocardi)

    def _resolve_articles(
        self,
        act_type: str,
        article_range: Optional[str],
        articles: Optional[List[str]],
    ) -> List[str]:
        """Resolve article range or list to list of article numbers."""
        if articles:
            if len(articles) > MAX_ARTICLES_PER_REQUEST:
                logger.warning(
                    "Article list truncated from %d to %d",
                    len(articles), MAX_ARTICLES_PER_REQUEST,
                )
                return articles[:MAX_ARTICLES_PER_REQUEST]
            return articles

        if article_range:
            # Parse range like "1470-1490"
            if "-" in article_range:
                parts = article_range.split("-")
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    # Enforce max range size
                    if end - start + 1 > MAX_ARTICLES_PER_REQUEST:
                        logger.warning(
                            "Article range %s exceeds max %d, truncating",
                            article_range, MAX_ARTICLES_PER_REQUEST,
                        )
                        end = start + MAX_ARTICLES_PER_REQUEST - 1
                    return [str(i) for i in range(start, end + 1)]
                except ValueError:
                    logger.warning("Invalid article range: %s", article_range)
                    return [article_range]
            return [article_range]

        return []

    async def _ingest_single(
        self,
        article_spec: str,
        act_type: Optional[str],
        include_brocardi: bool,
    ) -> IngestionResult:
        """
        Ingest a single article.

        Args:
            article_spec: URN or article number
            act_type: Act type (required if article_spec is just a number)
            include_brocardi: Whether to fetch Brocardi enrichment

        Returns:
            IngestionResult
        """
        # Build NormaVisitata
        if article_spec.startswith("urn:"):
            # Full URN provided - parse it
            norma_visitata = self._parse_urn_to_norma_visitata(article_spec)
        else:
            # Article number provided - need act_type
            if not act_type:
                return IngestionResult(
                    urn=article_spec,
                    success=False,
                    error="act_type required for article number",
                )
            norma_visitata = self._build_norma_visitata(act_type, article_spec)

        if not norma_visitata:
            return IngestionResult(
                urn=article_spec,
                success=False,
                error="Could not build NormaVisitata",
            )

        try:
            # Fetch article text
            article_text, url = await self.normattiva.get_document(norma_visitata)

            # Fetch Brocardi info if requested
            brocardi_info = None
            if include_brocardi and self.brocardi:
                try:
                    brocardi_result = await self.brocardi.get_info(norma_visitata)
                    if brocardi_result and brocardi_result[1]:
                        brocardi_info = brocardi_result[1]
                except Exception as e:
                    logger.warning(
                        "Brocardi fetch failed for %s: %s",
                        norma_visitata.urn, str(e),
                    )

            # Ingest into graph
            result = await self.ingester.ingest_article(
                norma_visitata=norma_visitata,
                article_text=article_text,
                brocardi_info=brocardi_info,
            )

            return result

        except Exception as e:
            logger.error(
                "Error fetching/ingesting %s: %s",
                norma_visitata.urn, str(e),
            )
            return IngestionResult(
                urn=norma_visitata.urn,
                success=False,
                error=str(e),
            )

    def _parse_urn_to_norma_visitata(self, urn: str) -> Optional[NormaVisitata]:
        """Parse a full URN into NormaVisitata."""
        # Pattern: urn:nir:authority:act.type:date;number~artN
        # Example: urn:nir:stato:regio.decreto:1942-03-16;262~art1453
        pattern = r"urn:nir:([^:]+):([^:]+):(\d{4}-\d{2}-\d{2});(\d+)(?::(\d+))?(?:~art(.+))?"

        match = re.match(pattern, urn)
        if not match:
            logger.warning("Could not parse URN: %s", urn)
            return None

        authority, act_type_urn, date, number, annex, article = match.groups()

        # Convert URN format to readable: "regio.decreto" -> "regio decreto"
        act_type = act_type_urn.replace(".", " ")

        try:
            norma = Norma(
                tipo_atto=act_type,
                data=date,
                numero_atto=number,
            )

            return NormaVisitata(
                norma=norma,
                allegato=annex,
                numero_articolo=article,
            )
        except Exception as e:
            logger.error("Error building NormaVisitata from URN %s: %s", urn, str(e))
            return None

    def _build_norma_visitata(
        self,
        act_type: str,
        article_number: str,
    ) -> Optional[NormaVisitata]:
        """Build NormaVisitata from act type and article number."""
        # Check if act_type is a codice alias
        codice_details = extract_codice_details(act_type)

        if codice_details:
            # Use extracted details
            norma = Norma(
                tipo_atto=act_type,
                data=codice_details["data"],
                numero_atto=codice_details["numero_atto"],
                tipo_atto_reale=codice_details["tipo_atto_reale"],
            )
            # Get annex from codice URN if present
            codice_urn = NORMATTIVA_URN_CODICI.get(act_type.lower().strip(), "")
            annex_match = re.search(r";\d+:(\d+)$", codice_urn)
            annex = annex_match.group(1) if annex_match else None
        else:
            # Non-codice act type
            norma = Norma(tipo_atto=act_type)
            annex = None

        try:
            return NormaVisitata(
                norma=norma,
                allegato=annex,
                numero_articolo=article_number,
            )
        except Exception as e:
            logger.error(
                "Error building NormaVisitata for %s art.%s: %s",
                act_type, article_number, str(e),
            )
            return None

    def _log_audit(
        self,
        job_id: str,
        status: str,
        total: int,
        succeeded: int,
        failed: int,
        duration: float,
    ) -> None:
        """Log audit entry for ingest operation."""
        logger.info(
            "AUDIT: Ingest job %s completed - status=%s, total=%d, succeeded=%d, failed=%d, duration=%.2fs",
            job_id, status, total, succeeded, failed, duration,
        )
