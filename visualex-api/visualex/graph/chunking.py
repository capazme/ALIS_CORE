"""
Legal Text Chunking Module
==========================

Source-type-aware chunking for legal texts, optimized for embedding and retrieval.

Supports:
- Norm chunking (comma-level, respects legal structure)
- Commentary chunking (paragraph-level, preserves massime)
- Jurisprudence chunking (preserves case citations)
- Doctrine chunking (section-level, preserves references)

Each chunk includes rich metadata for Bridge Table mapping and expert affinity.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

__all__ = [
    "SourceType",
    "ChunkResult",
    "LegalChunker",
    "ChunkingConfig",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class SourceType(str, Enum):
    """Type of legal source being chunked."""

    NORM = "norm"  # Legislative text (leggi, decreti, codici)
    JURISPRUDENCE = "jurisprudence"  # Court decisions (sentenze, massime)
    COMMENTARY = "commentary"  # Legal commentary (Brocardi, commentari)
    DOCTRINE = "doctrine"  # Academic treatises (manuali, trattati)


# Default authority scores by source type
SOURCE_AUTHORITY = {
    SourceType.NORM: 1.0,  # Primary source, highest authority
    SourceType.JURISPRUDENCE: 0.8,  # Cassazione level
    SourceType.COMMENTARY: 0.5,  # Secondary interpretive source
    SourceType.DOCTRINE: 0.4,  # Academic opinion
}

# Chunking parameters by source type
CHUNK_CONFIG = {
    SourceType.NORM: {
        "max_tokens": 512,
        "min_tokens": 50,
        "overlap_tokens": 50,
        "split_on_comma": True,
    },
    SourceType.JURISPRUDENCE: {
        "max_tokens": 512,
        "min_tokens": 100,
        "overlap_tokens": 100,
        "split_on_comma": False,
    },
    SourceType.COMMENTARY: {
        "max_tokens": 400,
        "min_tokens": 80,
        "overlap_tokens": 80,
        "split_on_comma": False,
    },
    SourceType.DOCTRINE: {
        "max_tokens": 512,
        "min_tokens": 100,
        "overlap_tokens": 100,
        "split_on_comma": False,
    },
}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""

    max_tokens: int = 512
    min_tokens: int = 50
    overlap_tokens: int = 50
    split_on_comma: bool = True


@dataclass
class ChunkResult:
    """Result of chunking a document."""

    chunk_id: str
    text: str
    source_type: SourceType
    source_urn: str
    source_authority: float
    chunk_position: int
    token_count: int
    parent_article_urn: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_type": self.source_type.value,
            "source_urn": self.source_urn,
            "source_authority": self.source_authority,
            "chunk_position": self.chunk_position,
            "token_count": self.token_count,
            "parent_article_urn": self.parent_article_urn,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class ChunkBatchResult:
    """Result of batch chunking."""

    total_documents: int
    total_chunks: int
    chunks: List[ChunkResult] = field(default_factory=list)
    failed_documents: List[Dict[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "failed_count": len(self.failed_documents),
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# Legal Chunker
# =============================================================================


class LegalChunker:
    """
    Source-type-aware chunker for legal texts.

    Applies different chunking strategies based on source type:
    - NORM: Comma-level chunking respecting legal structure
    - COMMENTARY: Paragraph-level, keeps massime intact
    - JURISPRUDENCE: Preserves case citations
    - DOCTRINE: Section-level with reference linking
    """

    # Pattern for comma/paragraph splitting in norms
    COMMA_PATTERN = re.compile(r"(?:^|\n)(\d+)\.\s+")

    # Pattern for paragraph splitting in prose
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    # Pattern for massime (kept as single chunks)
    MASSIMA_PATTERN = re.compile(
        r"(Massima:.*?)(?=\n\n|Massima:|$)", re.DOTALL | re.IGNORECASE
    )


    # Approximate tokens per word for Italian legal text.
    # Based on empirical observation: Italian legal language averages ~1.3 tokens/word
    # with common tokenizers (GPT, BERT). Legal terms like "inadempimento" or
    # "costituzionalità" often split into multiple subword tokens.
    TOKENS_PER_WORD = 1.3

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize LegalChunker.

        Args:
            config: Optional default configuration (overridden per source type)
        """
        self.default_config = config or ChunkingConfig()

    def chunk_document(
        self,
        text: str,
        source_urn: str,
        source_type: SourceType,
        parent_article_urn: Optional[str] = None,
        source_authority: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """
        Chunk a document based on its source type.

        Args:
            text: Document text to chunk
            source_urn: URN of the source document
            source_type: Type of legal source
            parent_article_urn: URN of parent article (for norm chunks)
            source_authority: Override default authority score
            extra_metadata: Additional metadata to include in each chunk

        Returns:
            List of ChunkResult objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking: %s", source_urn)
            return []

        # Get config for this source type
        config = self._get_config(source_type)

        # Get authority score
        authority = source_authority or SOURCE_AUTHORITY.get(source_type, 0.5)

        # Apply appropriate chunking strategy
        if source_type == SourceType.NORM:
            raw_chunks = self._chunk_norm(text, config)
        elif source_type == SourceType.COMMENTARY:
            raw_chunks = self._chunk_commentary(text, config)
        elif source_type == SourceType.JURISPRUDENCE:
            raw_chunks = self._chunk_jurisprudence(text, config)
        elif source_type == SourceType.DOCTRINE:
            raw_chunks = self._chunk_doctrine(text, config)
        else:
            # Fallback to simple chunking
            raw_chunks = self._chunk_simple(text, config)

        # Build ChunkResult objects
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = str(uuid.uuid4())
            token_count = self._estimate_tokens(chunk_text)

            metadata = {
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                **(extra_metadata or {}),
            }

            chunk = ChunkResult(
                chunk_id=chunk_id,
                text=chunk_text,
                source_type=source_type,
                source_urn=source_urn,
                source_authority=authority,
                chunk_position=i,
                token_count=token_count,
                parent_article_urn=parent_article_urn or source_urn,
                metadata=metadata,
            )
            chunks.append(chunk)

        logger.debug(
            "Chunked %s into %d chunks (source_type=%s)",
            source_urn,
            len(chunks),
            source_type.value,
        )

        return chunks

    def chunk_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> ChunkBatchResult:
        """
        Process a batch of documents.

        Each document dict should contain:
        - text: str (required)
        - source_urn: str (required)
        - source_type: SourceType or str (required)
        - parent_article_urn: Optional[str]
        - source_authority: Optional[float]
        - metadata: Optional[Dict]

        Args:
            documents: List of document dictionaries

        Returns:
            ChunkBatchResult with all chunks and failure info
        """
        start_time = datetime.now()
        all_chunks: List[ChunkResult] = []
        failed: List[Dict[str, str]] = []

        for i, doc in enumerate(documents):
            try:
                # Extract required fields
                text = doc.get("text", "")
                source_urn = doc.get("source_urn", f"unknown:{i}")

                # Parse source type
                source_type_raw = doc.get("source_type", SourceType.NORM)
                if isinstance(source_type_raw, str):
                    source_type = SourceType(source_type_raw)
                else:
                    source_type = source_type_raw

                chunks = self.chunk_document(
                    text=text,
                    source_urn=source_urn,
                    source_type=source_type,
                    parent_article_urn=doc.get("parent_article_urn"),
                    source_authority=doc.get("source_authority"),
                    extra_metadata=doc.get("metadata"),
                )
                all_chunks.extend(chunks)

                # Log progress
                if (i + 1) % 50 == 0:
                    logger.info(
                        "Batch chunking progress: %d/%d documents",
                        i + 1,
                        len(documents),
                    )

            except Exception as e:
                logger.error(
                    "Error chunking document %s: %s",
                    doc.get("source_urn", f"index:{i}"),
                    str(e),
                )
                failed.append({
                    "source_urn": doc.get("source_urn", f"index:{i}"),
                    "error": str(e),
                })

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(
            "Batch chunking complete: %d documents -> %d chunks (%.2fs)",
            len(documents),
            len(all_chunks),
            duration,
        )

        return ChunkBatchResult(
            total_documents=len(documents),
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            failed_documents=failed,
            duration_seconds=round(duration, 2),
        )

    # =========================================================================
    # Chunking Strategies
    # =========================================================================

    def _chunk_norm(self, text: str, config: ChunkingConfig) -> List[str]:
        """
        Chunk legislative text at comma boundaries.

        Respects legal structure: articles -> commi -> lettere.
        """
        chunks = []

        # Try to split by numbered commi
        parts = self.COMMA_PATTERN.split(text)

        if len(parts) > 1:
            # Numbered commi found
            i = 0
            # Handle text before first number (preamble)
            if parts[0].strip():
                chunks.append(parts[0].strip())
            i = 1

            while i < len(parts):
                if i < len(parts) and parts[i].isdigit():
                    comma_num = parts[i]
                    comma_text = parts[i + 1] if i + 1 < len(parts) else ""

                    if comma_text.strip():
                        # Prefix with comma number for context
                        chunk_text = f"{comma_num}. {comma_text.strip()}"

                        # Check token count
                        tokens = self._estimate_tokens(chunk_text)
                        if tokens > config.max_tokens:
                            # Split further if too long
                            sub_chunks = self._split_long_text(
                                chunk_text, config.max_tokens, config.overlap_tokens
                            )
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append(chunk_text)
                i += 2
        else:
            # No numbered commi - split by sentence or paragraph
            chunks = self._chunk_simple(text, config)

        return chunks

    def _chunk_commentary(self, text: str, config: ChunkingConfig) -> List[str]:
        """
        Chunk commentary text, preserving massime as single units.
        """
        chunks = []

        # First, extract and preserve massime
        massime = self.MASSIMA_PATTERN.findall(text)
        for massima in massime:
            if massima.strip():
                chunks.append(massima.strip())

        # Remove massime from text
        remaining_text = self.MASSIMA_PATTERN.sub("", text)

        # Split remaining text by paragraphs
        paragraphs = self.PARAGRAPH_PATTERN.split(remaining_text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            tokens = self._estimate_tokens(para)
            if tokens > config.max_tokens:
                sub_chunks = self._split_long_text(
                    para, config.max_tokens, config.overlap_tokens
                )
                chunks.extend(sub_chunks)
            else:
                # Include all paragraphs (don't skip short ones)
                chunks.append(para)

        return chunks

    def _chunk_jurisprudence(self, text: str, config: ChunkingConfig) -> List[str]:
        """
        Chunk court decisions, preserving legal citations within chunks.
        """
        # Split by paragraphs
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        chunks = []

        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._estimate_tokens(para)

            # Check if adding this paragraph would exceed max
            if current_tokens + para_tokens > config.max_tokens and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(
                    current_chunk, config.overlap_tokens
                )
                current_chunk = overlap_text + " " + para
                current_tokens = self._estimate_tokens(current_chunk)
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens

        # Don't forget last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_doctrine(self, text: str, config: ChunkingConfig) -> List[str]:
        """
        Chunk academic/doctrine text by paragraphs.

        Uses same strategy as jurisprudence (paragraph-based with overlap).
        Future enhancement: add section header detection for treatises.
        """
        return self._chunk_jurisprudence(text, config)

    def _chunk_simple(self, text: str, config: ChunkingConfig) -> List[str]:
        """
        Simple fallback chunking by token count with overlap.
        """
        return self._split_long_text(text, config.max_tokens, config.overlap_tokens)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_config(self, source_type: SourceType) -> ChunkingConfig:
        """Get chunking config for source type."""
        type_config = CHUNK_CONFIG.get(source_type, {})
        return ChunkingConfig(
            max_tokens=type_config.get("max_tokens", 512),
            min_tokens=type_config.get("min_tokens", 50),
            overlap_tokens=type_config.get("overlap_tokens", 50),
            split_on_comma=type_config.get("split_on_comma", True),
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (words * 1.3 for Italian legal text)."""
        if not text:
            return 0
        words = len(text.split())
        return int(words * self.TOKENS_PER_WORD)

    def _split_long_text(
        self, text: str, max_tokens: int, overlap_tokens: int
    ) -> List[str]:
        """Split long text into chunks with overlap."""
        words = text.split()
        max_words = int(max_tokens / self.TOKENS_PER_WORD)
        overlap_words = int(overlap_tokens / self.TOKENS_PER_WORD)

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + max_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break

            start = end - overlap_words

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens of text for overlap."""
        words = text.split()
        overlap_words = int(overlap_tokens / self.TOKENS_PER_WORD)
        if len(words) <= overlap_words:
            return text
        return " ".join(words[-overlap_words:])
