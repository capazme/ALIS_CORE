"""
SQLAlchemy Models for Expert System Q&A.
=========================================

Database models for storing Q&A traces and feedback.

Tables:
- qa_traces: Query execution traces
- qa_feedback: Multi-level feedback (inline, detailed, source-specific, refinement)

Usage:
    from merlt.experts.models import QATrace, QAFeedback
    from merlt.rlcf.database import get_async_session

    async with get_async_session() as session:
        trace = QATrace(user_id="user123", query="Cos'è la legittima difesa?")
        session.add(trace)
        await session.commit()
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime, ForeignKey,
    CheckConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from merlt.rlcf.database import Base


class QATrace(Base):
    """
    Trace di esecuzione query Q&A.

    Registra ogni query sottomessa al MultiExpertOrchestrator,
    gli expert selezionati, la modalità di sintesi, e i risultati.

    Attributes:
        trace_id: UUID unico per il trace
        user_id: ID utente che ha fatto la query
        query: Testo della query originale
        selected_experts: Lista expert utilizzati (es: ['literal', 'systemic'])
        synthesis_mode: convergent | divergent
        synthesis_text: Testo della sintesi finale
        sources: Array JSONB delle fonti citate con metadata
        execution_time_ms: Tempo di esecuzione in millisecondi
        created_at: Timestamp creazione

    Example:
        >>> trace = QATrace(
        ...     user_id="user123",
        ...     query="Cos'è la responsabilità contrattuale?",
        ...     selected_experts=["literal", "systemic"],
        ...     synthesis_mode="convergent",
        ...     synthesis_text="La responsabilità contrattuale...",
        ...     sources=[{"article_urn": "...", "expert": "literal", "relevance": 0.95}],
        ...     execution_time_ms=2450
        ... )
    """
    __tablename__ = "qa_traces"

    # Primary key
    trace_id = Column(
        String(50),
        primary_key=True,
        default=lambda: f"trace_{uuid4().hex[:12]}"
    )

    # User
    user_id = Column(String(50), nullable=False, index=True)

    # Query details
    query = Column(Text, nullable=False)
    selected_experts = Column(ARRAY(String(100)), nullable=True)  # ['literal', 'systemic', ...]

    # Synthesis results
    synthesis_mode = Column(String(20), nullable=True)  # convergent | divergent
    synthesis_text = Column(Text, nullable=True)
    sources = Column(JSONB, nullable=True)  # [{"article_urn": "...", "expert": "...", "relevance": ...}]

    # Performance
    execution_time_ms = Column(Integer, nullable=True)

    # Full scientific pipeline trace (JSON)
    full_trace = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    # Relationships
    feedbacks = relationship("QAFeedback", back_populates="trace", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "synthesis_mode IS NULL OR synthesis_mode IN ('convergent', 'divergent')",
            name="chk_synthesis_mode"
        ),
        Index("idx_qa_traces_user", "user_id"),
        Index("idx_qa_traces_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<QATrace(trace_id={self.trace_id}, query={self.query[:50]}...)>"


class QAFeedback(Base):
    """
    Feedback multi-livello per query Q&A.

    Supporta 5 tipi di feedback:
    1. Inline rating (quick thumbs up/down): inline_rating
    2. Detailed form (3 dimensions): retrieval_score, reasoning_score, synthesis_score
    3. Per-source rating: source_id + source_relevance
    4. Conversational refinement: follow_up_query + refined_trace_id
    5. Expert preference (for divergent mode): preferred_expert

    Un singolo record può contenere uno o più tipi di feedback.

    Attributes:
        id: Primary key auto-increment
        trace_id: FK to qa_traces
        user_id: ID utente che fornisce feedback

        # Type 1: Inline
        inline_rating: 1-5 (1=thumbs down, 5=thumbs up)

        # Type 2: Detailed
        retrieval_score: 0-1 (qualità retrieval)
        reasoning_score: 0-1 (qualità reasoning)
        synthesis_score: 0-1 (qualità sintesi)
        detailed_comment: Commento testuale opzionale

        # Type 3: Per-source
        source_id: URN dell'articolo citato
        source_relevance: 1-5 stars

        # Type 4: Refinement
        follow_up_query: Nuova query di approfondimento
        refined_trace_id: Link al trace generato dal follow-up

        user_authority: Authority dell'utente (per weighted feedback)
        created_at: Timestamp feedback

    Example Type 1 (Inline):
        >>> feedback = QAFeedback(
        ...     trace_id="trace_abc123",
        ...     user_id="user456",
        ...     inline_rating=5  # thumbs up
        ... )

    Example Type 2 (Detailed):
        >>> feedback = QAFeedback(
        ...     trace_id="trace_abc123",
        ...     user_id="user456",
        ...     retrieval_score=0.8,
        ...     reasoning_score=0.9,
        ...     synthesis_score=0.7,
        ...     detailed_comment="Buona risposta ma sintesi migliorabile"
        ... )

    Example Type 3 (Per-source):
        >>> feedback = QAFeedback(
        ...     trace_id="trace_abc123",
        ...     user_id="user456",
        ...     source_id="urn:nir:stato:codice.civile:1942;art1453",
        ...     source_relevance=5  # 5 stars
        ... )

    Example Type 4 (Refinement):
        >>> feedback = QAFeedback(
        ...     trace_id="trace_abc123",
        ...     user_id="user456",
        ...     follow_up_query="Puoi spiegare meglio il requisito della proporzione?",
        ...     refined_trace_id="trace_def456"
        ... )

    Example Type 5 (Expert Preference):
        >>> feedback = QAFeedback(
        ...     trace_id="trace_abc123",
        ...     user_id="user456",
        ...     preferred_expert="systemic",
        ...     detailed_comment="L'interpretazione sistematica e' piu' completa"
        ... )
    """
    __tablename__ = "qa_feedback"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign keys
    trace_id = Column(
        String(50),
        ForeignKey("qa_traces.trace_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id = Column(String(50), nullable=False, index=True)

    # Type 1: Inline rating (quick thumbs)
    inline_rating = Column(Integer, nullable=True)  # 1-5

    # Type 2: Detailed form (3 dimensions)
    retrieval_score = Column(Float, nullable=True)  # 0-1
    reasoning_score = Column(Float, nullable=True)  # 0-1
    synthesis_score = Column(Float, nullable=True)  # 0-1
    detailed_comment = Column(Text, nullable=True)

    # Type 3: Per-source rating
    source_id = Column(String(200), nullable=True)  # article URN
    source_relevance = Column(Integer, nullable=True)  # 1-5 stars

    # Type 4: Conversational refinement
    follow_up_query = Column(Text, nullable=True)
    refined_trace_id = Column(String(50), nullable=True)  # Link to new trace

    # Type 5: Expert preference (for divergent mode)
    preferred_expert = Column(String(50), nullable=True)  # literal, systemic, principles, precedent

    # User authority (for weighted feedback)
    user_authority = Column(Float, nullable=True)

    # Timestamp
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    # Relationships
    trace = relationship("QATrace", back_populates="feedbacks")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "inline_rating IS NULL OR (inline_rating >= 1 AND inline_rating <= 5)",
            name="chk_inline_rating"
        ),
        CheckConstraint(
            "retrieval_score IS NULL OR (retrieval_score >= 0 AND retrieval_score <= 1)",
            name="chk_retrieval_score"
        ),
        CheckConstraint(
            "reasoning_score IS NULL OR (reasoning_score >= 0 AND reasoning_score <= 1)",
            name="chk_reasoning_score"
        ),
        CheckConstraint(
            "synthesis_score IS NULL OR (synthesis_score >= 0 AND synthesis_score <= 1)",
            name="chk_synthesis_score"
        ),
        CheckConstraint(
            "source_relevance IS NULL OR (source_relevance >= 1 AND source_relevance <= 5)",
            name="chk_source_relevance"
        ),
        Index("idx_qa_feedback_trace", "trace_id"),
        Index("idx_qa_feedback_user", "user_id"),
        Index("idx_qa_feedback_type", "inline_rating", "retrieval_score", "source_relevance"),
    )

    def __repr__(self) -> str:
        feedback_type = "unknown"
        if self.inline_rating is not None:
            feedback_type = f"inline_{self.inline_rating}"
        elif self.retrieval_score is not None:
            feedback_type = "detailed"
        elif self.source_id is not None:
            feedback_type = "per_source"
        elif self.follow_up_query is not None:
            feedback_type = "refinement"
        elif self.preferred_expert is not None:
            feedback_type = f"preference_{self.preferred_expert}"

        return f"<QAFeedback(id={self.id}, trace_id={self.trace_id}, type={feedback_type})>"
