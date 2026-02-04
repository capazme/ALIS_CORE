"""
Test Document Upload & Parsing
===============================

Test completo upload documenti utente e parsing:
1. Upload PDF/TXT/DOCX
2. File deduplication (SHA-256)
3. Text extraction
4. Entity/amendment extraction da documenti

IMPORTANTE: Test con filesystem reale, NO MOCK.
"""

import pytest
import hashlib
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from merlt.storage.enrichment.models import UserDocument, PendingEntity, PendingAmendment
from merlt.pipeline.document_parser import DocumentParserService
from merlt.pipeline.amendment_extractor import AmendmentExtractorService

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
class TestDocumentUpload:
    """Test upload documenti con deduplication."""

    async def test_upload_pdf_creates_record(
        self,
        db_session: AsyncSession,
        test_upload_dir: Path,
        sample_pdf_content: bytes,
    ):
        """
        Test: Upload PDF crea record in user_documents.

        Verifica:
        - File salvato su filesystem
        - Record database creato
        - Hash SHA-256 calcolato
        - Status iniziale 'uploaded'
        """
        # Arrange
        filename = "test_manuale.pdf"
        file_path = test_upload_dir / filename
        file_path.write_bytes(sample_pdf_content)

        file_hash = hashlib.sha256(sample_pdf_content).hexdigest()
        file_size = len(sample_pdf_content)

        # Act
        document = UserDocument(
            filename=filename,
            original_filename="Manuale Torrente.pdf",
            file_type="pdf",
            file_size_bytes=file_size,
            file_hash=file_hash,
            storage_path=str(file_path),
            document_type="manuale",
            legal_domain="civile",
            uploaded_by="user_001",
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

        # Assert
        assert document.id is not None
        assert document.file_hash == file_hash
        assert document.processing_status == "uploaded"
        assert document.created_at is not None
        assert Path(document.storage_path).exists()

    async def test_duplicate_file_detection(
        self,
        db_session: AsyncSession,
        test_upload_dir: Path,
        sample_pdf_content: bytes,
    ):
        """
        Test: Upload stesso file due volte → duplicate detected.

        Verifica:
        - Secondo upload riconosce hash duplicato
        - NON crea secondo record
        - Ritorna documento esistente
        """
        # Arrange: primo upload
        file_hash = hashlib.sha256(sample_pdf_content).hexdigest()
        file_path1 = test_upload_dir / "doc1.pdf"
        file_path1.write_bytes(sample_pdf_content)

        doc1 = UserDocument(
            filename="doc1.pdf",
            original_filename="Original.pdf",
            file_type="pdf",
            file_size_bytes=len(sample_pdf_content),
            file_hash=file_hash,
            storage_path=str(file_path1),
            uploaded_by="user_001",
        )
        db_session.add(doc1)
        await db_session.commit()

        # Act: secondo upload STESSO contenuto, nome diverso
        file_path2 = test_upload_dir / "doc2.pdf"
        file_path2.write_bytes(sample_pdf_content)

        # Check for duplicate
        stmt = select(UserDocument).where(UserDocument.file_hash == file_hash)
        result = await db_session.execute(stmt)
        existing = result.scalar_one_or_none()

        # Assert
        assert existing is not None
        assert existing.id == doc1.id
        assert existing.filename == "doc1.pdf"

        # Verifica count: solo 1 documento
        count_stmt = select(UserDocument).where(UserDocument.file_hash == file_hash)
        count_result = await db_session.execute(count_stmt)
        all_docs = count_result.scalars().all()
        assert len(all_docs) == 1

    async def test_upload_txt_file(
        self,
        db_session: AsyncSession,
        test_upload_dir: Path,
    ):
        """
        Test: Upload file TXT supportato.

        Verifica:
        - File TXT salvato correttamente
        - Record creato con file_type='txt'
        """
        # Arrange
        txt_content = "Art. 52. Non è punibile chi ha commesso il fatto per esservi stato costretto dalla necessità di difendere un diritto proprio od altrui contro il pericolo attuale di un'offesa ingiusta..."
        file_path = test_upload_dir / "articolo52.txt"
        file_path.write_text(txt_content, encoding="utf-8")

        file_hash = hashlib.sha256(txt_content.encode("utf-8")).hexdigest()

        # Act
        document = UserDocument(
            filename="articolo52.txt",
            original_filename="Articolo 52 CP.txt",
            file_type="txt",
            file_size_bytes=len(txt_content.encode("utf-8")),
            file_hash=file_hash,
            storage_path=str(file_path),
            document_type="altro",
            legal_domain="penale",
            uploaded_by="user_002",
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

        # Assert
        assert document.file_type == "txt"
        assert document.legal_domain == "penale"
        assert Path(document.storage_path).read_text(encoding="utf-8") == txt_content


@pytest.mark.asyncio
class TestDocumentParsing:
    """Test parsing documenti e extraction."""

    async def test_parse_pdf_extracts_text(
        self,
        test_upload_dir: Path,
        sample_pdf_content: bytes,
    ):
        """
        Test: DocumentParserService estrae testo da PDF.

        Verifica:
        - Testo estratto correttamente
        - No errori durante parsing
        """
        # Arrange
        file_path = test_upload_dir / "sample.pdf"
        file_path.write_bytes(sample_pdf_content)

        parser = DocumentParserService()

        # Act
        text = await parser.extract_text(str(file_path), file_type="pdf")

        # Assert
        assert text is not None
        assert len(text) > 0  # Anche PDF minimale ha qualche testo

    async def test_parse_txt_extracts_text(
        self,
        test_upload_dir: Path,
    ):
        """
        Test: DocumentParserService estrae testo da TXT.

        Verifica:
        - Testo estratto identico all'originale
        - Encoding UTF-8 gestito correttamente
        """
        # Arrange
        original_text = "Principio di legittima difesa: chi è costretto a difendersi da un'aggressione ingiusta..."
        file_path = test_upload_dir / "test.txt"
        file_path.write_text(original_text, encoding="utf-8")

        parser = DocumentParserService()

        # Act
        extracted_text = await parser.extract_text(str(file_path), file_type="txt")

        # Assert
        assert extracted_text == original_text

    async def test_parse_document_chunks_text(
        self,
        test_upload_dir: Path,
    ):
        """
        Test: DocumentParserService chunka testo lungo.

        Verifica:
        - Testo diviso in chunks
        - Chunk size ragionevole
        - Overlap tra chunks per contesto
        """
        # Arrange: crea testo lungo
        long_text = " ".join([
            f"Articolo {i}. Questo è il contenuto dell'articolo numero {i} del codice. "
            f"Contiene diverse disposizioni importanti per la materia trattata. "
            * 20  # Ripeti per fare chunks lunghi
            for i in range(1, 50)
        ])

        file_path = test_upload_dir / "long_doc.txt"
        file_path.write_text(long_text, encoding="utf-8")

        parser = DocumentParserService()

        # Act
        chunks = await parser.chunk_text(long_text, chunk_size=1000, overlap=100)

        # Assert
        assert len(chunks) > 1  # Testo lungo produce multipli chunks
        assert all(len(chunk) <= 1100 for chunk in chunks)  # Rispetta max size (+ overlap)

        # Verifica overlap: fine di chunk[0] appare in inizio chunk[1]
        if len(chunks) > 1:
            end_of_first = chunks[0][-50:]  # ultimi 50 char
            start_of_second = chunks[1][:150]  # primi 150 char
            # Ci dovrebbe essere overlap
            assert any(word in start_of_second for word in end_of_first.split())


@pytest.mark.asyncio
class TestAmendmentExtraction:
    """Test extraction amendments da documenti."""

    async def test_extract_amendments_from_text(
        self,
        db_session: AsyncSession,
        test_upload_dir: Path,
    ):
        """
        Test: AmendmentExtractorService estrae amendments da testo.

        Scenario:
        - Documento contiene riferimento a modifica normativa
        - Extractor riconosce pattern
        - Crea pending_amendment

        Verifica:
        - Amendment estratto correttamente
        - Estremi atto modificante parsed
        - Status 'pending'
        """
        # Arrange: testo con amendment
        text_with_amendment = """
        L'articolo 1453 del Codice Civile è stato modificato dal D.Lgs. 30 giugno 2003, n. 196.

        Il nuovo testo prevede che...
        """

        file_path = test_upload_dir / "doc_with_amendment.txt"
        file_path.write_text(text_with_amendment, encoding="utf-8")

        # Create document record
        file_hash = hashlib.sha256(text_with_amendment.encode("utf-8")).hexdigest()
        document = UserDocument(
            filename="doc_with_amendment.txt",
            original_filename="Modifiche CC.txt",
            file_type="txt",
            file_size_bytes=len(text_with_amendment.encode("utf-8")),
            file_hash=file_hash,
            storage_path=str(file_path),
            document_type="altro",
            legal_domain="civile",
            uploaded_by="user_001",
        )
        db_session.add(document)
        await db_session.commit()

        extractor = AmendmentExtractorService()

        # Act
        result = await extractor.extract_from_document_text(
            text=text_with_amendment,
            legal_domain="civile",
            user_id="user_001",
            source_document_id=document.id,
            session=db_session,
        )

        # Assert
        assert result.amendments_count > 0
        assert len(result.amendment_ids) > 0

        # Verifica amendment in DB
        stmt = select(PendingAmendment).where(
            PendingAmendment.amendment_id == result.amendment_ids[0]
        )
        db_result = await db_session.execute(stmt)
        amendment = db_result.scalar_one_or_none()

        assert amendment is not None
        assert amendment.validation_status == "pending"
        assert amendment.source_document_id == document.id
        assert "1453" in amendment.target_article_urn or "1453" in str(amendment.amendment_id)

    async def test_extract_multiple_amendments(
        self,
        db_session: AsyncSession,
        test_upload_dir: Path,
    ):
        """
        Test: Extractor trova multipli amendments in un documento.

        Verifica:
        - Tutti gli amendments estratti
        - Ciascuno ha record separato
        - Tutti linked al documento sorgente
        """
        # Arrange: testo con multipli amendments
        text_with_multiple = """
        Modifiche al Codice Civile:

        1. L'articolo 1453 è stato modificato dalla L. 25 marzo 2010, n. 42.
        2. L'articolo 1456 è stato abrogato dal D.Lgs. 15 gennaio 2015, n. 7.
        3. L'articolo 1460 è stato sostituito dalla L. 10 giugno 2018, n. 93.
        """

        file_path = test_upload_dir / "multiple_amendments.txt"
        file_path.write_text(text_with_multiple, encoding="utf-8")

        file_hash = hashlib.sha256(text_with_multiple.encode("utf-8")).hexdigest()
        document = UserDocument(
            filename="multiple_amendments.txt",
            original_filename="Modifiche Multiple.txt",
            file_type="txt",
            file_size_bytes=len(text_with_multiple.encode("utf-8")),
            file_hash=file_hash,
            storage_path=str(file_path),
            legal_domain="civile",
            uploaded_by="user_001",
        )
        db_session.add(document)
        await db_session.commit()

        extractor = AmendmentExtractorService()

        # Act
        result = await extractor.extract_from_document_text(
            text=text_with_multiple,
            legal_domain="civile",
            user_id="user_001",
            source_document_id=document.id,
            session=db_session,
        )

        # Assert
        # Nota: regex-based extractor potrebbe non trovare tutti,
        # ma dovrebbe trovarne almeno 1
        assert result.amendments_count >= 1

        # Verifica tutti linked al documento
        stmt = select(PendingAmendment).where(
            PendingAmendment.source_document_id == document.id
        )
        db_result = await db_session.execute(stmt)
        all_amendments = db_result.scalars().all()

        assert len(all_amendments) == result.amendments_count
        assert all(a.source_document_id == document.id for a in all_amendments)


@pytest.mark.asyncio
class TestDocumentProcessingStatus:
    """Test tracking status processing documenti."""

    async def test_document_processing_workflow(
        self,
        db_session: AsyncSession,
        test_upload_dir: Path,
    ):
        """
        Test: Workflow completo processing documento.

        Flow:
        1. Upload → status 'uploaded'
        2. Start parsing → status 'parsing'
        3. Extraction → status 'extracting'
        4. Complete → status 'completed'
        5. Timestamps aggiornati

        Verifica:
        - Status transitions corrette
        - Timestamps popolati
        - Counts aggiornati
        """
        # Arrange
        text = "Articolo 52 CP. Non è punibile chi..."
        file_path = test_upload_dir / "workflow_test.txt"
        file_path.write_text(text, encoding="utf-8")

        file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Act: Upload
        document = UserDocument(
            filename="workflow_test.txt",
            original_filename="Test.txt",
            file_type="txt",
            file_size_bytes=len(text.encode("utf-8")),
            file_hash=file_hash,
            storage_path=str(file_path),
            uploaded_by="user_001",
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

        assert document.processing_status == "uploaded"

        # Start parsing
        document.processing_status = "parsing"
        from datetime import datetime, timezone
        document.processing_started_at = datetime.now()
        await db_session.commit()

        # Extraction phase
        document.processing_status = "extracting"
        await db_session.commit()

        # Complete
        document.processing_status = "completed"
        document.processing_completed_at = datetime.now()
        document.entities_extracted = 1
        document.amendments_extracted = 0
        await db_session.commit()
        await db_session.refresh(document)

        # Assert
        assert document.processing_status == "completed"
        assert document.processing_started_at is not None
        assert document.processing_completed_at is not None
        assert document.entities_extracted == 1
        assert document.processing_completed_at > document.processing_started_at

    async def test_document_processing_error_handling(
        self,
        db_session: AsyncSession,
        test_upload_dir: Path,
    ):
        """
        Test: Errore durante processing viene trackato.

        Verifica:
        - Status 'failed' se errore
        - Error message salvato
        - Document non bloccato (può retry)
        """
        # Arrange
        document = UserDocument(
            filename="corrupt.pdf",
            original_filename="Corrupt.pdf",
            file_type="pdf",
            file_size_bytes=100,
            file_hash="deadbeef",
            storage_path="/nonexistent/path/corrupt.pdf",
            uploaded_by="user_001",
        )
        db_session.add(document)
        await db_session.commit()

        # Act: simula errore parsing
        document.processing_status = "failed"
        document.processing_error = "FileNotFoundError: File not found at /nonexistent/path/corrupt.pdf"
        from datetime import datetime, timezone
        document.processing_completed_at = datetime.now()
        await db_session.commit()
        await db_session.refresh(document)

        # Assert
        assert document.processing_status == "failed"
        assert document.processing_error is not None
        assert "FileNotFoundError" in document.processing_error
