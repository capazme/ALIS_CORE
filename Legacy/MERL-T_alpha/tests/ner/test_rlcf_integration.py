"""
Test RLCF Authority Integration in NER Pipeline
================================================

Tests per verificare che il sistema RLCF authority sia
correttamente integrato nella pipeline NER.
"""

import asyncio
import pytest
from merlt.rlcf.ner_feedback_buffer import NERFeedbackBuffer, NERFeedback


class TestRLCFAuthorityIntegration:
    """Test suite per RLCF Authority Integration."""

    def test_buffer_initialization(self):
        """Verifica inizializzazione buffer con threshold."""
        buffer = NERFeedbackBuffer(training_threshold=10)
        assert not buffer.has_data()
        assert not buffer.should_train()

    @pytest.mark.asyncio
    async def test_add_feedback_calculates_authority(self):
        """Verifica che add_feedback calcoli l'authority correttamente."""
        buffer = NERFeedbackBuffer(training_threshold=50)

        # Primo feedback - nuovo utente, dovrebbe avere BASE_AUTHORITY
        feedback_id = await buffer.add_feedback(
            article_urn="urn:nir:stato:codice.civile:art1218",
            user_id="user_001",
            selected_text="art. 1218 c.c.",
            start_offset=0,
            end_offset=14,
            context_window="L'art. 1218 c.c. disciplina...",
            feedback_type="correction",
            correct_reference={
                "tipo_atto": "codice civile",
                "articoli": ["1218"],
            },
            source="citation_preview",
        )

        assert feedback_id is not None
        assert buffer.has_data()

        # Verifica feedback salvato con authority
        feedbacks = buffer.get_all()
        assert len(feedbacks) == 1
        fb = feedbacks[0]

        # BASE_AUTHORITY = 0.3 + CORRECTION_BONUS = 0.1 = 0.4
        assert fb.user_authority == pytest.approx(0.4, rel=0.01)
        # Weight mapping: 0.4 authority -> ~1.0 weight
        assert 0.5 <= fb.sample_weight <= 2.0

    @pytest.mark.asyncio
    async def test_authority_increases_with_feedback_volume(self):
        """Verifica che l'authority aumenti con il volume di feedback."""
        buffer = NERFeedbackBuffer(training_threshold=50)

        # Aggiungi 15 feedback dallo stesso utente
        for i in range(15):
            await buffer.add_feedback(
                article_urn=f"urn:test:art{i}",
                user_id="expert_user",
                selected_text=f"art. {i}",
                start_offset=0,
                end_offset=6,
                context_window=f"L'art. {i} prevede...",
                feedback_type="confirmation",
                correct_reference={"tipo_atto": "legge", "articoli": [str(i)]},
                source="citation_preview",
            )

        feedbacks = buffer.get_all()
        # L'ultimo feedback dovrebbe avere authority più alta
        # perché ha più feedback volume
        first_authority = feedbacks[0].user_authority
        last_authority = feedbacks[-1].user_authority

        assert last_authority > first_authority

    @pytest.mark.asyncio
    async def test_correction_bonus(self):
        """Verifica che le correction abbiano bonus rispetto alle confirmation."""
        buffer = NERFeedbackBuffer(training_threshold=50)

        # Aggiungi confirmation
        await buffer.add_feedback(
            article_urn="urn:test:art1",
            user_id="test_user",
            selected_text="art. 1",
            start_offset=0,
            end_offset=6,
            context_window="L'art. 1 prevede...",
            feedback_type="confirmation",
            correct_reference={"tipo_atto": "legge", "articoli": ["1"]},
            source="citation_preview",
        )

        # Aggiungi correction da utente diverso (stessa authority base)
        await buffer.add_feedback(
            article_urn="urn:test:art2",
            user_id="test_user_2",
            selected_text="art. 2",
            start_offset=0,
            end_offset=6,
            context_window="L'art. 2 prevede...",
            feedback_type="correction",
            correct_reference={"tipo_atto": "legge", "articoli": ["2"]},
            source="citation_preview",
        )

        feedbacks = buffer.get_all()
        confirmation_fb = feedbacks[0]
        correction_fb = feedbacks[1]

        # Correction should have higher authority
        assert correction_fb.user_authority > confirmation_fb.user_authority

    @pytest.mark.asyncio
    async def test_user_authority_override(self):
        """Verifica che user_authority_override funzioni."""
        buffer = NERFeedbackBuffer(training_threshold=50)

        # Passa authority esterna (es. da AuthorityModule)
        await buffer.add_feedback(
            article_urn="urn:test:art1",
            user_id="expert",
            selected_text="art. 1",
            start_offset=0,
            end_offset=6,
            context_window="L'art. 1 prevede...",
            feedback_type="correction",
            correct_reference={"tipo_atto": "legge", "articoli": ["1"]},
            source="citation_preview",
            user_authority_override=0.95,  # Expert from AuthorityModule
        )

        feedbacks = buffer.get_all()
        fb = feedbacks[0]

        assert fb.user_authority == 0.95
        # High authority -> high weight
        assert fb.sample_weight > 1.5

    @pytest.mark.asyncio
    async def test_export_for_spacy_weighted(self):
        """Verifica export weighted per spaCy."""
        buffer = NERFeedbackBuffer(training_threshold=50)

        # Aggiungi feedback con diverse authority
        await buffer.add_feedback(
            article_urn="urn:test:art1",
            user_id="novice",
            selected_text="art. 1 c.c.",
            start_offset=0,
            end_offset=11,
            context_window="L'art. 1 c.c. stabilisce...",
            feedback_type="confirmation",
            correct_reference={"tipo_atto": "codice civile", "articoli": ["1"]},
            source="citation_preview",
            user_authority_override=0.2,
        )

        await buffer.add_feedback(
            article_urn="urn:test:art2",
            user_id="expert",
            selected_text="artt. 3 e 4",
            start_offset=0,
            end_offset=11,
            context_window="Gli artt. 3 e 4 del D.Lgs. 50/2016...",
            feedback_type="correction",
            correct_reference={
                "tipo_atto": "decreto legislativo",
                "numero_atto": "50",
                "anno": "2016",
                "articoli": ["3", "4"],
            },
            source="citation_preview",
            user_authority_override=0.9,
        )

        # Export weighted
        weighted_data = await buffer.export_for_spacy_weighted()

        assert len(weighted_data) == 2

        # Verify format: (text, annotations, weight)
        for text, annotations, weight in weighted_data:
            assert isinstance(text, str)
            assert "entities" in annotations
            assert isinstance(weight, float)
            assert 0.5 <= weight <= 2.0

        # Expert should have higher weight
        novice_weight = weighted_data[0][2]
        expert_weight = weighted_data[1][2]
        assert expert_weight > novice_weight

    @pytest.mark.asyncio
    async def test_get_authority_stats(self):
        """Verifica statistiche authority."""
        buffer = NERFeedbackBuffer(training_threshold=50)

        # Aggiungi feedback da utenti diversi
        users = [
            ("low_auth", 0.15),
            ("medium_auth", 0.45),
            ("high_auth", 0.65),
            ("expert_auth", 0.85),
        ]

        for user_id, auth in users:
            await buffer.add_feedback(
                article_urn=f"urn:test:{user_id}",
                user_id=user_id,
                selected_text="art. 1",
                start_offset=0,
                end_offset=6,
                context_window="L'art. 1...",
                feedback_type="correction",
                correct_reference={"tipo_atto": "legge", "articoli": ["1"]},
                source="citation_preview",
                user_authority_override=auth,
            )

        stats = await buffer.get_authority_stats()

        assert stats["total_users"] == 4
        assert "authority_distribution" in stats
        assert "top_contributors" in stats

        # Verify distribution
        dist = stats["authority_distribution"]
        assert dist["low"] >= 1
        assert dist["expert"] >= 1

    @pytest.mark.asyncio
    async def test_buffer_stats_include_authority_info(self):
        """Verifica che buffer stats includano info authority."""
        buffer = NERFeedbackBuffer(training_threshold=50)

        await buffer.add_feedback(
            article_urn="urn:test:art1",
            user_id="user1",
            selected_text="art. 1",
            start_offset=0,
            end_offset=6,
            context_window="L'art. 1...",
            feedback_type="correction",
            correct_reference={"tipo_atto": "legge", "articoli": ["1"]},
            source="citation_preview",
        )

        stats = await buffer.get_buffer_stats()

        assert stats["size"] == 1
        assert not stats["training_ready"]  # threshold is 50
        assert "feedback_types" in stats
        assert stats["feedback_types"]["correction"] == 1

    def test_authority_to_sample_weight_mapping(self):
        """Verifica il mapping authority -> sample_weight."""
        buffer = NERFeedbackBuffer()

        # Test edge cases
        # Min authority (0.1) -> min weight (0.5)
        assert buffer._authority_to_sample_weight(0.1) == pytest.approx(0.5, rel=0.01)

        # Max authority (1.0) -> max weight (2.0)
        assert buffer._authority_to_sample_weight(1.0) == pytest.approx(2.0, rel=0.01)

        # Middle authority (0.55) -> middle weight (~1.25)
        mid_weight = buffer._authority_to_sample_weight(0.55)
        assert 1.0 < mid_weight < 1.5

    def test_calculate_user_authority_formula(self):
        """Verifica la formula di calcolo authority."""
        buffer = NERFeedbackBuffer()

        # New user, confirmation
        auth = buffer._calculate_user_authority("new_user", "confirmation")
        # BASE = 0.3, no correction bonus
        assert auth == pytest.approx(0.3, rel=0.01)

        # New user, correction
        auth = buffer._calculate_user_authority("new_user_2", "correction")
        # BASE = 0.3 + CORRECTION_BONUS = 0.1 = 0.4
        assert auth == pytest.approx(0.4, rel=0.01)


class TestNERTrainerWeighted:
    """Test per NERTrainer con weighted training."""

    def test_weighted_sample_uniform(self):
        """Verifica weighted sampling con pesi uniformi."""
        from merlt.ner.training import NERTrainer

        # Mock objects
        class MockModel:
            nlp = None

        class MockBuffer:
            def has_data(self):
                return False

            def get_all(self):
                return []

        trainer = NERTrainer(MockModel(), MockBuffer())

        # Test con pesi uniformi
        examples = [(f"example_{i}", 1.0) for i in range(10)]
        sampled = trainer._weighted_sample(examples, 5)

        assert len(sampled) == 5
        # Tutti gli elementi dovrebbero essere stringhe
        assert all(isinstance(s, str) for s in sampled)

    def test_weighted_sample_biased(self):
        """Verifica che weighted sampling sia biased verso pesi alti."""
        from merlt.ner.training import NERTrainer

        class MockModel:
            nlp = None

        class MockBuffer:
            def has_data(self):
                return False

            def get_all(self):
                return []

        trainer = NERTrainer(MockModel(), MockBuffer())

        # 1 elemento con peso alto, 9 con peso basso
        examples = [("high_weight", 10.0)] + [(f"low_{i}", 0.1) for i in range(9)]

        # Campiona molte volte per verificare bias
        high_count = 0
        n_trials = 100

        for _ in range(n_trials):
            sampled = trainer._weighted_sample(examples, 3)
            if "high_weight" in sampled:
                high_count += 1

        # L'elemento ad alto peso dovrebbe apparire nella maggior parte dei campioni
        assert high_count > n_trials * 0.5  # Almeno 50% delle volte
