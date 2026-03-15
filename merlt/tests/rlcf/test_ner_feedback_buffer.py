"""Tests for NERFeedbackBuffer and NERFeedback."""
import pytest
from datetime import timezone, datetime

from merlt.rlcf.ner_feedback_buffer import (
    NERFeedback,
    NERFeedbackBuffer,
    UserNERStats,
    get_ner_feedback_buffer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buffer(threshold=5) -> NERFeedbackBuffer:
    return NERFeedbackBuffer(training_threshold=threshold)


async def _add(
    buf: NERFeedbackBuffer,
    user_id="u1",
    feedback_type="confirmation",
    article_urn="urn:nir:codice.civile:art:1218",
    selected_text="art. 1218 c.c.",
    context="Ai sensi dell'art. 1218 c.c. il debitore è responsabile.",
):
    return await buf.add_feedback(
        article_urn=article_urn,
        user_id=user_id,
        selected_text=selected_text,
        start_offset=0,
        end_offset=len(selected_text),
        context_window=context,
        feedback_type=feedback_type,
        correct_reference={"tipo_atto": "codice civile", "articoli": ["1218"]},
    )


# ---------------------------------------------------------------------------
# NERFeedback.to_spacy_format
# ---------------------------------------------------------------------------


class TestNERFeedbackToSpacyFormat:
    def test_entity_found_in_context(self):
        fb = NERFeedback(
            feedback_id="fid",
            article_urn="urn:x",
            user_id="u1",
            selected_text="art. 1218",
            start_offset=0,
            end_offset=9,
            context_window="Vedi art. 1218 del codice.",
            feedback_type="confirmation",
            original_parsed=None,
            correct_reference={},
            confidence_before=None,
            source="test",
        )
        text, annots = fb.to_spacy_format()

        assert text == "Vedi art. 1218 del codice."
        entities = annots["entities"]
        assert len(entities) == 1
        start, end, label = entities[0]
        assert label == "NORMA"
        assert text[start:end] == "art. 1218"

    def test_entity_not_in_context_uses_fallback(self):
        fb = NERFeedback(
            feedback_id="fid2",
            article_urn="urn:x",
            user_id="u1",
            selected_text="art. 9999",
            start_offset=0,
            end_offset=9,
            context_window="Nessuna corrispondenza qui.",
            feedback_type="correction",
            original_parsed=None,
            correct_reference={},
            confidence_before=None,
            source="test",
        )
        text, annots = fb.to_spacy_format()

        entities = annots["entities"]
        assert len(entities) == 1
        start, end, label = entities[0]
        assert start == 0
        assert end == len("art. 9999")
        assert label == "NORMA"

    def test_to_dict_contains_created_at_isoformat(self):
        fb = NERFeedback(
            feedback_id="fid3",
            article_urn="urn:x",
            user_id="u1",
            selected_text="art. 1",
            start_offset=0,
            end_offset=6,
            context_window="art. 1 c.c.",
            feedback_type="annotation",
            original_parsed=None,
            correct_reference={},
            confidence_before=None,
            source="test",
        )
        d = fb.to_dict()
        assert isinstance(d["created_at"], str)
        # should be parseable
        datetime.fromisoformat(d["created_at"])


# ---------------------------------------------------------------------------
# NERFeedbackBuffer – authority calculation
# ---------------------------------------------------------------------------


class TestCalculateUserAuthority:
    def test_base_authority_new_user(self):
        buf = _make_buffer()
        auth = buf._calculate_user_authority("new_user", "confirmation")
        assert auth == pytest.approx(0.3, abs=0.01)

    def test_correction_bonus_applied(self):
        buf = _make_buffer()
        auth_correction = buf._calculate_user_authority("u1", "correction")
        auth_confirmation = buf._calculate_user_authority("u1", "confirmation")
        assert auth_correction > auth_confirmation

    def test_feedback_volume_increases_authority(self):
        buf = _make_buffer()
        # Simulate 10 prior feedbacks
        buf._user_stats["u_vol"] = UserNERStats(user_id="u_vol", total_feedback=10)
        auth = buf._calculate_user_authority("u_vol", "confirmation")
        # 0.3 base + 0.1 from 10 feedbacks
        assert auth >= 0.4

    def test_authority_capped_at_max(self):
        buf = _make_buffer()
        buf._user_stats["u_max"] = UserNERStats(
            user_id="u_max",
            total_feedback=1000,
            validated_correct=100,
            validated_incorrect=0,
        )
        auth = buf._calculate_user_authority("u_max", "correction")
        assert auth <= buf.MAX_AUTHORITY

    def test_authority_never_below_min(self):
        buf = _make_buffer()
        auth = buf._calculate_user_authority("brand_new", "confirmation")
        assert auth >= buf.MIN_AUTHORITY


# ---------------------------------------------------------------------------
# NERFeedbackBuffer – authority_to_sample_weight
# ---------------------------------------------------------------------------


class TestAuthorityToSampleWeight:
    def test_min_authority_maps_to_low_weight(self):
        buf = _make_buffer()
        w = buf._authority_to_sample_weight(0.1)
        assert w == pytest.approx(0.5, abs=0.01)

    def test_max_authority_maps_to_high_weight(self):
        buf = _make_buffer()
        w = buf._authority_to_sample_weight(1.0)
        assert w == pytest.approx(2.0, abs=0.01)

    def test_midpoint(self):
        buf = _make_buffer()
        w = buf._authority_to_sample_weight(0.55)
        # Linear: 0.5 + (0.55 - 0.1) * (1.5 / 0.9)
        expected = 0.5 + (0.55 - 0.1) * (1.5 / 0.9)
        assert w == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# NERFeedbackBuffer – add_feedback
# ---------------------------------------------------------------------------


class TestAddFeedback:
    @pytest.mark.asyncio
    async def test_returns_feedback_id_string(self):
        buf = _make_buffer()
        fid = await _add(buf)
        assert isinstance(fid, str)
        assert len(fid) > 0

    @pytest.mark.asyncio
    async def test_buffer_grows(self):
        buf = _make_buffer()
        await _add(buf, user_id="u1")
        await _add(buf, user_id="u2")
        # get_all() is the sync version (overrides the async one)
        items = buf.get_all()
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_user_stats_updated(self):
        buf = _make_buffer()
        await _add(buf, user_id="u_stats", feedback_type="correction")
        assert buf._user_stats["u_stats"].total_feedback == 1
        assert buf._user_stats["u_stats"].corrections == 1

    @pytest.mark.asyncio
    async def test_confirmation_increments_confirmations(self):
        buf = _make_buffer()
        await _add(buf, user_id="u_conf", feedback_type="confirmation")
        assert buf._user_stats["u_conf"].confirmations == 1

    @pytest.mark.asyncio
    async def test_annotation_increments_annotations(self):
        buf = _make_buffer()
        await _add(buf, user_id="u_ann", feedback_type="annotation")
        assert buf._user_stats["u_ann"].annotations == 1

    @pytest.mark.asyncio
    async def test_authority_override_used(self):
        buf = _make_buffer()
        await buf.add_feedback(
            article_urn="urn:x",
            user_id="u_override",
            selected_text="art. 1",
            start_offset=0,
            end_offset=6,
            context_window="art. 1 del cc",
            feedback_type="correction",
            correct_reference={},
            user_authority_override=0.95,
        )
        items = buf.get_all()
        assert items[0].user_authority == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_sample_weight_computed_from_authority(self):
        buf = _make_buffer()
        await buf.add_feedback(
            article_urn="urn:x",
            user_id="u_sw",
            selected_text="art. 1",
            start_offset=0,
            end_offset=6,
            context_window="art. 1",
            feedback_type="confirmation",
            correct_reference={},
            user_authority_override=0.1,
        )
        items = buf.get_all()
        assert items[0].sample_weight == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# NERFeedbackBuffer – should_train / has_data
# ---------------------------------------------------------------------------


class TestShouldTrain:
    @pytest.mark.asyncio
    async def test_false_when_below_threshold(self):
        buf = _make_buffer(threshold=10)
        for _ in range(5):
            await _add(buf)
        assert buf.should_train() is False

    @pytest.mark.asyncio
    async def test_true_when_at_threshold(self):
        buf = _make_buffer(threshold=3)
        for _ in range(3):
            await _add(buf)
        assert buf.should_train() is True

    @pytest.mark.asyncio
    async def test_has_data_false_when_empty(self):
        buf = _make_buffer()
        assert buf.has_data() is False

    @pytest.mark.asyncio
    async def test_has_data_true_after_add(self):
        buf = _make_buffer()
        await _add(buf)
        assert buf.has_data() is True


# ---------------------------------------------------------------------------
# NERFeedbackBuffer – get_buffer_stats
# ---------------------------------------------------------------------------


class TestGetBufferStats:
    @pytest.mark.asyncio
    async def test_empty_buffer(self):
        buf = _make_buffer(threshold=50)
        stats = await buf.get_buffer_stats()
        assert stats["size"] == 0
        assert stats["training_ready"] is False
        assert stats["training_threshold"] == 50
        assert stats["feedback_types"] == {}

    @pytest.mark.asyncio
    async def test_stats_after_adds(self):
        buf = _make_buffer(threshold=10)
        await _add(buf, feedback_type="correction")
        await _add(buf, feedback_type="confirmation")
        await _add(buf, feedback_type="correction")

        stats = await buf.get_buffer_stats()
        assert stats["size"] == 3
        assert stats["feedback_types"]["correction"] == 2
        assert stats["feedback_types"]["confirmation"] == 1
        assert stats["oldest_feedback"] is not None
        assert stats["newest_feedback"] is not None


# ---------------------------------------------------------------------------
# NERFeedbackBuffer – export_for_spacy / export_for_spacy_weighted
# ---------------------------------------------------------------------------


class TestExportForSpacy:
    @pytest.mark.asyncio
    async def test_export_returns_list_of_tuples(self):
        buf = _make_buffer()
        await _add(buf)
        data = await buf.export_for_spacy()
        assert len(data) == 1
        text, annots = data[0]
        assert isinstance(text, str)
        assert "entities" in annots

    @pytest.mark.asyncio
    async def test_export_weighted_includes_weight(self):
        buf = _make_buffer()
        await _add(buf)
        data = await buf.export_for_spacy_weighted()
        assert len(data) == 1
        text, annots, weight = data[0]
        assert isinstance(weight, float)
        assert weight >= 0.5

    @pytest.mark.asyncio
    async def test_empty_export(self):
        buf = _make_buffer()
        assert await buf.export_for_spacy() == []
        assert await buf.export_for_spacy_weighted() == []


# ---------------------------------------------------------------------------
# NERFeedbackBuffer – get_authority_stats
# ---------------------------------------------------------------------------


class TestGetAuthorityStats:
    @pytest.mark.asyncio
    async def test_empty(self):
        buf = _make_buffer()
        stats = await buf.get_authority_stats()
        assert stats["total_users"] == 0
        assert stats["avg_authority"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_after_feedback(self):
        buf = _make_buffer()
        await buf.add_feedback(
            article_urn="urn:x",
            user_id="u_a",
            selected_text="art. 1",
            start_offset=0,
            end_offset=6,
            context_window="art. 1",
            feedback_type="confirmation",
            correct_reference={},
            user_authority_override=0.8,
        )
        await buf.add_feedback(
            article_urn="urn:x",
            user_id="u_b",
            selected_text="art. 2",
            start_offset=0,
            end_offset=6,
            context_window="art. 2",
            feedback_type="correction",
            correct_reference={},
            user_authority_override=0.4,
        )

        stats = await buf.get_authority_stats()
        assert stats["total_users"] == 2
        assert stats["avg_authority"] == pytest.approx(0.6, abs=0.01)
        top = stats["top_contributors"]
        assert len(top) <= 5

    @pytest.mark.asyncio
    async def test_distribution_buckets(self):
        buf = _make_buffer()
        await buf.add_feedback(
            article_urn="urn:x",
            user_id="u_expert",
            selected_text="x",
            start_offset=0,
            end_offset=1,
            context_window="x y z",
            feedback_type="confirmation",
            correct_reference={},
            user_authority_override=0.9,
        )
        stats = await buf.get_authority_stats()
        dist = stats["authority_distribution"]
        assert dist["expert"] == 1
        assert dist["low"] == 0


# ---------------------------------------------------------------------------
# NERFeedbackBuffer – export_to_dict / clear / remove_feedback
# ---------------------------------------------------------------------------


class TestBufferMutations:
    @pytest.mark.asyncio
    async def test_export_to_dict(self):
        buf = _make_buffer()
        await _add(buf)
        exported = await buf.export_to_dict()
        assert len(exported) == 1
        assert "feedback_id" in exported[0]
        assert "created_at" in exported[0]

    @pytest.mark.asyncio
    async def test_clear_removes_all(self):
        buf = _make_buffer()
        await _add(buf)
        await _add(buf)
        removed = await buf.clear()
        assert removed == 2
        assert buf.has_data() is False

    @pytest.mark.asyncio
    async def test_remove_feedback_existing(self):
        buf = _make_buffer()
        fid = await _add(buf)
        removed = await buf.remove_feedback(fid)
        assert removed is True
        assert buf.has_data() is False

    @pytest.mark.asyncio
    async def test_remove_feedback_not_found(self):
        buf = _make_buffer()
        removed = await buf.remove_feedback("nonexistent-id")
        assert removed is False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestGetNERFeedbackBuffer:
    def test_returns_same_instance(self):
        import merlt.rlcf.ner_feedback_buffer as mod
        mod._global_buffer = None
        b1 = get_ner_feedback_buffer()
        b2 = get_ner_feedback_buffer()
        assert b1 is b2
        mod._global_buffer = None

    def test_default_threshold_is_50(self):
        import merlt.rlcf.ner_feedback_buffer as mod
        mod._global_buffer = None
        buf = get_ner_feedback_buffer()
        assert buf._training_threshold == 50
        mod._global_buffer = None
