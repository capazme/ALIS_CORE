"""Tests for NERRLCFIntegration."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from merlt.rlcf.ner_rlcf_integration import (
    NERRLCFIntegration,
    NERFeedbackRLCFResult,
    NERFeedbackHistoryItem,
    get_ner_rlcf_integration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_integration():
    """Create NERRLCFIntegration with all external deps mocked."""
    authority_svc = MagicMock()
    authority_svc.calculate_authority_delta = MagicMock(return_value=0.05)
    authority_svc.sync_user = AsyncMock(return_value=(0.7, None))

    buffer = AsyncMock()
    buffer.add_feedback = AsyncMock()
    buffer.get_buffer_stats = AsyncMock(return_value={
        "size": 3,
        "training_threshold": 50,
        "training_ready": False,
    })
    buffer.should_train = MagicMock(return_value=False)
    buffer.get_authority_stats = AsyncMock(return_value={
        "top_contributors": [],
        "total_users": 0,
        "avg_authority": 0.0,
        "authority_distribution": {},
    })

    with patch(
        "merlt.rlcf.ner_rlcf_integration.get_ner_feedback_buffer",
        return_value=buffer,
    ):
        integ = NERRLCFIntegration(authority_service=authority_svc)

    integ.authority_service = authority_svc
    integ.buffer = buffer
    return integ


# ---------------------------------------------------------------------------
# NERFeedbackHistoryItem
# ---------------------------------------------------------------------------


class TestNERFeedbackHistoryItem:
    def test_to_dict_serializes_created_at(self):
        item = NERFeedbackHistoryItem(
            feedback_id="fid1",
            article_urn="urn:x",
            selected_text="art. 1",
            feedback_type="confirmation",
            correct_reference={"tipo_atto": "cc"},
            user_authority=0.6,
            created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        d = item.to_dict()
        assert isinstance(d["created_at"], str)
        assert "2025-01-15" in d["created_at"]
        assert d["feedback_id"] == "fid1"
        assert d["user_authority"] == 0.6


# ---------------------------------------------------------------------------
# NERRLCFIntegration – _extract_domain
# ---------------------------------------------------------------------------


class TestExtractDomain:
    @pytest.fixture
    def integ(self):
        return _make_integration()

    def test_codice_civile(self, integ):
        assert integ._extract_domain("urn:nir:stato:codice.civile:1942:art:1218") == "civile"

    def test_codice_penale(self, integ):
        assert integ._extract_domain("urn:nir:stato:codice.penale:1930:art:575") == "penale"

    def test_procedura_civile(self, integ):
        assert integ._extract_domain("urn:nir:procedura.civile:art:138") == "procedura_civile"

    def test_procedura_penale(self, integ):
        assert integ._extract_domain("urn:nir:procedura.penale:art:530") == "procedura_penale"

    def test_costituzione(self, integ):
        assert integ._extract_domain("urn:nir:costituzione:art:32") == "costituzionale"

    def test_amministrativo(self, integ):
        assert integ._extract_domain("urn:nir:codice.amministrativo:art:5") == "amministrativo"

    def test_unknown_returns_none(self, integ):
        assert integ._extract_domain("urn:nir:sconosciuto:art:1") is None


# ---------------------------------------------------------------------------
# NERRLCFIntegration – _get_user_authority
# ---------------------------------------------------------------------------


class TestGetUserAuthority:
    @pytest.mark.asyncio
    async def test_with_qualification_calls_authority_service(self):
        integ = _make_integration()
        from merlt.rlcf.authority_sync import AuthorityBreakdown
        mock_breakdown = MagicMock(spec=AuthorityBreakdown)
        integ.authority_service.sync_user = AsyncMock(return_value=(0.8, mock_breakdown))

        authority, breakdown = await integ._get_user_authority(
            user_id="u1",
            qualification="avvocato",
            years_experience=10,
            total_feedback=50,
        )

        assert authority == 0.8
        assert breakdown is mock_breakdown
        integ.authority_service.sync_user.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_without_qualification_returns_default(self):
        integ = _make_integration()
        # Buffer returns empty top_contributors
        integ.buffer.get_authority_stats = AsyncMock(return_value={
            "top_contributors": [],
        })

        authority, breakdown = await integ._get_user_authority(
            user_id="u_anonymous",
            qualification=None,
        )

        assert authority == 0.5
        assert breakdown is None

    @pytest.mark.asyncio
    async def test_without_qualification_finds_user_in_contributors(self):
        integ = _make_integration()
        integ.buffer.get_authority_stats = AsyncMock(return_value={
            "top_contributors": [
                {"user_id": "u_known", "avg_authority": 0.72, "feedback_count": 20},
            ]
        })

        authority, breakdown = await integ._get_user_authority(
            user_id="u_known",
        )

        assert authority == 0.72
        assert breakdown is None


# ---------------------------------------------------------------------------
# NERRLCFIntegration – process_ner_feedback (happy path)
# ---------------------------------------------------------------------------


class TestProcessNERFeedback:
    @pytest.mark.asyncio
    async def test_success_returns_result_with_success_true(self):
        integ = _make_integration()

        with (
            patch.object(integ, "_get_user_authority", AsyncMock(return_value=(0.6, None))),
            patch.object(integ, "_persist_to_rlcf_db", AsyncMock(return_value=True)),
            patch.object(integ, "_update_user_track_record", AsyncMock(return_value=True)),
        ):
            result = await integ.process_ner_feedback(
                user_id="u1",
                article_urn="urn:nir:codice.civile:art:1218",
                selected_text="art. 1218 c.c.",
                context_window="Ai sensi dell'art. 1218 c.c.",
                feedback_type="confirmation",
                correct_reference={"tipo_atto": "codice civile", "articoli": ["1218"]},
            )

        assert result.success is True
        assert result.user_authority == 0.6
        assert result.persisted_to_db is True
        assert result.track_record_updated is True
        assert isinstance(result.feedback_id, str)
        assert len(result.feedback_id) > 0

    @pytest.mark.asyncio
    async def test_buffer_called_with_correct_args(self):
        integ = _make_integration()

        with (
            patch.object(integ, "_get_user_authority", AsyncMock(return_value=(0.5, None))),
            patch.object(integ, "_persist_to_rlcf_db", AsyncMock(return_value=True)),
            patch.object(integ, "_update_user_track_record", AsyncMock(return_value=True)),
        ):
            await integ.process_ner_feedback(
                user_id="u_test",
                article_urn="urn:nir:codice.penale:art:575",
                selected_text="art. 575",
                context_window="contestazione art. 575",
                feedback_type="correction",
                correct_reference={"tipo_atto": "cp"},
            )

        integ.buffer.add_feedback.assert_awaited_once()
        call_kwargs = integ.buffer.add_feedback.call_args.kwargs
        assert call_kwargs["user_id"] == "u_test"
        assert call_kwargs["feedback_type"] == "correction"
        assert call_kwargs["user_authority_override"] == 0.5

    @pytest.mark.asyncio
    async def test_training_ready_reflected_in_result(self):
        integ = _make_integration()
        integ.buffer.should_train = MagicMock(return_value=True)
        integ.buffer.get_buffer_stats = AsyncMock(return_value={
            "size": 50,
            "training_threshold": 50,
            "training_ready": True,
        })

        with (
            patch.object(integ, "_get_user_authority", AsyncMock(return_value=(0.5, None))),
            patch.object(integ, "_persist_to_rlcf_db", AsyncMock(return_value=True)),
            patch.object(integ, "_update_user_track_record", AsyncMock(return_value=True)),
        ):
            result = await integ.process_ner_feedback(
                user_id="u1",
                article_urn="urn:nir:cc:art:1",
                selected_text="art. 1",
                context_window="context",
                feedback_type="confirmation",
                correct_reference={},
            )

        assert result.training_ready is True
        assert result.buffer_size == 50

    @pytest.mark.asyncio
    async def test_exception_returns_failure_result(self):
        integ = _make_integration()

        with patch.object(
            integ,
            "_get_user_authority",
            AsyncMock(side_effect=RuntimeError("db failure")),
        ):
            result = await integ.process_ner_feedback(
                user_id="u_err",
                article_urn="urn:x",
                selected_text="x",
                context_window="x",
                feedback_type="confirmation",
                correct_reference={},
            )

        assert result.success is False
        assert result.user_authority == 0.5
        assert result.persisted_to_db is False
        assert "Errore" in result.message

    @pytest.mark.asyncio
    async def test_authority_breakdown_serialized_when_present(self):
        integ = _make_integration()
        breakdown = MagicMock()
        breakdown.to_dict = MagicMock(return_value={"component_a": 0.5})

        with (
            patch.object(integ, "_get_user_authority", AsyncMock(return_value=(0.7, breakdown))),
            patch.object(integ, "_persist_to_rlcf_db", AsyncMock(return_value=True)),
            patch.object(integ, "_update_user_track_record", AsyncMock(return_value=True)),
        ):
            result = await integ.process_ner_feedback(
                user_id="u1",
                article_urn="urn:x",
                selected_text="art. 1",
                context_window="ctx",
                feedback_type="annotation",
                correct_reference={},
            )

        assert result.authority_breakdown == {"component_a": 0.5}


# ---------------------------------------------------------------------------
# NERRLCFIntegration – _update_user_track_record
# ---------------------------------------------------------------------------


class TestUpdateUserTrackRecord:
    @pytest.mark.asyncio
    async def test_uses_correct_action_type_for_confirmation(self):
        integ = _make_integration()

        with patch("merlt.rlcf.ner_rlcf_integration.get_async_session") as mock_session:
            # session returns no user found
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=AsyncMock(
                execute=AsyncMock(return_value=MagicMock(
                    scalar_one_or_none=MagicMock(return_value=None)
                )),
                commit=AsyncMock(),
            ))
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_session.return_value = ctx

            result = await integ._update_user_track_record(
                user_id="u_track",
                feedback_type="confirmation",
                current_authority=0.5,
            )

        integ.authority_service.calculate_authority_delta.assert_called_once_with(
            action="feedback_simple",
            current_authority=0.5,
        )
        # No user found → returns False
        assert result is False

    @pytest.mark.asyncio
    async def test_correction_uses_detailed_action_type(self):
        integ = _make_integration()

        with patch("merlt.rlcf.ner_rlcf_integration.get_async_session") as mock_session:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=AsyncMock(
                execute=AsyncMock(return_value=MagicMock(
                    scalar_one_or_none=MagicMock(return_value=None)
                )),
                commit=AsyncMock(),
            ))
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_session.return_value = ctx

            await integ._update_user_track_record(
                user_id="u_correction",
                feedback_type="correction",
                current_authority=0.4,
            )

        integ.authority_service.calculate_authority_delta.assert_called_once_with(
            action="feedback_detailed",
            current_authority=0.4,
        )

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        integ = _make_integration()

        with patch(
            "merlt.rlcf.ner_rlcf_integration.get_async_session",
            side_effect=Exception("conn error"),
        ):
            result = await integ._update_user_track_record(
                user_id="u_fail",
                feedback_type="confirmation",
                current_authority=0.5,
            )

        assert result is False


# ---------------------------------------------------------------------------
# NERRLCFIntegration – get_user_ner_history (DB unavailable)
# ---------------------------------------------------------------------------


class TestGetUserNERHistory:
    @pytest.mark.asyncio
    async def test_returns_empty_on_db_error(self):
        integ = _make_integration()

        with patch(
            "merlt.rlcf.ner_rlcf_integration.get_async_session",
            side_effect=Exception("no db"),
        ):
            history = await integ.get_user_ner_history("u_nodb")

        assert history == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_user_not_found(self):
        integ = _make_integration()

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        ))

        with patch("merlt.rlcf.ner_rlcf_integration.get_async_session") as mock_ctx:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.return_value = ctx

            history = await integ.get_user_ner_history("u_missing")

        assert history == []


# ---------------------------------------------------------------------------
# NERRLCFIntegration – get_ner_feedback_stats
# ---------------------------------------------------------------------------


class TestGetNERFeedbackStats:
    @pytest.mark.asyncio
    async def test_returns_dict_with_buffer_stats(self):
        integ = _make_integration()
        integ.buffer.get_buffer_stats = AsyncMock(return_value={
            "size": 7,
            "training_threshold": 50,
            "training_ready": False,
        })
        integ.buffer.get_authority_stats = AsyncMock(return_value={
            "total_users": 2,
        })

        with patch("merlt.rlcf.ner_rlcf_integration.get_async_session") as mock_ctx:
            session = AsyncMock()
            count_result = MagicMock(scalar=MagicMock(return_value=0))
            domain_result = MagicMock(all=MagicMock(return_value=[]))
            session.execute = AsyncMock(side_effect=[count_result, domain_result])
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.return_value = ctx

            stats = await integ.get_ner_feedback_stats()

        assert "buffer_stats" in stats
        assert stats["buffer_stats"]["size"] == 7
        assert "authority_stats" in stats

    @pytest.mark.asyncio
    async def test_graceful_on_db_error(self):
        integ = _make_integration()

        with patch(
            "merlt.rlcf.ner_rlcf_integration.get_async_session",
            side_effect=Exception("db down"),
        ):
            stats = await integ.get_ner_feedback_stats()

        # Should return minimal stats without raising
        assert "total_feedback" in stats
        assert stats["total_feedback"] == 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestGetNERRLCFIntegration:
    def test_returns_instance(self):
        import merlt.rlcf.ner_rlcf_integration as mod
        mod._global_integration = None
        with patch("merlt.rlcf.ner_rlcf_integration.get_ner_feedback_buffer", return_value=MagicMock()):
            integ = get_ner_rlcf_integration()
        assert isinstance(integ, NERRLCFIntegration)
        mod._global_integration = None

    def test_returns_same_instance_on_second_call(self):
        import merlt.rlcf.ner_rlcf_integration as mod
        mod._global_integration = None
        with patch("merlt.rlcf.ner_rlcf_integration.get_ner_feedback_buffer", return_value=MagicMock()):
            i1 = get_ner_rlcf_integration()
            i2 = get_ner_rlcf_integration()
        assert i1 is i2
        mod._global_integration = None
