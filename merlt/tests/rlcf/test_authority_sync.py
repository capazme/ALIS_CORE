"""
Test suite per Authority Sync Service
======================================

Test per:
- VisualexUserSync dataclass
- AuthorityBreakdown dataclass
- AuthoritySyncService
- Calcolo baseline, track record, domain authority

Esempio:
    pytest tests/rlcf/test_authority_sync.py -v
"""

import pytest
from datetime import datetime, timezone


# =============================================================================
# TEST DATACLASS
# =============================================================================

class TestVisualexUserSync:
    """Test per VisualexUserSync."""

    def test_user_sync_creation_minimal(self):
        """Test creazione con campi minimi."""
        from merlt.rlcf.authority_sync import VisualexUserSync

        user = VisualexUserSync(
            visualex_user_id="visualex-123",
            merlt_user_id="merl-t-456",
            qualification="avvocato",
        )

        assert user.visualex_user_id == "visualex-123"
        assert user.merlt_user_id == "merl-t-456"
        assert user.qualification == "avvocato"
        assert user.years_experience == 0
        assert user.specializations == []
        assert user.total_feedback == 0

    def test_user_sync_creation_full(self):
        """Test creazione con tutti i campi."""
        from merlt.rlcf.authority_sync import VisualexUserSync

        user = VisualexUserSync(
            visualex_user_id="visualex-123",
            merlt_user_id="merl-t-456",
            qualification="magistrato",
            specializations=["civile", "penale"],
            years_experience=15,
            institution="Corte di Cassazione",
            total_feedback=50,
            validated_feedback=30,
            ingestions=10,
            validations=100,
            domain_activity={"civile": 80, "penale": 40},
        )

        assert user.qualification == "magistrato"
        assert user.years_experience == 15
        assert len(user.specializations) == 2
        assert user.institution == "Corte di Cassazione"
        assert user.total_feedback == 50
        assert user.validated_feedback == 30
        assert user.ingestions == 10
        assert user.validations == 100
        assert user.domain_activity["civile"] == 80

    def test_user_sync_has_timestamp(self):
        """Test che synced_at sia impostato automaticamente."""
        from merlt.rlcf.authority_sync import VisualexUserSync

        before = datetime.now()
        user = VisualexUserSync(
            visualex_user_id="v-1",
            merlt_user_id="m-1",
            qualification="studente",
        )
        after = datetime.now()

        assert before <= user.synced_at <= after


class TestAuthorityBreakdown:
    """Test per AuthorityBreakdown."""

    def test_breakdown_creation(self):
        """Test creazione breakdown."""
        from merlt.rlcf.authority_sync import AuthorityBreakdown

        breakdown = AuthorityBreakdown(
            baseline=0.6,
            track_record=0.5,
            level_authority=0.7,
            domain_scores={"civile": 0.8, "penale": 0.6},
            final_authority=0.58,
        )

        assert breakdown.baseline == 0.6
        assert breakdown.track_record == 0.5
        assert breakdown.level_authority == 0.7
        assert breakdown.final_authority == 0.58

    def test_breakdown_to_dict(self):
        """Test serializzazione."""
        from merlt.rlcf.authority_sync import AuthorityBreakdown

        breakdown = AuthorityBreakdown(
            baseline=0.6,
            track_record=0.5,
            level_authority=0.7,
            domain_scores={"civile": 0.8},
            final_authority=0.58,
        )

        result = breakdown.to_dict()

        assert result["baseline"] == 0.6
        assert result["track_record"] == 0.5
        assert result["level_authority"] == 0.7
        assert result["final_authority"] == 0.58
        assert "civile" in result["domain_scores"]


# =============================================================================
# TEST BASELINE CALCULATION
# =============================================================================

class TestBaselineCalculation:
    """Test per calcolo baseline credentials (B_u)."""

    @pytest.fixture
    def service(self):
        """Fixture per service."""
        from merlt.rlcf.authority_sync import AuthoritySyncService
        return AuthoritySyncService()

    def test_baseline_studente(self, service):
        """Test baseline per studente."""
        baseline = service._calculate_baseline(
            qualification="studente",
            years_experience=0,
            specializations=[],
        )
        assert baseline == 0.2

    def test_baseline_avvocato(self, service):
        """Test baseline per avvocato."""
        baseline = service._calculate_baseline(
            qualification="avvocato",
            years_experience=0,
            specializations=[],
        )
        assert baseline == 0.6

    def test_baseline_magistrato(self, service):
        """Test baseline per magistrato."""
        baseline = service._calculate_baseline(
            qualification="magistrato",
            years_experience=0,
            specializations=[],
        )
        assert baseline == 0.8

    def test_baseline_unknown_qualification(self, service):
        """Test baseline per qualifica sconosciuta (default 0.3)."""
        baseline = service._calculate_baseline(
            qualification="unknown",
            years_experience=0,
            specializations=[],
        )
        assert baseline == 0.3

    def test_baseline_case_insensitive(self, service):
        """Test che qualifica sia case-insensitive."""
        baseline1 = service._calculate_baseline(
            qualification="AVVOCATO",
            years_experience=0,
            specializations=[],
        )
        baseline2 = service._calculate_baseline(
            qualification="avvocato",
            years_experience=0,
            specializations=[],
        )
        assert baseline1 == baseline2 == 0.6

    def test_baseline_with_years_bonus(self, service):
        """Test bonus anni esperienza."""
        baseline_no_exp = service._calculate_baseline(
            qualification="avvocato",
            years_experience=0,
            specializations=[],
        )
        baseline_with_exp = service._calculate_baseline(
            qualification="avvocato",
            years_experience=5,
            specializations=[],
        )

        # 5 anni * 0.01 = 0.05 bonus
        assert baseline_with_exp == baseline_no_exp + 0.05

    def test_baseline_years_bonus_capped(self, service):
        """Test che bonus anni sia capped a 0.1."""
        baseline = service._calculate_baseline(
            qualification="avvocato",
            years_experience=30,  # 30 * 0.01 = 0.3, ma cap a 0.1
            specializations=[],
        )

        # 0.6 + 0.1 (capped) = 0.7
        assert baseline == 0.7

    def test_baseline_with_specializations(self, service):
        """Test bonus specializzazioni."""
        baseline = service._calculate_baseline(
            qualification="avvocato",
            years_experience=0,
            specializations=["civile", "penale"],
        )

        # 0.6 + (2 * 0.025) = 0.65
        assert baseline == 0.65

    def test_baseline_specializations_capped(self, service):
        """Test che bonus specializzazioni sia capped a 0.05."""
        baseline = service._calculate_baseline(
            qualification="avvocato",
            years_experience=0,
            specializations=["civile", "penale", "amministrativo", "tributario"],
        )

        # 0.6 + 0.05 (capped, 4 * 0.025 = 0.1 ma cap 0.05) = 0.65
        assert baseline == 0.65

    def test_baseline_combined_bonuses(self, service):
        """Test combinazione bonus anni + specializzazioni."""
        baseline = service._calculate_baseline(
            qualification="avvocato",
            years_experience=10,  # +0.1
            specializations=["civile"],  # +0.025
        )

        # 0.6 + 0.1 + 0.025 = 0.725
        assert baseline == 0.725

    def test_baseline_capped_at_one(self, service):
        """Test che baseline sia capped a 1.0."""
        baseline = service._calculate_baseline(
            qualification="giudice_suprema",  # 0.9
            years_experience=20,  # +0.1
            specializations=["civile", "penale"],  # +0.05
        )

        # 0.9 + 0.1 + 0.05 = 1.05 → capped a 1.0
        assert baseline == 1.0


# =============================================================================
# TEST TRACK RECORD CALCULATION
# =============================================================================

class TestTrackRecordCalculation:
    """Test per calcolo track record (T_u)."""

    @pytest.fixture
    def service(self):
        """Fixture per service."""
        from merlt.rlcf.authority_sync import AuthoritySyncService
        return AuthoritySyncService()

    def test_track_record_zero(self, service):
        """Test track record con zero attività."""
        track_record = service._calculate_track_record(
            total_feedback=0,
            validated_feedback=0,
            ingestions=0,
            validations=0,
        )
        assert track_record == 0.0

    def test_track_record_feedback_contribution(self, service):
        """Test contributo feedback."""
        track_record = service._calculate_track_record(
            total_feedback=10,  # 10 * 0.01 = 0.1
            validated_feedback=0,
            ingestions=0,
            validations=0,
        )
        assert track_record == 0.1

    def test_track_record_feedback_capped(self, service):
        """Test che feedback sia capped a 0.3."""
        track_record = service._calculate_track_record(
            total_feedback=100,  # 100 * 0.01 = 1.0, ma cap 0.3
            validated_feedback=0,
            ingestions=0,
            validations=0,
        )
        assert track_record == 0.3

    def test_track_record_validated_contribution(self, service):
        """Test contributo validated feedback."""
        track_record = service._calculate_track_record(
            total_feedback=0,
            validated_feedback=5,  # 5 * 0.02 = 0.1
            ingestions=0,
            validations=0,
        )
        assert track_record == 0.1

    def test_track_record_validated_capped(self, service):
        """Test che validated sia capped a 0.2."""
        track_record = service._calculate_track_record(
            total_feedback=0,
            validated_feedback=50,  # 50 * 0.02 = 1.0, ma cap 0.2
            ingestions=0,
            validations=0,
        )
        assert track_record == 0.2

    def test_track_record_ingestion_contribution(self, service):
        """Test contributo ingestions."""
        track_record = service._calculate_track_record(
            total_feedback=0,
            validated_feedback=0,
            ingestions=2,  # 2 * 0.05 = 0.1
            validations=0,
        )
        assert track_record == 0.1

    def test_track_record_ingestion_capped(self, service):
        """Test che ingestions sia capped a 0.2."""
        track_record = service._calculate_track_record(
            total_feedback=0,
            validated_feedback=0,
            ingestions=20,  # 20 * 0.05 = 1.0, ma cap 0.2
            validations=0,
        )
        assert track_record == 0.2

    def test_track_record_validation_contribution(self, service):
        """Test contributo validations."""
        track_record = service._calculate_track_record(
            total_feedback=0,
            validated_feedback=0,
            ingestions=0,
            validations=10,  # 10 * 0.01 = 0.1
        )
        assert track_record == 0.1

    def test_track_record_validation_capped(self, service):
        """Test che validations sia capped a 0.1."""
        track_record = service._calculate_track_record(
            total_feedback=0,
            validated_feedback=0,
            ingestions=0,
            validations=50,  # 50 * 0.01 = 0.5, ma cap 0.1
        )
        assert track_record == 0.1

    def test_track_record_combined(self, service):
        """Test combinazione tutti i contributi."""
        track_record = service._calculate_track_record(
            total_feedback=20,  # 0.2
            validated_feedback=5,  # 0.1
            ingestions=2,  # 0.1
            validations=5,  # 0.05
        )

        # 0.2 + 0.1 + 0.1 + 0.05 = 0.45
        assert track_record == 0.45

    def test_track_record_maximum(self, service):
        """Test track record massimo possibile."""
        track_record = service._calculate_track_record(
            total_feedback=100,  # cap 0.3
            validated_feedback=50,  # cap 0.2
            ingestions=20,  # cap 0.2
            validations=50,  # cap 0.1
        )

        # 0.3 + 0.2 + 0.2 + 0.1 = 0.8
        assert track_record == pytest.approx(0.8, abs=0.001)

    def test_track_record_capped_at_one(self, service):
        """Test che track record sia capped a 1.0."""
        # Anche se la somma supera 1.0 teoricamente,
        # i singoli cap limitano a 0.8 max
        track_record = service._calculate_track_record(
            total_feedback=200,
            validated_feedback=100,
            ingestions=50,
            validations=100,
        )

        assert track_record <= 1.0


# =============================================================================
# TEST DOMAIN AUTHORITY CALCULATION
# =============================================================================

class TestDomainAuthorityCalculation:
    """Test per calcolo domain authority (P_u)."""

    @pytest.fixture
    def service(self):
        """Fixture per service."""
        from merlt.rlcf.authority_sync import AuthoritySyncService
        return AuthoritySyncService()

    def test_domain_authority_empty(self, service):
        """Test con nessuna attività."""
        domain_scores = service._calculate_domain_authority({})
        assert domain_scores == {}

    def test_domain_authority_zero_activity(self, service):
        """Test con attività zero."""
        domain_scores = service._calculate_domain_authority(
            {"civile": 0, "penale": 0}
        )
        assert domain_scores == {}

    def test_domain_authority_single_domain(self, service):
        """Test con singolo dominio."""
        domain_scores = service._calculate_domain_authority(
            {"civile": 50}
        )

        assert "civile" in domain_scores
        # proportion = 1.0, engagement = 0.5, score = 0.5
        # No boost (activity <= 50)
        assert domain_scores["civile"] == 0.5

    def test_domain_authority_high_activity_boost(self, service):
        """Test boost per alta attività (>50)."""
        domain_scores = service._calculate_domain_authority(
            {"civile": 60}
        )

        # proportion = 1.0, engagement = 0.6, score = 0.6
        # Boost 1.2x → 0.72
        assert domain_scores["civile"] == 0.72

    def test_domain_authority_multiple_domains(self, service):
        """Test con multipli domini."""
        domain_scores = service._calculate_domain_authority(
            {"civile": 60, "penale": 40}
        )

        assert "civile" in domain_scores
        assert "penale" in domain_scores

        # civile: proportion = 0.6, engagement = 0.6, score = 0.36, boost → 0.432
        # penale: proportion = 0.4, engagement = 0.4, score = 0.16
        assert domain_scores["civile"] == 0.432
        assert domain_scores["penale"] == 0.16

    def test_domain_authority_saturation(self, service):
        """Test saturazione engagement a 1.0."""
        domain_scores = service._calculate_domain_authority(
            {"civile": 200}  # engagement = 200/100 = 2.0 → capped a 1.0
        )

        # proportion = 1.0, engagement = 1.0, score = 1.0
        # Boost 1.2x → 1.2 → capped a 1.0
        assert domain_scores["civile"] == 1.0

    def test_domain_authority_rounded(self, service):
        """Test che scores siano arrotondati a 3 decimali."""
        domain_scores = service._calculate_domain_authority(
            {"civile": 33, "penale": 67}
        )

        # Verifica che siano arrotondati
        for score in domain_scores.values():
            assert score == round(score, 3)


# =============================================================================
# TEST SYNC USER (INTEGRATION)
# =============================================================================

class TestSyncUser:
    """Test per sync_user (integrazione)."""

    @pytest.fixture
    def service(self):
        """Fixture per service."""
        from merlt.rlcf.authority_sync import AuthoritySyncService
        return AuthoritySyncService()

    @pytest.mark.asyncio
    async def test_sync_user_basic(self, service):
        """Test sync utente base."""
        from merlt.rlcf.authority_sync import VisualexUserSync

        user_data = VisualexUserSync(
            visualex_user_id="visualex-123",
            merlt_user_id="merl-t-456",
            qualification="avvocato",
        )

        authority, breakdown = await service.sync_user(user_data)

        # baseline = 0.6, track_record = 0, level_authority = 0
        # 0.4 * 0.6 + 0.4 * 0 + 0.2 * 0 = 0.24
        assert authority == 0.24
        assert breakdown.baseline == 0.6
        assert breakdown.track_record == 0.0
        assert breakdown.level_authority == 0.0

    @pytest.mark.asyncio
    async def test_sync_user_experienced(self, service):
        """Test sync utente con esperienza."""
        from merlt.rlcf.authority_sync import VisualexUserSync

        user_data = VisualexUserSync(
            visualex_user_id="visualex-123",
            merlt_user_id="merl-t-456",
            qualification="avvocato",
            years_experience=10,
            specializations=["civile"],
            total_feedback=20,
            validated_feedback=10,
        )

        authority, breakdown = await service.sync_user(user_data)

        # baseline = 0.6 + 0.1 (anni) + 0.025 (spec) = 0.725
        # track_record = 0.2 (feedback) + 0.2 (validated) = 0.4
        # level_authority = 0 (no domain activity)
        # 0.4 * 0.725 + 0.4 * 0.4 + 0.2 * 0 = 0.29 + 0.16 = 0.45
        assert breakdown.baseline == 0.725
        assert breakdown.track_record == 0.4
        assert authority == pytest.approx(0.45, abs=0.01)

    @pytest.mark.asyncio
    async def test_sync_user_expert(self, service):
        """Test sync utente esperto completo."""
        from merlt.rlcf.authority_sync import VisualexUserSync

        user_data = VisualexUserSync(
            visualex_user_id="visualex-123",
            merlt_user_id="merl-t-456",
            qualification="magistrato",
            years_experience=15,
            specializations=["civile", "penale"],
            total_feedback=30,
            validated_feedback=20,
            ingestions=5,
            validations=50,
            domain_activity={"civile": 80, "penale": 40},
        )

        authority, breakdown = await service.sync_user(user_data)

        # baseline = 0.8 + 0.1 (anni capped) + 0.05 (spec capped) = 0.95
        # track_record = 0.3 (feedback) + 0.2 (validated capped) + 0.2 (ingestion capped) + 0.1 (validation capped)
        # Note: feedback = 30 * 0.01 = 0.3, validated = 20 * 0.02 = 0.4 → cap 0.2
        # ingestions = 5 * 0.05 = 0.25 → cap 0.2, validations = 50 * 0.01 = 0.5 → cap 0.1
        # Total = 0.3 + 0.2 + 0.2 + 0.1 = 0.8
        # Verifica breakdown
        assert breakdown.baseline == pytest.approx(0.95, abs=0.001)
        assert breakdown.track_record == pytest.approx(0.8, abs=0.01)

        # authority alto
        assert authority >= 0.5

    @pytest.mark.asyncio
    async def test_sync_user_clamped_to_one(self, service):
        """Test che authority sia clamped a 1.0."""
        from merlt.rlcf.authority_sync import VisualexUserSync

        user_data = VisualexUserSync(
            visualex_user_id="visualex-123",
            merlt_user_id="merl-t-456",
            qualification="giudice_suprema",
            years_experience=30,
            specializations=["civile", "penale", "amministrativo"],
            total_feedback=100,
            validated_feedback=50,
            ingestions=20,
            validations=100,
            domain_activity={"civile": 200, "penale": 200},
        )

        authority, _ = await service.sync_user(user_data)

        assert authority <= 1.0

    @pytest.mark.asyncio
    async def test_sync_user_returns_breakdown(self, service):
        """Test che sync ritorni breakdown completo."""
        from merlt.rlcf.authority_sync import VisualexUserSync, AuthorityBreakdown

        user_data = VisualexUserSync(
            visualex_user_id="visualex-123",
            merlt_user_id="merl-t-456",
            qualification="docente",
            domain_activity={"civile": 60},
        )

        authority, breakdown = await service.sync_user(user_data)

        assert isinstance(breakdown, AuthorityBreakdown)
        assert breakdown.baseline > 0
        assert breakdown.final_authority == authority
        assert "civile" in breakdown.domain_scores


# =============================================================================
# TEST HELPER METHODS
# =============================================================================

class TestHelperMethods:
    """Test per metodi helper."""

    @pytest.fixture
    def service(self):
        """Fixture per service."""
        from merlt.rlcf.authority_sync import AuthoritySyncService
        return AuthoritySyncService()

    def test_calculate_authority_delta_feedback_simple(self, service):
        """Test delta per feedback semplice."""
        delta = service.calculate_authority_delta("feedback_simple", 0.5)
        assert delta == 0.001

    def test_calculate_authority_delta_feedback_detailed(self, service):
        """Test delta per feedback dettagliato."""
        delta = service.calculate_authority_delta("feedback_detailed", 0.5)
        assert delta == 0.005

    def test_calculate_authority_delta_ingestion(self, service):
        """Test delta per ingestion approvata."""
        delta = service.calculate_authority_delta("ingestion_approved", 0.5)
        assert delta == 0.01

    def test_calculate_authority_delta_validation_correct(self, service):
        """Test delta per validazione corretta."""
        delta = service.calculate_authority_delta("validation_correct", 0.5)
        assert delta == 0.003

    def test_calculate_authority_delta_validation_incorrect(self, service):
        """Test delta per validazione incorretta (negativo)."""
        delta = service.calculate_authority_delta("validation_incorrect", 0.5)
        assert delta == -0.002

    def test_calculate_authority_delta_unknown_action(self, service):
        """Test delta per azione sconosciuta."""
        delta = service.calculate_authority_delta("unknown_action", 0.5)
        assert delta == 0

    def test_calculate_authority_delta_diminishing_returns_high(self, service):
        """Test diminishing returns per authority alta (> 0.8)."""
        delta_normal = service.calculate_authority_delta("feedback_detailed", 0.5)
        delta_high = service.calculate_authority_delta("feedback_detailed", 0.85)

        # Con authority > 0.8, delta * 0.5
        assert delta_high == delta_normal * 0.5

    def test_calculate_authority_delta_diminishing_returns_very_high(self, service):
        """Test diminishing returns per authority molto alta (> 0.9)."""
        delta_normal = service.calculate_authority_delta("feedback_detailed", 0.5)
        delta_very_high = service.calculate_authority_delta("feedback_detailed", 0.95)

        # Con authority > 0.9, delta * 0.25
        assert delta_very_high == delta_normal * 0.25

    def test_estimate_authority_studente(self, service):
        """Test stima rapida per studente."""
        estimated = service.estimate_authority("studente")

        # baseline = 0.2, only baseline component
        # 0.2 * 0.4 = 0.08
        assert estimated == pytest.approx(0.08, abs=0.001)

    def test_estimate_authority_avvocato(self, service):
        """Test stima rapida per avvocato."""
        estimated = service.estimate_authority("avvocato")

        # baseline = 0.6, only baseline component
        # 0.6 * 0.4 = 0.24
        assert estimated == 0.24

    def test_estimate_authority_with_experience(self, service):
        """Test stima con esperienza."""
        estimated = service.estimate_authority(
            qualification="avvocato",
            years_experience=10,
            specializations=["civile"],
        )

        # baseline = 0.6 + 0.1 + 0.025 = 0.725
        # 0.725 * 0.4 = 0.29
        assert estimated == pytest.approx(0.29, abs=0.01)


# =============================================================================
# TEST SERVICE INITIALIZATION
# =============================================================================

class TestServiceInitialization:
    """Test per inizializzazione service."""

    def test_service_creates_without_clients(self):
        """Test che service si crei senza client db/cache."""
        from merlt.rlcf.authority_sync import AuthoritySyncService

        service = AuthoritySyncService()

        assert service.db_client is None
        assert service.cache_client is None

    def test_service_creates_with_clients(self):
        """Test che service accetti client custom."""
        from merlt.rlcf.authority_sync import AuthoritySyncService

        mock_db = object()
        mock_cache = object()

        service = AuthoritySyncService(
            db_client=mock_db,
            cache_client=mock_cache,
        )

        assert service.db_client is mock_db
        assert service.cache_client is mock_cache

    def test_service_constants(self):
        """Test che costanti siano corrette."""
        from merlt.rlcf.authority_sync import AuthoritySyncService

        assert AuthoritySyncService.WEIGHT_BASELINE == 0.4
        assert AuthoritySyncService.WEIGHT_TRACK_RECORD == 0.4
        assert AuthoritySyncService.WEIGHT_LEVEL_AUTHORITY == 0.2

        # Somma pesi = 1.0
        total_weight = (
            AuthoritySyncService.WEIGHT_BASELINE +
            AuthoritySyncService.WEIGHT_TRACK_RECORD +
            AuthoritySyncService.WEIGHT_LEVEL_AUTHORITY
        )
        assert total_weight == 1.0
