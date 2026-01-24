"""
Test suite per 6-Dimensional Bias Detection
============================================

Test per:
- BiasDetector: rilevamento bias completo
- Singole dimensioni: demographic, professional, temporal, geographic, confirmation, anchoring
- Classificazione livelli e raccomandazioni

Esempio:
    pytest tests/rlcf/test_bias_detection.py -v
"""

import pytest
import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from merlt.rlcf.bias_detection import (
    BiasDetector,
    BiasReport,
    BiasLevel,
    BiasDimension,
    UserProfile,
    FeedbackForBias,
    calculate_demographic_bias,
    calculate_professional_clustering_bias,
    calculate_temporal_bias,
    calculate_geographic_bias,
    calculate_confirmation_bias,
    calculate_anchoring_bias,
    classify_bias_level,
    generate_mitigation_recommendations,
    create_bias_detector,
    calculate_bias_summary,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def bias_detector():
    """Crea BiasDetector con configurazione default."""
    return BiasDetector()


@pytest.fixture
def user_profiles():
    """Crea profili utente diversificati."""
    return [
        UserProfile(user_id="u1", profession="avvocato", region="nord", age_group="31-45"),
        UserProfile(user_id="u2", profession="avvocato", region="nord", age_group="31-45"),
        UserProfile(user_id="u3", profession="magistrato", region="sud", age_group="46-60"),
        UserProfile(user_id="u4", profession="magistrato", region="sud", age_group="46-60"),
        UserProfile(user_id="u5", profession="docente", region="centro", age_group="46-60"),
        UserProfile(user_id="u6", profession="studente", region="centro", age_group="18-30"),
    ]


@pytest.fixture
def diverse_feedbacks(user_profiles):
    """Crea feedbacks con diversità."""
    base_time = datetime.now()
    return [
        FeedbackForBias(
            feedback_id="f1", user_id="u1", task_id="task1",
            position="correct",
            timestamp=base_time,
            user_profile=user_profiles[0]
        ),
        FeedbackForBias(
            feedback_id="f2", user_id="u2", task_id="task1",
            position="correct",
            timestamp=base_time + timedelta(minutes=10),
            user_profile=user_profiles[1]
        ),
        FeedbackForBias(
            feedback_id="f3", user_id="u3", task_id="task1",
            position="incorrect",
            timestamp=base_time + timedelta(minutes=20),
            user_profile=user_profiles[2]
        ),
        FeedbackForBias(
            feedback_id="f4", user_id="u4", task_id="task1",
            position="incorrect",
            timestamp=base_time + timedelta(minutes=30),
            user_profile=user_profiles[3]
        ),
        FeedbackForBias(
            feedback_id="f5", user_id="u5", task_id="task1",
            position="partial",
            timestamp=base_time + timedelta(minutes=40),
            user_profile=user_profiles[4]
        ),
        FeedbackForBias(
            feedback_id="f6", user_id="u6", task_id="task1",
            position="correct",
            timestamp=base_time + timedelta(minutes=50),
            user_profile=user_profiles[5]
        ),
    ]


@pytest.fixture
def homogeneous_feedbacks():
    """Crea feedbacks omogenei (alto bias)."""
    base_time = datetime.now()
    profile = UserProfile(user_id="u1", profession="avvocato", region="nord")

    return [
        FeedbackForBias(
            feedback_id=f"f{i}", user_id=f"u{i}", task_id="task1",
            position="correct",  # Tutti stessa posizione
            timestamp=base_time + timedelta(minutes=i*5),
            user_profile=profile
        )
        for i in range(6)
    ]


@pytest.fixture
def temporal_drift_feedbacks():
    """Crea feedbacks con drift temporale."""
    base_time = datetime.now()

    feedbacks = []
    # Prima metà: tutti "correct"
    for i in range(5):
        feedbacks.append(FeedbackForBias(
            feedback_id=f"f{i}", user_id=f"u{i}", task_id="task1",
            position="correct",
            timestamp=base_time + timedelta(minutes=i*5),
            user_profile=None
        ))

    # Seconda metà: tutti "incorrect"
    for i in range(5, 10):
        feedbacks.append(FeedbackForBias(
            feedback_id=f"f{i}", user_id=f"u{i}", task_id="task1",
            position="incorrect",
            timestamp=base_time + timedelta(hours=2, minutes=i*5),
            user_profile=None
        ))

    return feedbacks


# =============================================================================
# TEST BIAS DETECTOR
# =============================================================================

class TestBiasDetector:
    """Test per BiasDetector principale."""

    @pytest.mark.asyncio
    async def test_calculate_total_bias_diverse(self, bias_detector, diverse_feedbacks):
        """Test con feedbacks diversificati (bias basso)."""
        report = await bias_detector.calculate_total_bias(
            task_id="task1",
            feedbacks=diverse_feedbacks
        )

        assert isinstance(report, BiasReport)
        assert report.task_id == "task1"
        assert report.num_feedbacks_analyzed == 6
        assert len(report.bias_scores) == 6
        assert report.total_bias_score >= 0

    @pytest.mark.asyncio
    async def test_calculate_total_bias_homogeneous(self, bias_detector, homogeneous_feedbacks):
        """Test con feedbacks omogenei (bias potenzialmente alto)."""
        report = await bias_detector.calculate_total_bias(
            task_id="task2",
            feedbacks=homogeneous_feedbacks
        )

        assert isinstance(report, BiasReport)
        assert report.num_feedbacks_analyzed == 6
        # Con feedback omogenei, alcuni bias potrebbero essere alti

    @pytest.mark.asyncio
    async def test_calculate_total_bias_empty(self, bias_detector):
        """Test con lista vuota."""
        report = await bias_detector.calculate_total_bias(
            task_id="task_empty",
            feedbacks=[]
        )

        assert report.total_bias_score == 0.0
        assert report.bias_level == BiasLevel.LOW
        assert report.num_feedbacks_analyzed == 0

    @pytest.mark.asyncio
    async def test_bias_formula(self, bias_detector, diverse_feedbacks):
        """Test che B_total = sqrt(sum(b_i^2))."""
        report = await bias_detector.calculate_total_bias(
            task_id="task_formula",
            feedbacks=diverse_feedbacks
        )

        # Ricalcola manualmente
        sum_squared = sum(b**2 for b in report.bias_scores.values())
        expected_total = math.sqrt(sum_squared)

        assert report.total_bias_score == pytest.approx(expected_total, abs=0.0001)

    @pytest.mark.asyncio
    async def test_report_to_dict(self, bias_detector, diverse_feedbacks):
        """Test serializzazione report."""
        report = await bias_detector.calculate_total_bias(
            task_id="task_dict",
            feedbacks=diverse_feedbacks
        )

        d = report.to_dict()

        assert "task_id" in d
        assert "bias_scores" in d
        assert "total_bias_score" in d
        assert "bias_level" in d
        assert "mitigation_recommendations" in d


# =============================================================================
# TEST DEMOGRAPHIC BIAS
# =============================================================================

class TestDemographicBias:
    """Test per calcolo b1: demographic bias."""

    def test_demographic_bias_diverse_professions(self, diverse_feedbacks):
        """Test con professioni diverse."""
        score, details = calculate_demographic_bias(diverse_feedbacks, "profession")

        assert score >= 0
        assert score <= 1
        assert "num_groups" in details
        assert details["num_groups"] >= 2  # Almeno 2 professioni diverse

    def test_demographic_bias_single_profession(self, homogeneous_feedbacks):
        """Test con professione unica."""
        score, details = calculate_demographic_bias(homogeneous_feedbacks, "profession")

        # Con una sola professione, diversità = 1 (quella professione ha tutte le posizioni)
        # Ma tutti hanno stessa posizione, quindi diversity = 1, bias = 0
        assert score >= 0

    def test_demographic_bias_empty(self):
        """Test con lista vuota."""
        score, details = calculate_demographic_bias([])

        assert score == 0.0
        assert "message" in details


# =============================================================================
# TEST PROFESSIONAL CLUSTERING BIAS
# =============================================================================

class TestProfessionalClusteringBias:
    """Test per calcolo b2: professional clustering."""

    def test_professional_clustering_diverse(self, diverse_feedbacks):
        """Test con feedback diversificati."""
        score, details = calculate_professional_clustering_bias(diverse_feedbacks)

        assert score >= 0
        assert score <= 1
        assert "num_professions" in details

    def test_professional_clustering_homogeneous(self, homogeneous_feedbacks):
        """Test con feedback omogenei."""
        score, details = calculate_professional_clustering_bias(homogeneous_feedbacks)

        # Con tutti stessa professione e posizione
        assert score >= 0

    def test_professional_clustering_empty(self):
        """Test con lista vuota."""
        score, details = calculate_professional_clustering_bias([])

        assert score == 0.0


# =============================================================================
# TEST TEMPORAL BIAS
# =============================================================================

class TestTemporalBias:
    """Test per calcolo b3: temporal drift."""

    def test_temporal_bias_with_drift(self, temporal_drift_feedbacks):
        """Test con drift temporale significativo."""
        score, details = calculate_temporal_bias(temporal_drift_feedbacks)

        # Prima metà correct, seconda metà incorrect = alto drift
        assert score > 0.5  # Dovrebbe essere alto
        assert details["dominant_position_changed"] is True

    def test_temporal_bias_stable(self, homogeneous_feedbacks):
        """Test senza drift temporale."""
        score, details = calculate_temporal_bias(homogeneous_feedbacks)

        # Tutti correct = nessun drift
        assert score == pytest.approx(0.0, abs=0.01)
        assert details["dominant_position_changed"] is False

    def test_temporal_bias_insufficient_data(self):
        """Test con dati insufficienti."""
        base_time = datetime.now()
        few_feedbacks = [
            FeedbackForBias(
                feedback_id="f1", user_id="u1", task_id="t1",
                position="correct", timestamp=base_time, user_profile=None
            )
        ]

        score, details = calculate_temporal_bias(few_feedbacks)

        assert score == 0.0
        assert "message" in details


# =============================================================================
# TEST GEOGRAPHIC BIAS
# =============================================================================

class TestGeographicBias:
    """Test per calcolo b4: geographic concentration."""

    def test_geographic_bias_diverse(self, diverse_feedbacks):
        """Test con regioni diverse."""
        score, details = calculate_geographic_bias(diverse_feedbacks)

        assert score >= 0
        assert score <= 1
        assert "num_regions" in details
        assert details["num_regions"] >= 2

    def test_geographic_bias_single_region(self, homogeneous_feedbacks):
        """Test con regione unica."""
        score, details = calculate_geographic_bias(homogeneous_feedbacks)

        assert score == 0.0  # Single region = not measurable

    def test_geographic_bias_empty(self):
        """Test con lista vuota."""
        score, details = calculate_geographic_bias([])

        assert score == 0.0


# =============================================================================
# TEST CONFIRMATION BIAS
# =============================================================================

class TestConfirmationBias:
    """Test per calcolo b5: confirmation bias."""

    def test_confirmation_bias_with_history(self, diverse_feedbacks):
        """Test con storico utenti."""
        # Storico: utenti hanno votato sempre "correct" in passato
        user_history = {
            "u1": ["correct", "correct", "correct"],
            "u2": ["correct", "correct"],
            "u3": ["incorrect", "incorrect"],  # u3 conferma
            "u4": ["correct"],  # u4 non conferma
        }

        score, details = calculate_confirmation_bias(diverse_feedbacks, user_history)

        assert score >= 0
        assert score <= 1
        assert "users_with_history" in details

    def test_confirmation_bias_no_history(self, diverse_feedbacks):
        """Test senza storico."""
        score, details = calculate_confirmation_bias(diverse_feedbacks, {})

        assert score == 0.0
        assert "message" in details

    def test_confirmation_bias_high_confirmation(self):
        """Test con alta conferma."""
        base_time = datetime.now()
        feedbacks = [
            FeedbackForBias(
                feedback_id=f"f{i}", user_id=f"u{i}", task_id="t1",
                position="correct", timestamp=base_time, user_profile=None
            )
            for i in range(5)
        ]

        # Tutti hanno storico "correct"
        user_history = {f"u{i}": ["correct", "correct"] for i in range(5)}

        score, details = calculate_confirmation_bias(feedbacks, user_history)

        # Alta conferma = alto bias
        assert score > 0.8


# =============================================================================
# TEST ANCHORING BIAS
# =============================================================================

class TestAnchoringBias:
    """Test per calcolo b6: anchoring bias."""

    def test_anchoring_bias_with_followers(self):
        """Test con followers dell'anchor."""
        base_time = datetime.now()

        # Prime 3 risposte: "correct" (anchor)
        # Successive: 4 "correct", 2 "incorrect" -> 4/6 = 0.67 followers
        feedbacks = [
            FeedbackForBias(
                feedback_id=f"f{i}", user_id=f"u{i}", task_id="t1",
                position="correct", timestamp=base_time + timedelta(minutes=i*5),
                user_profile=None
            )
            for i in range(7)
        ] + [
            FeedbackForBias(
                feedback_id=f"f{7+i}", user_id=f"u{7+i}", task_id="t1",
                position="incorrect",
                timestamp=base_time + timedelta(minutes=(7+i)*5),
                user_profile=None
            )
            for i in range(2)
        ]

        score, details = calculate_anchoring_bias(feedbacks, anchor_window=3)

        assert score >= 0
        assert score <= 1
        assert details["anchor_dominant_position"] == "correct"
        assert "follower_ratio" in details

    def test_anchoring_bias_no_followers(self):
        """Test senza followers."""
        base_time = datetime.now()

        # Prime 3: "correct", successive tutte "incorrect"
        feedbacks = [
            FeedbackForBias(
                feedback_id=f"f{i}", user_id=f"u{i}", task_id="t1",
                position="correct", timestamp=base_time + timedelta(minutes=i*5),
                user_profile=None
            )
            for i in range(3)
        ] + [
            FeedbackForBias(
                feedback_id=f"f{3+i}", user_id=f"u{3+i}", task_id="t1",
                position="incorrect",
                timestamp=base_time + timedelta(minutes=(3+i)*5),
                user_profile=None
            )
            for i in range(5)
        ]

        score, details = calculate_anchoring_bias(feedbacks, anchor_window=3)

        # Nessun follower = basso bias
        assert score == pytest.approx(0.0, abs=0.01)

    def test_anchoring_bias_insufficient_data(self):
        """Test con dati insufficienti."""
        feedbacks = [
            FeedbackForBias(
                feedback_id="f1", user_id="u1", task_id="t1",
                position="correct", timestamp=datetime.now(), user_profile=None
            )
        ]

        score, details = calculate_anchoring_bias(feedbacks, anchor_window=3)

        assert score == 0.0
        assert "message" in details


# =============================================================================
# TEST CLASSIFY BIAS LEVEL
# =============================================================================

class TestClassifyBiasLevel:
    """Test per classificazione livello bias."""

    def test_low_bias(self):
        """Test livello LOW."""
        assert classify_bias_level(0.0) == BiasLevel.LOW
        assert classify_bias_level(0.3) == BiasLevel.LOW
        assert classify_bias_level(0.5) == BiasLevel.LOW

    def test_medium_bias(self):
        """Test livello MEDIUM."""
        assert classify_bias_level(0.51) == BiasLevel.MEDIUM
        assert classify_bias_level(0.7) == BiasLevel.MEDIUM
        assert classify_bias_level(1.0) == BiasLevel.MEDIUM

    def test_high_bias(self):
        """Test livello HIGH."""
        assert classify_bias_level(1.01) == BiasLevel.HIGH
        assert classify_bias_level(1.5) == BiasLevel.HIGH
        assert classify_bias_level(2.0) == BiasLevel.HIGH


# =============================================================================
# TEST MITIGATION RECOMMENDATIONS
# =============================================================================

class TestMitigationRecommendations:
    """Test per generazione raccomandazioni."""

    def test_recommendations_high_demographic(self):
        """Test raccomandazione per alto demographic bias."""
        bias_scores = {
            "demographic": 0.8,
            "professional": 0.2,
            "temporal": 0.1,
            "geographic": 0.1,
            "confirmation": 0.1,
            "anchoring": 0.1,
        }

        recommendations = generate_mitigation_recommendations(bias_scores)

        assert len(recommendations) >= 1
        assert any("demografico" in r.lower() for r in recommendations)

    def test_recommendations_multiple_high(self):
        """Test con multipli bias alti."""
        bias_scores = {
            "demographic": 0.7,
            "professional": 0.6,
            "temporal": 0.3,
            "geographic": 0.1,
            "confirmation": 0.8,
            "anchoring": 0.7,
        }

        recommendations = generate_mitigation_recommendations(bias_scores)

        # Almeno 4 raccomandazioni (demographic, professional, confirmation, anchoring)
        assert len(recommendations) >= 4

    def test_recommendations_all_low(self):
        """Test con tutti i bias bassi."""
        bias_scores = {
            "demographic": 0.1,
            "professional": 0.2,
            "temporal": 0.1,
            "geographic": 0.1,
            "confirmation": 0.1,
            "anchoring": 0.2,
        }

        recommendations = generate_mitigation_recommendations(bias_scores)

        # Messaggio positivo
        assert len(recommendations) == 1
        assert "nessun bias significativo" in recommendations[0].lower()


# =============================================================================
# TEST FACTORY FUNCTIONS
# =============================================================================

class TestFactoryFunctions:
    """Test per factory functions."""

    def test_create_bias_detector_default(self):
        """Test factory con default."""
        detector = create_bias_detector()

        assert detector.threshold == 0.5
        assert detector.anchor_window == 3

    def test_create_bias_detector_custom(self):
        """Test factory con parametri custom."""
        detector = create_bias_detector(
            threshold=0.6,
            anchor_window=5,
            time_window_hours=48
        )

        assert detector.threshold == 0.6
        assert detector.anchor_window == 5
        assert detector.time_window_hours == 48

    def test_calculate_bias_summary(self, diverse_feedbacks):
        """Test calcolo summary rapido."""
        summary = calculate_bias_summary(diverse_feedbacks)

        assert "demographic" in summary
        assert "professional" in summary
        assert "temporal" in summary
        assert "geographic" in summary
        assert "confirmation" in summary
        assert "anchoring" in summary
        assert "total" in summary
        assert "level" in summary


# =============================================================================
# TEST DATACLASSES
# =============================================================================

class TestDataclasses:
    """Test per dataclass."""

    def test_user_profile_creation(self):
        """Test creazione UserProfile."""
        profile = UserProfile(
            user_id="u1",
            profession="avvocato",
            region="lombardia",
            age_group="31-45"
        )

        assert profile.user_id == "u1"
        assert profile.profession == "avvocato"

    def test_feedback_from_dict(self):
        """Test FeedbackForBias.from_dict."""
        data = {
            "feedback_id": "f1",
            "user_id": "u1",
            "task_id": "t1",
            "position": "correct",
            "timestamp": "2025-12-30T10:00:00",
            "user_profile": {
                "user_id": "u1",
                "profession": "avvocato"
            }
        }

        feedback = FeedbackForBias.from_dict(data)

        assert feedback.feedback_id == "f1"
        assert feedback.user_profile is not None
        assert feedback.user_profile.profession == "avvocato"

    def test_bias_report_to_dict(self):
        """Test BiasReport.to_dict."""
        report = BiasReport(
            task_id="t1",
            bias_scores={"demographic": 0.3, "professional": 0.4},
            total_bias_score=0.5,
            bias_level=BiasLevel.LOW,
            num_feedbacks_analyzed=10
        )

        d = report.to_dict()

        assert d["task_id"] == "t1"
        assert "bias_scores" in d
        assert d["bias_level"] == "low"


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test per casi limite."""

    def test_single_feedback(self):
        """Test con singolo feedback."""
        feedback = FeedbackForBias(
            feedback_id="f1",
            user_id="u1",
            task_id="t1",
            position="correct",
            timestamp=datetime.now(),
            user_profile=None
        )

        # Demographic
        score, _ = calculate_demographic_bias([feedback])
        assert score == 0.0  # Single position

        # Professional
        score, _ = calculate_professional_clustering_bias([feedback])
        assert score == 0.0

    def test_all_same_position(self, homogeneous_feedbacks):
        """Test quando tutti hanno stessa posizione."""
        # Temporal dovrebbe essere 0 (no drift)
        score, details = calculate_temporal_bias(homogeneous_feedbacks)
        assert score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_very_large_feedback_set(self, bias_detector):
        """Test con molti feedback."""
        base_time = datetime.now()

        feedbacks = [
            FeedbackForBias(
                feedback_id=f"f{i}",
                user_id=f"u{i % 10}",  # 10 utenti
                task_id="t1",
                position=["correct", "incorrect", "partial"][i % 3],
                timestamp=base_time + timedelta(minutes=i),
                user_profile=UserProfile(
                    user_id=f"u{i % 10}",
                    profession=["avvocato", "magistrato", "docente"][i % 3],
                    region=["nord", "sud", "centro"][i % 3]
                )
            )
            for i in range(100)
        ]

        report = await bias_detector.calculate_total_bias(
            task_id="large_task",
            feedbacks=feedbacks
        )

        assert report.num_feedbacks_analyzed == 100
        assert report.total_bias_score >= 0


# =============================================================================
# TEST SINGLE DIMENSION CALCULATION
# =============================================================================

class TestSingleDimensionCalculation:
    """Test per calcolo singola dimensione."""

    def test_calculate_single_demographic(self, bias_detector, diverse_feedbacks):
        """Test calcolo singola dimensione demographic."""
        score, details = bias_detector.calculate_single_dimension(
            BiasDimension.DEMOGRAPHIC,
            diverse_feedbacks
        )

        assert score >= 0
        assert score <= 1

    def test_calculate_single_anchoring(self, bias_detector, diverse_feedbacks):
        """Test calcolo singola dimensione anchoring."""
        score, details = bias_detector.calculate_single_dimension(
            BiasDimension.ANCHORING,
            diverse_feedbacks
        )

        assert score >= 0
        assert score <= 1

    def test_calculate_single_invalid(self, bias_detector, diverse_feedbacks):
        """Test con dimensione non valida."""
        with pytest.raises(ValueError):
            bias_detector.calculate_single_dimension(
                "invalid_dimension",  # type: ignore
                diverse_feedbacks
            )
