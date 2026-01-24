"""
Test suite per Devil's Advocate System
======================================

Test per:
- DevilsAdvocateAssigner: assegnazione probabilistica
- Critical prompts: generazione task-specific
- Effectiveness metrics: analisi impatto advocate

Esempio:
    pytest tests/rlcf/test_devils_advocate.py -v
"""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from merlt.rlcf.devils_advocate import (
    DevilsAdvocateAssigner,
    DevilsAdvocateAssignment,
    AdvocateFeedback,
    EffectivenessMetrics,
    TaskType,
    CRITICAL_KEYWORDS,
    CRITICAL_KEYWORDS_IT,
    CRITICAL_KEYWORDS_EN,
    BASE_CRITICAL_QUESTIONS,
    TASK_CRITICAL_PROMPTS,
    create_devils_advocate_assigner,
    analyze_feedback_for_critical_thinking,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def assigner():
    """Crea assigner con configurazione default."""
    return DevilsAdvocateAssigner()


@pytest.fixture
def assigner_custom():
    """Crea assigner con configurazione custom."""
    return DevilsAdvocateAssigner(
        max_advocate_ratio=0.2,
        min_advocates=2,
        min_authority_threshold=0.3
    )


@pytest.fixture
def eligible_users():
    """Lista di utenti eligibili."""
    return [
        {"user_id": "user_1", "authority": 0.9},
        {"user_id": "user_2", "authority": 0.7},
        {"user_id": "user_3", "authority": 0.6},
        {"user_id": "user_4", "authority": 0.5},
        {"user_id": "user_5", "authority": 0.4},  # Sotto soglia default
        {"user_id": "user_6", "authority": 0.3},  # Sotto soglia default
    ]


@pytest.fixture
def advocate_feedbacks():
    """Lista di feedback da advocate."""
    return [
        AdvocateFeedback(
            user_id="user_1",
            task_id="task_001",
            position="incorrect",
            reasoning="Tuttavia, l'interpretazione è troppo restrittiva. Esistono alternative valide.",
            critical_points=["Interpretazione restrittiva", "Manca giurisprudenza recente"],
            confidence=0.8
        ),
        AdvocateFeedback(
            user_id="user_2",
            task_id="task_001",
            position="needs_revision",
            reasoning="Sebbene il ragionamento sia logico, presenta debolezze strutturali. Problema con le fonti.",
            critical_points=["Debolezze strutturali", "Fonti incomplete"],
            confidence=0.7
        )
    ]


@pytest.fixture
def regular_feedbacks():
    """Lista di feedback regolari."""
    return [
        {"user_id": "user_3", "position": "correct", "reasoning": "La risposta è completa."},
        {"user_id": "user_4", "position": "correct", "reasoning": "Condivido l'analisi."},
        {"user_id": "user_5", "position": "partially_correct", "reasoning": "Alcuni punti validi."},
    ]


# =============================================================================
# TEST PROBABILITY CALCULATION
# =============================================================================

class TestProbabilityCalculation:
    """Test per calcolo probabilità P(advocate) = min(0.1, 3/|E|)."""

    def test_probability_small_pool(self, assigner):
        """Con pochi utenti, probabilità = 3/|E| > 0.1."""
        # 10 utenti: 3/10 = 0.3, ma capped a 0.1
        prob = assigner.calculate_advocate_probability(10)
        assert prob == 0.1

    def test_probability_medium_pool(self, assigner):
        """Con pool medio, probabilità segue formula."""
        # 50 utenti: 3/50 = 0.06 < 0.1
        prob = assigner.calculate_advocate_probability(50)
        assert prob == pytest.approx(0.06, abs=0.001)

    def test_probability_large_pool(self, assigner):
        """Con pool grande, probabilità piccola."""
        # 100 utenti: 3/100 = 0.03
        prob = assigner.calculate_advocate_probability(100)
        assert prob == pytest.approx(0.03, abs=0.001)

    def test_probability_very_small_pool(self, assigner):
        """Con pochissimi utenti, capped a 0.1."""
        # 5 utenti: 3/5 = 0.6, ma capped a 0.1
        prob = assigner.calculate_advocate_probability(5)
        assert prob == 0.1

    def test_probability_empty_pool(self, assigner):
        """Con pool vuoto, probabilità 0."""
        prob = assigner.calculate_advocate_probability(0)
        assert prob == 0.0

    def test_probability_negative_pool(self, assigner):
        """Con pool negativo, probabilità 0."""
        prob = assigner.calculate_advocate_probability(-5)
        assert prob == 0.0

    def test_probability_single_user(self, assigner):
        """Con singolo utente, capped a 0.1."""
        prob = assigner.calculate_advocate_probability(1)
        assert prob == 0.1  # min(0.1, 3/1) = 0.1


class TestNumAdvocatesCalculation:
    """Test per calcolo numero advocate."""

    def test_num_advocates_small_pool(self, assigner):
        """Con pool piccolo, almeno 1."""
        num = assigner.calculate_num_advocates(5)
        assert num >= 1
        assert num <= 3  # Non più di min_advocates

    def test_num_advocates_large_pool(self, assigner):
        """Con pool grande, segue probabilità."""
        num = assigner.calculate_num_advocates(100)
        # 100 * 0.03 = 3
        assert num == 3

    def test_num_advocates_empty_pool(self, assigner):
        """Con pool vuoto, 0 advocate."""
        num = assigner.calculate_num_advocates(0)
        assert num == 0

    def test_num_advocates_medium_pool(self, assigner):
        """Con pool medio."""
        num = assigner.calculate_num_advocates(30)
        # max(1, 30 * 0.1) = max(1, 3) = 3
        assert num >= 1
        assert num <= 3


# =============================================================================
# TEST ASSIGNMENT
# =============================================================================

class TestAssignment:
    """Test per assegnazione advocate."""

    @pytest.mark.asyncio
    async def test_assign_basic(self, assigner, eligible_users):
        """Test assegnazione base."""
        assignments = await assigner.assign_advocates_for_task(
            task_id="task_001",
            eligible_users=eligible_users,
            task_type=TaskType.QA
        )

        assert len(assignments) >= 1
        for a in assignments:
            assert isinstance(a, DevilsAdvocateAssignment)
            assert a.task_id == "task_001"
            assert a.critical_prompt != ""

    @pytest.mark.asyncio
    async def test_assign_filters_by_authority(self, assigner, eligible_users):
        """Test che filtra per authority >= 0.5."""
        assignments = await assigner.assign_advocates_for_task(
            task_id="task_002",
            eligible_users=eligible_users,
            task_type=TaskType.QA
        )

        # Solo utenti con authority >= 0.5 (user_1, 2, 3, 4)
        assigned_ids = [a.user_id for a in assignments]
        assert "user_5" not in assigned_ids
        assert "user_6" not in assigned_ids

    @pytest.mark.asyncio
    async def test_assign_with_custom_threshold(self, assigner_custom, eligible_users):
        """Test con soglia authority custom (0.3)."""
        assignments = await assigner_custom.assign_advocates_for_task(
            task_id="task_003",
            eligible_users=eligible_users,
            task_type=TaskType.QA
        )

        # Tutti tranne user_6 (0.3) potrebbero essere selezionati
        assert len(assignments) >= 1

    @pytest.mark.asyncio
    async def test_assign_empty_pool(self, assigner):
        """Test con pool vuoto."""
        assignments = await assigner.assign_advocates_for_task(
            task_id="task_004",
            eligible_users=[],
            task_type=TaskType.QA
        )

        assert len(assignments) == 0

    @pytest.mark.asyncio
    async def test_assign_no_qualified_users(self, assigner):
        """Test quando nessuno supera soglia authority."""
        low_authority_users = [
            {"user_id": "low_1", "authority": 0.2},
            {"user_id": "low_2", "authority": 0.3},
        ]

        assignments = await assigner.assign_advocates_for_task(
            task_id="task_005",
            eligible_users=low_authority_users,
            task_type=TaskType.QA
        )

        assert len(assignments) == 0

    @pytest.mark.asyncio
    async def test_assign_caches_assignments(self, assigner, eligible_users):
        """Test che le assegnazioni sono cached."""
        await assigner.assign_advocates_for_task(
            task_id="task_cache",
            eligible_users=eligible_users,
            task_type=TaskType.QA
        )

        cached = assigner.get_assignments_for_task("task_cache")
        assert len(cached) >= 1


# =============================================================================
# TEST CRITICAL PROMPTS
# =============================================================================

class TestCriticalPrompts:
    """Test per generazione prompt critici."""

    def test_generate_prompt_qa(self, assigner):
        """Test prompt per task QA."""
        prompt = assigner.generate_critical_prompt(TaskType.QA)

        assert "Devil's Advocate" in prompt
        assert "Completezza" in prompt or "completamente" in prompt.lower()
        # Base questions included
        assert "punti deboli" in prompt.lower()

    def test_generate_prompt_classification(self, assigner):
        """Test prompt per task CLASSIFICATION."""
        prompt = assigner.generate_critical_prompt(TaskType.CLASSIFICATION)

        assert "Devil's Advocate" in prompt
        assert "Casi Limite" in prompt
        assert "categorie" in prompt.lower()

    def test_generate_prompt_prediction(self, assigner):
        """Test prompt per task PREDICTION."""
        prompt = assigner.generate_critical_prompt(TaskType.PREDICTION)

        assert "Devil's Advocate" in prompt
        assert "Esiti Alternativi" in prompt

    def test_generate_prompt_drafting(self, assigner):
        """Test prompt per task DRAFTING."""
        prompt = assigner.generate_critical_prompt(TaskType.DRAFTING)

        assert "Devil's Advocate" in prompt
        assert "Precisione Giuridica" in prompt
        assert "ambiguità" in prompt.lower()

    def test_all_task_types_have_prompts(self):
        """Test che tutti i TaskType hanno prompt definiti."""
        for task_type in TaskType:
            assert task_type in TASK_CRITICAL_PROMPTS or task_type == TaskType.QA
            # QA è il default, altri devono essere definiti

    def test_base_questions_present(self, assigner):
        """Test che le domande base sono sempre presenti."""
        prompt = assigner.generate_critical_prompt(TaskType.QA)

        for question in BASE_CRITICAL_QUESTIONS:
            assert question in prompt


# =============================================================================
# TEST CRITICAL ENGAGEMENT ANALYSIS
# =============================================================================

class TestCriticalEngagementAnalysis:
    """Test per analisi engagement critico."""

    def test_analyze_italian_keywords(self, assigner):
        """Test rilevamento keyword italiane."""
        text = "Tuttavia, esistono problemi con questa interpretazione. Sebbene sia logica, presenta debolezze."

        score, count = assigner.analyze_critical_engagement(text)

        assert count >= 3  # tuttavia, problemi, sebbene, debolezze
        assert score == 1.0  # Max score con 3+ keywords

    def test_analyze_english_keywords(self, assigner):
        """Test rilevamento keyword inglesi."""
        text = "However, there are issues with this approach. Although valid, it has limitations."

        score, count = assigner.analyze_critical_engagement(text)

        assert count >= 3  # however, issues, although, limitations
        assert score == 1.0

    def test_analyze_mixed_keywords(self, assigner):
        """Test con mix italiano/inglese."""
        text = "Tuttavia there is a problem. However, ci sono alternative."

        score, count = assigner.analyze_critical_engagement(text)

        assert count >= 4
        assert score == 1.0

    def test_analyze_no_critical_keywords(self, assigner):
        """Test senza keyword critiche."""
        text = "La risposta è corretta e completa. Tutto bene."

        score, count = assigner.analyze_critical_engagement(text)

        assert count == 0
        assert score == 0.0

    def test_analyze_empty_text(self, assigner):
        """Test con testo vuoto."""
        score, count = assigner.analyze_critical_engagement("")

        assert count == 0
        assert score == 0.0

    def test_analyze_partial_keywords(self, assigner):
        """Test con alcune keyword."""
        # "problema" contiene "problem" come sottostringa, quindi matcha 2 keywords
        text = "C'è un problema con questa analisi."

        score, count = assigner.analyze_critical_engagement(text)

        assert count >= 2  # "problema" (IT) + "problem" (EN substring)
        assert score == pytest.approx(0.667, abs=0.01)  # 2/3


# =============================================================================
# TEST EFFECTIVENESS METRICS
# =============================================================================

class TestEffectivenessMetrics:
    """Test per metriche effectiveness."""

    def test_effectiveness_with_unique_positions(self, assigner, advocate_feedbacks, regular_feedbacks):
        """Test diversità con posizioni uniche."""
        metrics = assigner.analyze_advocate_effectiveness(
            advocate_feedbacks,
            regular_feedbacks
        )

        assert isinstance(metrics, EffectivenessMetrics)
        # Advocate hanno posizioni diverse da regular
        assert metrics.unique_positions_introduced >= 1
        assert metrics.diversity_score > 0

    def test_effectiveness_engagement_score(self, assigner, advocate_feedbacks, regular_feedbacks):
        """Test engagement score."""
        metrics = assigner.analyze_advocate_effectiveness(
            advocate_feedbacks,
            regular_feedbacks
        )

        # Engagement basato su lunghezza e keyword critiche
        assert metrics.engagement_score > 0
        assert metrics.avg_reasoning_length > 0

    def test_effectiveness_empty_advocate_feedbacks(self, assigner, regular_feedbacks):
        """Test con nessun feedback advocate."""
        metrics = assigner.analyze_advocate_effectiveness(
            [],
            regular_feedbacks
        )

        assert metrics.diversity_score == 0.0
        assert metrics.engagement_score == 0.0
        assert metrics.overall_effectiveness == 0.0

    def test_effectiveness_overall_calculation(self, assigner, advocate_feedbacks, regular_feedbacks):
        """Test calcolo overall effectiveness."""
        metrics = assigner.analyze_advocate_effectiveness(
            advocate_feedbacks,
            regular_feedbacks
        )

        # Overall = 0.4 * diversity + 0.6 * engagement
        expected = 0.4 * metrics.diversity_score + 0.6 * metrics.engagement_score
        assert metrics.overall_effectiveness == pytest.approx(expected, abs=0.001)

    def test_effectiveness_to_dict(self, assigner, advocate_feedbacks, regular_feedbacks):
        """Test serializzazione metrics."""
        metrics = assigner.analyze_advocate_effectiveness(
            advocate_feedbacks,
            regular_feedbacks
        )

        d = metrics.to_dict()

        assert "diversity_score" in d
        assert "engagement_score" in d
        assert "overall_effectiveness" in d
        assert isinstance(d["diversity_score"], float)


# =============================================================================
# TEST DATACLASSES
# =============================================================================

class TestDataclasses:
    """Test per dataclass."""

    def test_assignment_to_dict(self):
        """Test serializzazione DevilsAdvocateAssignment."""
        assignment = DevilsAdvocateAssignment(
            assignment_id="adv_001",
            task_id="task_001",
            user_id="user_1",
            critical_prompt="Test prompt",
            instructions="Test instructions"
        )

        d = assignment.to_dict()

        assert d["assignment_id"] == "adv_001"
        assert d["task_id"] == "task_001"
        assert d["user_id"] == "user_1"
        assert d["completed"] is False
        assert "assigned_at" in d

    def test_advocate_feedback_to_dict(self):
        """Test serializzazione AdvocateFeedback."""
        feedback = AdvocateFeedback(
            user_id="user_1",
            task_id="task_001",
            position="incorrect",
            reasoning="Test reasoning",
            critical_points=["Point 1"],
            confidence=0.8
        )

        d = feedback.to_dict()

        assert d["user_id"] == "user_1"
        assert d["position"] == "incorrect"
        assert d["confidence"] == 0.8
        assert len(d["critical_points"]) == 1


# =============================================================================
# TEST FACTORY FUNCTIONS
# =============================================================================

class TestFactoryFunctions:
    """Test per factory functions."""

    def test_create_assigner_default(self):
        """Test factory con default."""
        assigner = create_devils_advocate_assigner()

        assert assigner.max_advocate_ratio == 0.1
        assert assigner.min_advocates == 3
        assert assigner.min_authority_threshold == 0.5

    def test_create_assigner_custom(self):
        """Test factory con parametri custom."""
        assigner = create_devils_advocate_assigner(
            max_advocate_ratio=0.15,
            min_advocates=5,
            min_authority_threshold=0.6
        )

        assert assigner.max_advocate_ratio == 0.15
        assert assigner.min_advocates == 5
        assert assigner.min_authority_threshold == 0.6

    def test_analyze_feedback_italian(self):
        """Test analyze_feedback_for_critical_thinking italiano."""
        result = analyze_feedback_for_critical_thinking(
            "Tuttavia ci sono problemi con questa interpretazione.",
            language="it"
        )

        assert result["has_critical_thinking"] is True
        assert result["score"] > 0
        assert len(result["keywords_found"]) > 0

    def test_analyze_feedback_english(self):
        """Test analyze_feedback_for_critical_thinking inglese."""
        result = analyze_feedback_for_critical_thinking(
            "However there are issues with this approach.",
            language="en"
        )

        assert result["has_critical_thinking"] is True
        assert result["score"] > 0

    def test_analyze_feedback_empty(self):
        """Test con feedback vuoto."""
        result = analyze_feedback_for_critical_thinking("")

        assert result["has_critical_thinking"] is False
        assert result["score"] == 0.0


# =============================================================================
# TEST MARK ASSIGNMENT COMPLETED
# =============================================================================

class TestMarkCompleted:
    """Test per mark_assignment_completed."""

    @pytest.mark.asyncio
    async def test_mark_completed(self, assigner, eligible_users):
        """Test marking assignment as completed."""
        assignments = await assigner.assign_advocates_for_task(
            task_id="task_complete",
            eligible_users=eligible_users,
            task_type=TaskType.QA
        )

        assert len(assignments) >= 1
        user_id = assignments[0].user_id

        # Mark completed
        result = assigner.mark_assignment_completed(
            task_id="task_complete",
            user_id=user_id,
            effectiveness_score=0.85
        )

        assert result is True

        # Verify
        updated = assigner.get_assignments_for_task("task_complete")
        completed = [a for a in updated if a.user_id == user_id][0]
        assert completed.completed is True
        assert completed.effectiveness_score == 0.85

    def test_mark_completed_not_found(self, assigner):
        """Test marking non-existent assignment."""
        result = assigner.mark_assignment_completed(
            task_id="nonexistent",
            user_id="nonexistent_user"
        )

        assert result is False


# =============================================================================
# TEST CONSTANTS
# =============================================================================

class TestConstants:
    """Test per costanti."""

    def test_critical_keywords_sets(self):
        """Test che keyword sets sono popolati."""
        assert len(CRITICAL_KEYWORDS_IT) > 10
        assert len(CRITICAL_KEYWORDS_EN) > 10
        assert len(CRITICAL_KEYWORDS) == len(CRITICAL_KEYWORDS_IT) + len(CRITICAL_KEYWORDS_EN)

    def test_all_task_types_defined(self):
        """Test che tutti i TaskType hanno config."""
        expected_types = ["QA", "CLASSIFICATION", "PREDICTION", "DRAFTING"]
        for task_type in expected_types:
            assert TaskType(task_type) in TASK_CRITICAL_PROMPTS

    def test_base_questions_count(self):
        """Test numero domande base."""
        assert len(BASE_CRITICAL_QUESTIONS) == 4


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test per casi limite."""

    @pytest.mark.asyncio
    async def test_single_eligible_user(self, assigner):
        """Test con singolo utente eligibile."""
        single_user = [{"user_id": "solo", "authority": 0.8}]

        assignments = await assigner.assign_advocates_for_task(
            task_id="task_single",
            eligible_users=single_user,
            task_type=TaskType.QA
        )

        assert len(assignments) == 1
        assert assignments[0].user_id == "solo"

    @pytest.mark.asyncio
    async def test_all_users_same_authority(self, assigner):
        """Test con tutti utenti stessa authority."""
        same_auth_users = [
            {"user_id": f"user_{i}", "authority": 0.7}
            for i in range(10)
        ]

        assignments = await assigner.assign_advocates_for_task(
            task_id="task_same_auth",
            eligible_users=same_auth_users,
            task_type=TaskType.QA
        )

        assert len(assignments) >= 1
        assert len(assignments) <= 3  # max min_advocates

    def test_very_long_reasoning_text(self, assigner):
        """Test con testo reasoning molto lungo."""
        long_text = "Tuttavia " * 100  # 900+ caratteri con keyword

        score, count = assigner.analyze_critical_engagement(long_text)

        # Dovrebbe contare multiple occorrenze
        assert count >= 1
        assert score > 0

    def test_special_characters_in_text(self, assigner):
        """Test con caratteri speciali."""
        special_text = "Tuttavia, c'è un problema!!! @#$% Alternative?"

        score, count = assigner.analyze_critical_engagement(special_text)

        assert count >= 2  # tuttavia, problema, alternative
        assert score > 0
