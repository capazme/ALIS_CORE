"""
Test per Curriculum Learning
============================

Test completi per DifficultyAssessor, CurriculumScheduler, QueryPool.
"""

import pytest
from unittest.mock import MagicMock

from merlt.rlcf.curriculum_learning import (
    DifficultyLevel,
    CurriculumStage,
    DifficultyAssessment,
    CurriculumConfig,
    CurriculumStats,
    DifficultyAssessor,
    CurriculumScheduler,
    CurriculumQuery,
    QueryPool
)


# =============================================================================
# TEST DIFFICULTY LEVEL
# =============================================================================

class TestDifficultyLevel:
    """Test per DifficultyLevel enum."""

    def test_values(self):
        """Test valori enum."""
        assert DifficultyLevel.EASY.value == "easy"
        assert DifficultyLevel.MEDIUM.value == "medium"
        assert DifficultyLevel.HARD.value == "hard"

    def test_from_string(self):
        """Test conversione da stringa."""
        assert DifficultyLevel("easy") == DifficultyLevel.EASY
        assert DifficultyLevel("hard") == DifficultyLevel.HARD


# =============================================================================
# TEST CURRICULUM STAGE
# =============================================================================

class TestCurriculumStage:
    """Test per CurriculumStage enum."""

    def test_values(self):
        """Test valori enum."""
        assert CurriculumStage.WARMUP.value == "warmup"
        assert CurriculumStage.EASY.value == "easy"
        assert CurriculumStage.MIXED.value == "mixed"


# =============================================================================
# TEST DIFFICULTY ASSESSMENT
# =============================================================================

class TestDifficultyAssessment:
    """Test per DifficultyAssessment."""

    def test_create_assessment(self):
        """Test creazione assessment."""
        assessment = DifficultyAssessment(
            level=DifficultyLevel.EASY,
            score=0.25
        )

        assert assessment.level == DifficultyLevel.EASY
        assert assessment.score == 0.25
        assert assessment.confidence == 1.0

    def test_with_factors(self):
        """Test con fattori."""
        factors = {
            "linguistic_complexity": 0.3,
            "legal_concepts": 0.5,
            "expert_diversity": 0.2
        }

        assessment = DifficultyAssessment(
            level=DifficultyLevel.MEDIUM,
            score=0.5,
            factors=factors,
            confidence=0.8
        )

        assert assessment.factors["legal_concepts"] == 0.5
        assert assessment.confidence == 0.8

    def test_to_dict(self):
        """Test serializzazione."""
        assessment = DifficultyAssessment(
            level=DifficultyLevel.HARD,
            score=0.789123,
            factors={"test": 0.5},
            confidence=0.95
        )

        data = assessment.to_dict()

        assert data["level"] == "hard"
        assert data["score"] == 0.7891
        assert data["confidence"] == 0.95


# =============================================================================
# TEST CURRICULUM CONFIG
# =============================================================================

class TestCurriculumConfig:
    """Test per CurriculumConfig."""

    def test_default_config(self):
        """Test config default."""
        config = CurriculumConfig()

        assert config.warmup_epochs == 5
        assert config.performance_threshold_advance == 0.7
        assert config.min_epochs_per_stage == 3

    def test_custom_config(self):
        """Test config custom."""
        config = CurriculumConfig(
            warmup_epochs=10,
            performance_threshold_advance=0.8,
            performance_threshold_regress=0.3
        )

        assert config.warmup_epochs == 10
        assert config.performance_threshold_advance == 0.8

    def test_difficulty_weights(self):
        """Test pesi difficoltà."""
        config = CurriculumConfig()

        weights = config.difficulty_weights
        assert "linguistic_complexity" in weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_stage_mix(self):
        """Test mix difficoltà per stage."""
        config = CurriculumConfig()

        warmup_mix = config.stage_difficulty_mix["warmup"]
        assert warmup_mix["easy"] == 1.0
        assert warmup_mix["hard"] == 0.0

        mixed_mix = config.stage_difficulty_mix["mixed"]
        assert mixed_mix["easy"] > 0
        assert mixed_mix["hard"] > 0

    def test_to_dict(self):
        """Test serializzazione."""
        config = CurriculumConfig()
        data = config.to_dict()

        assert "warmup_epochs" in data
        assert "difficulty_weights" in data


# =============================================================================
# TEST DIFFICULTY ASSESSOR
# =============================================================================

class TestDifficultyAssessor:
    """Test per DifficultyAssessor."""

    @pytest.fixture
    def assessor(self):
        """Crea assessor per test."""
        return DifficultyAssessor()

    def test_create_assessor(self):
        """Test creazione assessor."""
        assessor = DifficultyAssessor()
        assert assessor is not None

    def test_assess_simple_query(self, assessor):
        """Test valutazione query semplice."""
        query = "Cos'è il contratto?"

        assessment = assessor.assess(query)

        assert assessment.level in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
        assert 0 <= assessment.score <= 1

    def test_assess_complex_query(self, assessor):
        """Test valutazione query complessa."""
        query = """
        Qual è il rapporto tra la responsabilità precontrattuale
        derivante dalla culpa in contrahendo e la disciplina del
        risarcimento del danno per inadempimento contrattuale,
        considerando anche il bilanciamento tra buona fede oggettiva
        e tutela dell'affidamento secondo la giurisprudenza?
        """

        assessment = assessor.assess(query)

        # Query complessa dovrebbe essere medium o hard
        assert assessment.level in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        assert assessment.score > 0.3  # Non easy

    def test_linguistic_complexity(self, assessor):
        """Test fattore complessità linguistica."""
        short_query = "Cos'è?"
        long_query = " ".join(["parola"] * 50)  # 50 parole

        short_assessment = assessor.assess(short_query)
        long_assessment = assessor.assess(long_query)

        # Query più lunga = più complessa linguisticamente
        short_ling = short_assessment.factors.get("linguistic_complexity", 0)
        long_ling = long_assessment.factors.get("linguistic_complexity", 0)

        assert long_ling >= short_ling

    def test_legal_concepts(self, assessor):
        """Test fattore concetti giuridici."""
        simple_query = "Come funziona?"
        legal_query = "Qual è la responsabilità del debitore per inadempimento dell'obbligazione contrattuale?"

        simple_assessment = assessor.assess(simple_query)
        legal_assessment = assessor.assess(legal_query)

        simple_concepts = simple_assessment.factors.get("legal_concepts", 0)
        legal_concepts = legal_assessment.factors.get("legal_concepts", 0)

        assert legal_concepts >= simple_concepts

    def test_assess_with_metadata(self, assessor):
        """Test con metadata."""
        query = "Domanda generica"
        metadata = {
            "domain": "civile",
            "expected_experts": ["literal", "systemic", "principles"]
        }

        assessment = assessor.assess(query, metadata)

        # Expert diversity dovrebbe essere alta (3 expert)
        expert_div = assessment.factors.get("expert_diversity", 0)
        assert expert_div >= 0.5

    def test_historical_performance(self, assessor):
        """Test aggiornamento performance storica."""
        domain = "penale"

        assessor.update_historical_performance(domain, 0.8)
        assert assessor.historical_performance[domain] == 0.8

        assessor.update_historical_performance(domain, 0.6)
        # Exponential moving average
        assert 0.7 < assessor.historical_performance[domain] < 0.8

    def test_interpretive_ambiguity(self, assessor):
        """Test fattore ambiguità."""
        clear_query = "Definizione di contratto"
        ambiguous_query = "Come si interpretano le diverse opinioni sulla controversa questione?"

        clear_assessment = assessor.assess(clear_query)
        ambiguous_assessment = assessor.assess(ambiguous_query)

        clear_ambig = clear_assessment.factors.get("interpretive_ambiguity", 0)
        ambig_ambig = ambiguous_assessment.factors.get("interpretive_ambiguity", 0)

        assert ambig_ambig >= clear_ambig

    def test_confidence_calculation(self, assessor):
        """Test calcolo confidenza."""
        assessment = assessor.assess("Query normale")

        assert 0 <= assessment.confidence <= 1


# =============================================================================
# TEST CURRICULUM SCHEDULER
# =============================================================================

class TestCurriculumScheduler:
    """Test per CurriculumScheduler."""

    @pytest.fixture
    def scheduler(self):
        """Crea scheduler per test."""
        return CurriculumScheduler()

    def test_create_scheduler(self):
        """Test creazione scheduler."""
        scheduler = CurriculumScheduler()

        assert scheduler.current_stage == CurriculumStage.WARMUP
        assert scheduler.stats.epochs_in_stage == 0

    def test_create_with_custom_config(self):
        """Test con config custom."""
        config = CurriculumConfig(warmup_epochs=10)
        scheduler = CurriculumScheduler(config)

        assert scheduler.config.warmup_epochs == 10

    def test_assess_difficulty(self, scheduler):
        """Test valutazione difficoltà."""
        assessment = scheduler.assess_difficulty("Cos'è il contratto?")

        assert isinstance(assessment, DifficultyAssessment)
        assert assessment.level in DifficultyLevel

    def test_should_include_in_batch_warmup(self, scheduler):
        """Test inclusione in batch durante warmup."""
        # Durante warmup, solo easy dovrebbe essere incluso (prob=1.0)
        easy_query = "Cos'è?"
        hard_query = "Complessa interpretazione sistematica delle antinomie normative"

        # Mock assessor per controllo
        scheduler.assessor = MagicMock()

        scheduler.assessor.assess.return_value = DifficultyAssessment(
            level=DifficultyLevel.EASY, score=0.2
        )
        easy_include, easy_prob = scheduler.should_include_in_batch(easy_query)
        assert easy_prob == 1.0

        scheduler.assessor.assess.return_value = DifficultyAssessment(
            level=DifficultyLevel.HARD, score=0.8
        )
        hard_include, hard_prob = scheduler.should_include_in_batch(hard_query)
        assert hard_prob == 0.0

    def test_filter_batch_by_curriculum(self, scheduler):
        """Test filtro batch."""
        queries = [
            {"query": f"Query {i}"} for i in range(100)
        ]

        filtered = scheduler.filter_batch_by_curriculum(queries, target_size=20)

        # Durante warmup, dovrebbe restituire al massimo target_size
        assert len(filtered) <= 20

    def test_update_after_epoch(self, scheduler):
        """Test aggiornamento dopo epoch."""
        result = scheduler.update_after_epoch(avg_reward=0.5)

        assert result["stage_changed"] is False
        assert scheduler.stats.total_epochs == 1
        assert scheduler.stats.epochs_in_stage == 1

    def test_warmup_progression(self, scheduler):
        """Test progressione da warmup."""
        config = CurriculumConfig(warmup_epochs=3, min_epochs_per_stage=1)
        scheduler = CurriculumScheduler(config)

        # Completa warmup
        for i in range(4):
            result = scheduler.update_after_epoch(avg_reward=0.5)

        # Dovrebbe essere passato a EASY
        assert scheduler.current_stage == CurriculumStage.EASY

    def test_performance_advance(self, scheduler):
        """Test avanzamento per performance."""
        config = CurriculumConfig(
            warmup_epochs=0,  # Skip warmup
            min_epochs_per_stage=1,
            performance_threshold_advance=0.7
        )
        scheduler = CurriculumScheduler(config)
        scheduler.stats.current_stage = CurriculumStage.EASY

        # Alta performance - avanza
        result = scheduler.update_after_epoch(avg_reward=0.8)

        # Dovrebbe essere avanzato almeno a MEDIUM (potrebbe anche a HARD se continua)
        assert scheduler.current_stage != CurriculumStage.EASY
        assert result["stage_changed"] is True

    def test_performance_regress(self, scheduler):
        """Test regressione per performance bassa."""
        config = CurriculumConfig(
            warmup_epochs=0,
            min_epochs_per_stage=1,
            performance_threshold_regress=0.4
        )
        scheduler = CurriculumScheduler(config)
        scheduler.stats.current_stage = CurriculumStage.MEDIUM
        scheduler.stats.epochs_in_stage = 3

        # Bassa performance
        result = scheduler.update_after_epoch(avg_reward=0.3)

        # Dovrebbe regredire a EASY
        assert scheduler.current_stage == CurriculumStage.EASY
        assert result["stage_changed"] is True

    def test_min_epochs_before_transition(self, scheduler):
        """Test minimo epochs prima di transizione."""
        config = CurriculumConfig(
            warmup_epochs=0,
            min_epochs_per_stage=5,
            performance_threshold_advance=0.7
        )
        scheduler = CurriculumScheduler(config)
        scheduler.stats.current_stage = CurriculumStage.EASY

        # Alta performance ma pochi epochs
        for i in range(3):
            scheduler.update_after_epoch(avg_reward=0.8)

        # Non dovrebbe ancora avanzare
        assert scheduler.current_stage == CurriculumStage.EASY

    def test_stage_history(self, scheduler):
        """Test tracking storia stage."""
        config = CurriculumConfig(warmup_epochs=2, min_epochs_per_stage=1)
        scheduler = CurriculumScheduler(config)

        # Completa warmup
        scheduler.update_after_epoch(avg_reward=0.5)
        scheduler.update_after_epoch(avg_reward=0.5)
        scheduler.update_after_epoch(avg_reward=0.5)

        history = scheduler.stats.stage_history
        assert len(history) >= 1
        assert history[0]["from_stage"] == "warmup"

    def test_get_stats(self, scheduler):
        """Test statistiche."""
        scheduler.update_after_epoch(avg_reward=0.6)
        stats = scheduler.get_stats()

        assert stats.total_epochs == 1
        assert stats.current_stage == CurriculumStage.WARMUP

    def test_reset(self, scheduler):
        """Test reset."""
        scheduler.update_after_epoch(avg_reward=0.5)
        scheduler.update_after_epoch(avg_reward=0.6)

        scheduler.reset()

        assert scheduler.stats.total_epochs == 0
        assert scheduler.current_stage == CurriculumStage.WARMUP


# =============================================================================
# TEST QUERY POOL
# =============================================================================

class TestQueryPool:
    """Test per QueryPool."""

    @pytest.fixture
    def pool(self):
        """Crea pool per test."""
        return QueryPool()

    def test_create_pool(self):
        """Test creazione pool."""
        pool = QueryPool()

        assert len(pool) == 0
        assert pool.stats()["total"] == 0

    def test_add_query(self, pool):
        """Test aggiunta query."""
        level = pool.add("Cos'è il contratto?")

        assert len(pool) == 1
        assert level in DifficultyLevel

    def test_add_with_difficulty_override(self, pool):
        """Test con override difficoltà."""
        level = pool.add(
            "Query qualsiasi",
            difficulty=DifficultyLevel.HARD
        )

        assert level == DifficultyLevel.HARD
        assert len(pool.get_by_difficulty(DifficultyLevel.HARD)) == 1

    def test_add_with_domain(self, pool):
        """Test con domain."""
        pool.add("Query", domain="civile")

        query = pool._all_queries[0]
        assert query.domain == "civile"

    def test_add_batch(self, pool):
        """Test aggiunta batch."""
        queries = [
            {"query": "Query semplice", "difficulty": "easy"},
            {"query": "Query media", "difficulty": "medium"},
            {"query": "Query difficile", "difficulty": "hard"}
        ]

        counts = pool.add_batch(queries)

        assert counts["easy"] == 1
        assert counts["medium"] == 1
        assert counts["hard"] == 1
        assert len(pool) == 3

    def test_sample(self, pool):
        """Test sampling."""
        # Aggiungi query
        for i in range(30):
            pool.add(f"Easy query {i}", difficulty=DifficultyLevel.EASY)
        for i in range(30):
            pool.add(f"Medium query {i}", difficulty=DifficultyLevel.MEDIUM)
        for i in range(30):
            pool.add(f"Hard query {i}", difficulty=DifficultyLevel.HARD)

        sampled = pool.sample(30)

        assert len(sampled) == 30
        assert all(isinstance(q, CurriculumQuery) for q in sampled)

    def test_sample_with_mix(self, pool):
        """Test sampling con mix specifico."""
        # Aggiungi query
        for i in range(50):
            pool.add(f"Easy query {i}", difficulty=DifficultyLevel.EASY)
        for i in range(50):
            pool.add(f"Hard query {i}", difficulty=DifficultyLevel.HARD)

        # Sample solo easy
        sampled = pool.sample(
            20,
            difficulty_mix={"easy": 1.0, "medium": 0.0, "hard": 0.0}
        )

        # Tutti dovrebbero essere easy
        for q in sampled:
            assert "Easy" in q.query

    def test_get_by_difficulty(self, pool):
        """Test recupero per difficoltà."""
        pool.add("Easy", difficulty=DifficultyLevel.EASY)
        pool.add("Medium", difficulty=DifficultyLevel.MEDIUM)

        easy = pool.get_by_difficulty(DifficultyLevel.EASY)
        assert len(easy) == 1
        assert easy[0].query == "Easy"

    def test_stats(self, pool):
        """Test statistiche pool."""
        pool.add("E1", difficulty=DifficultyLevel.EASY)
        pool.add("E2", difficulty=DifficultyLevel.EASY)
        pool.add("M1", difficulty=DifficultyLevel.MEDIUM)
        pool.add("H1", difficulty=DifficultyLevel.HARD)

        stats = pool.stats()

        assert stats["total"] == 4
        assert stats["easy"] == 2
        assert stats["medium"] == 1
        assert stats["hard"] == 1


# =============================================================================
# TEST INTEGRAZIONE
# =============================================================================

class TestCurriculumIntegration:
    """Test di integrazione curriculum learning."""

    def test_full_training_simulation(self):
        """Simula training completo con curriculum."""
        config = CurriculumConfig(
            warmup_epochs=2,
            min_epochs_per_stage=2,
            performance_threshold_advance=0.65,
            performance_threshold_regress=0.35
        )
        scheduler = CurriculumScheduler(config)

        # Crea pool
        pool = QueryPool()
        for i in range(30):
            pool.add(f"Easy {i}", difficulty=DifficultyLevel.EASY)
        for i in range(30):
            pool.add(f"Medium {i}", difficulty=DifficultyLevel.MEDIUM)
        for i in range(30):
            pool.add(f"Hard {i}", difficulty=DifficultyLevel.HARD)

        # Simula training
        stages_visited = [scheduler.current_stage]

        for epoch in range(20):
            # Get difficulty mix per stage corrente
            mix = config.stage_difficulty_mix[scheduler.current_stage.value]

            # Sample batch
            batch = pool.sample(20, difficulty_mix=mix)

            # Simula reward (migliora nel tempo, ma varia)
            base_reward = 0.5 + epoch * 0.02
            noise = (epoch % 3 - 1) * 0.1
            avg_reward = max(0.2, min(0.9, base_reward + noise))

            # Update
            result = scheduler.update_after_epoch(avg_reward)

            if result["stage_changed"]:
                stages_visited.append(scheduler.current_stage)

        # Verifica progressione
        assert scheduler.stats.total_epochs == 20
        # Dovrebbe aver progredito oltre warmup
        assert len(stages_visited) > 1

    def test_adaptive_regression(self):
        """Test regressione adattiva."""
        config = CurriculumConfig(
            warmup_epochs=0,
            min_epochs_per_stage=2,
            performance_threshold_advance=0.7,
            performance_threshold_regress=0.3
        )
        scheduler = CurriculumScheduler(config)

        # Parti da MEDIUM
        scheduler.stats.current_stage = CurriculumStage.MEDIUM
        scheduler.stats.epochs_in_stage = 3

        # Performance che crolla
        scheduler.update_after_epoch(avg_reward=0.6)
        assert scheduler.current_stage == CurriculumStage.MEDIUM

        scheduler.update_after_epoch(avg_reward=0.25)

        # Dovrebbe regredire
        assert scheduler.current_stage == CurriculumStage.EASY

    def test_difficulty_distribution_tracking(self):
        """Test tracking distribuzione difficoltà."""
        scheduler = CurriculumScheduler()

        queries = [
            {"query": "Simple question"},
            {"query": "Medium complexity with contratto and obbligazione"},
            {"query": "Very complex interpretation of antinomie normative"}
        ]

        scheduler.filter_batch_by_curriculum(queries, target_size=3)

        # Distribution dovrebbe essere tracciata
        dist = scheduler.stats.difficulty_distribution
        assert sum(dist.values()) == 3
