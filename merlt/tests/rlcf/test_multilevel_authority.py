"""
Test per RLCF Multilivello (Fase 1 v2 Recovery).

Verifica:
1. Inizializzazione authority per dominio e livello
2. Specializzazione utenti per dominio
3. Feedback multilivello
4. Authority combinata
5. Evoluzione authority nel tempo
"""

import pytest
from merlt.rlcf.simulator.users import (
    SyntheticUser,
    UserPool,
    DomainAuthority,
    PipelineLevelAuthority,
    create_user_pool,
    create_user_from_profile,
    create_domain_specialist_user,
    LEGAL_DOMAINS,
    PIPELINE_LEVELS,
    DEFAULT_COMBINATION_WEIGHTS,
    PROFILES,
)


class TestDomainAuthority:
    """Test dataclass DomainAuthority."""

    def test_init_defaults(self):
        """Verifica inizializzazione con default."""
        auth = DomainAuthority(domain="civile")
        assert auth.domain == "civile"
        assert auth.baseline == 0.5
        assert auth.track_record == 0.5
        assert auth.current == 0.5
        assert auth.feedback_count == 0

    def test_to_dict(self):
        """Verifica serializzazione."""
        auth = DomainAuthority(
            domain="penale",
            baseline=0.8,
            track_record=0.75,
            current=0.77,
            feedback_count=10
        )
        d = auth.to_dict()
        assert d["domain"] == "penale"
        assert d["baseline"] == 0.8
        assert d["feedback_count"] == 10

    def test_from_dict(self):
        """Verifica deserializzazione."""
        data = {
            "domain": "amministrativo",
            "baseline": 0.6,
            "track_record": 0.65,
            "current": 0.62,
            "feedback_count": 5
        }
        auth = DomainAuthority.from_dict(data)
        assert auth.domain == "amministrativo"
        assert auth.baseline == 0.6
        assert auth.feedback_count == 5


class TestPipelineLevelAuthority:
    """Test dataclass PipelineLevelAuthority."""

    def test_init_defaults(self):
        """Verifica inizializzazione con default."""
        auth = PipelineLevelAuthority(level="reasoning")
        assert auth.level == "reasoning"
        assert auth.baseline == 0.5
        assert auth.current == 0.5

    def test_to_from_dict(self):
        """Verifica round-trip serializzazione."""
        original = PipelineLevelAuthority(
            level="synthesis",
            baseline=0.7,
            track_record=0.72,
            current=0.71,
            feedback_count=15
        )
        data = original.to_dict()
        restored = PipelineLevelAuthority.from_dict(data)

        assert restored.level == original.level
        assert restored.baseline == original.baseline
        assert restored.feedback_count == original.feedback_count


class TestSyntheticUserMultilevel:
    """Test SyntheticUser con supporto multilivello."""

    def test_domain_authorities_initialized(self):
        """Verifica che domain authorities siano inizializzate."""
        user = create_user_from_profile("strict_expert", user_id=1)

        # Tutti i domini devono essere presenti
        for domain in LEGAL_DOMAINS:
            assert domain in user.domain_authorities
            assert isinstance(user.domain_authorities[domain], DomainAuthority)

    def test_level_authorities_initialized(self):
        """Verifica che level authorities siano inizializzate."""
        user = create_user_from_profile("domain_specialist", user_id=1)

        # Tutti i livelli devono essere presenti
        for level in PIPELINE_LEVELS:
            assert level in user.level_authorities
            assert isinstance(user.level_authorities[level], PipelineLevelAuthority)

    def test_specialization_boosts_domain_authority(self):
        """Verifica che specializzazione aumenti authority dominio."""
        # Crea utente specializzato in diritto civile
        user = create_user_from_profile(
            "domain_specialist",
            user_id=1,
            override={"credentials": {"specialization": "diritto civile"}}
        )

        # Authority civile deve essere più alta di penale
        civile_auth = user.get_domain_authority("civile")
        penale_auth = user.get_domain_authority("penale")

        assert civile_auth > penale_auth

    def test_get_domain_authority_unknown_domain(self):
        """Verifica fallback per dominio sconosciuto."""
        user = create_user_from_profile("strict_expert", user_id=1)
        assert user.get_domain_authority("unknown") == 0.5

    def test_get_level_authority_unknown_level(self):
        """Verifica fallback per livello sconosciuto."""
        user = create_user_from_profile("strict_expert", user_id=1)
        assert user.get_level_authority("unknown") == 0.5

    def test_get_combined_authority(self):
        """Verifica calcolo authority combinata."""
        user = create_user_from_profile("strict_expert", user_id=1)

        combined = user.get_combined_authority("civile", "reasoning")

        # Deve essere media pesata
        expected = (
            DEFAULT_COMBINATION_WEIGHTS["general"] * user.current_authority +
            DEFAULT_COMBINATION_WEIGHTS["domain"] * user.get_domain_authority("civile") +
            DEFAULT_COMBINATION_WEIGHTS["level"] * user.get_level_authority("reasoning")
        )

        assert abs(combined - expected) < 0.01

    def test_get_combined_authority_custom_weights(self):
        """Verifica authority combinata con pesi custom."""
        user = create_user_from_profile("strict_expert", user_id=1)

        custom_weights = {"general": 0.5, "domain": 0.3, "level": 0.2}
        combined = user.get_combined_authority("penale", "synthesis", weights=custom_weights)

        expected = (
            0.5 * user.current_authority +
            0.3 * user.get_domain_authority("penale") +
            0.2 * user.get_level_authority("synthesis")
        )

        assert abs(combined - expected) < 0.01


class TestRecordFeedbackMultilevel:
    """Test record_feedback_multilevel."""

    def test_feedback_increases_domain_authority(self):
        """Verifica che feedback positivo aumenti authority dominio."""
        user = create_user_from_profile("lenient_student", user_id=1)

        initial_domain = user.get_domain_authority("penale")
        initial_level = user.get_level_authority("reasoning")

        # Simula feedback positivo
        result = user.record_feedback_multilevel(
            feedback={"rating": 0.9},
            quality_score=0.9,
            domain="penale",
            level="reasoning",
            feedback_accuracy=0.9
        )

        # Authority deve aumentare
        assert user.get_domain_authority("penale") > initial_domain
        assert user.get_level_authority("reasoning") > initial_level

        # Result contiene tutte le componenti
        assert "general" in result
        assert "domain" in result
        assert "level" in result
        assert "combined" in result

    def test_feedback_decreases_authority_on_low_score(self):
        """Verifica che feedback negativo diminuisca authority."""
        user = create_user_from_profile("strict_expert", user_id=1)

        initial_domain = user.get_domain_authority("civile")

        # Simula feedback negativo
        user.record_feedback_multilevel(
            feedback={"rating": 0.2},
            quality_score=0.2,
            domain="civile",
            level="retrieval",
            feedback_accuracy=0.2
        )

        # Authority deve diminuire
        assert user.get_domain_authority("civile") < initial_domain

    def test_feedback_updates_only_specified_domain(self):
        """Verifica che feedback aggiorni solo dominio specificato."""
        user = create_user_from_profile("domain_specialist", user_id=1)

        initial_penale = user.get_domain_authority("penale")
        initial_civile = user.get_domain_authority("civile")

        # Feedback solo su penale
        user.record_feedback_multilevel(
            feedback={},
            quality_score=0.95,
            domain="penale",
            level="reasoning",
            feedback_accuracy=0.95
        )

        # Penale cambia, civile no
        assert user.get_domain_authority("penale") > initial_penale
        assert user.get_domain_authority("civile") == initial_civile

    def test_feedback_updates_only_specified_level(self):
        """Verifica che feedback aggiorni solo livello specificato."""
        user = create_user_from_profile("domain_specialist", user_id=1)

        initial_reasoning = user.get_level_authority("reasoning")
        initial_synthesis = user.get_level_authority("synthesis")

        # Feedback solo su reasoning
        user.record_feedback_multilevel(
            feedback={},
            quality_score=0.95,
            domain="civile",
            level="reasoning",
            feedback_accuracy=0.95
        )

        # Reasoning cambia, synthesis no
        assert user.get_level_authority("reasoning") > initial_reasoning
        assert user.get_level_authority("synthesis") == initial_synthesis

    def test_feedback_increments_count(self):
        """Verifica che feedback incrementi counter."""
        user = create_user_from_profile("strict_expert", user_id=1)

        assert user.domain_authorities["penale"].feedback_count == 0
        assert user.level_authorities["reasoning"].feedback_count == 0

        user.record_feedback_multilevel(
            feedback={},
            quality_score=0.8,
            domain="penale",
            level="reasoning"
        )

        assert user.domain_authorities["penale"].feedback_count == 1
        assert user.level_authorities["reasoning"].feedback_count == 1

    def test_authority_clamped_to_bounds(self):
        """Verifica che authority rimanga in [0, 1]."""
        user = create_user_from_profile("strict_expert", user_id=1)

        # Simula 100 feedback ottimi
        for _ in range(100):
            user.record_feedback_multilevel(
                feedback={},
                quality_score=1.0,
                domain="civile",
                level="synthesis",
                feedback_accuracy=1.0
            )

        # Authority non deve superare 1.0
        assert user.get_domain_authority("civile") <= 1.0
        assert user.get_level_authority("synthesis") <= 1.0


class TestCreateDomainSpecialistUser:
    """Test funzione create_domain_specialist_user."""

    def test_creates_specialist(self):
        """Verifica creazione specialista."""
        user = create_domain_specialist_user(
            user_id=1,
            domain="penale",
            baseline_authority=0.85
        )

        assert user.user_id == 1
        assert "penale" in user.profile_type
        assert user.credentials.get("specialization") == "penale"

    def test_specialist_has_higher_domain_authority(self):
        """Verifica che specialista abbia authority più alta nel suo dominio."""
        user = create_domain_specialist_user(
            user_id=1,
            domain="costituzionale",
            baseline_authority=0.8
        )

        # Authority costituzionale deve essere più alta di altri domini
        costituzionale = user.get_domain_authority("costituzionale")
        civile = user.get_domain_authority("civile")

        assert costituzionale > civile

    @pytest.mark.parametrize("domain", LEGAL_DOMAINS)
    def test_all_domains_valid(self, domain):
        """Verifica che tutti i domini siano validi per creazione."""
        user = create_domain_specialist_user(
            user_id=1,
            domain=domain,
            baseline_authority=0.75
        )

        assert user is not None
        assert user.credentials.get("specialization") == domain

    def test_invalid_domain_raises(self):
        """Verifica che dominio non valido sollevi errore."""
        with pytest.raises(ValueError, match="Dominio sconosciuto"):
            create_domain_specialist_user(
                user_id=1,
                domain="fantasy_law",
                baseline_authority=0.8
            )


class TestUserPoolMultilevel:
    """Test UserPool con supporto multilivello."""

    def test_pool_users_have_multilevel(self):
        """Verifica che utenti del pool abbiano authority multilivello."""
        pool = create_user_pool({
            "strict_expert": 2,
            "lenient_student": 2
        })

        for user in pool.users:
            assert len(user.domain_authorities) == len(LEGAL_DOMAINS)
            assert len(user.level_authorities) == len(PIPELINE_LEVELS)

    def test_to_dict_includes_multilevel(self):
        """Verifica che serializzazione includa dati multilivello."""
        user = create_user_from_profile("strict_expert", user_id=1)
        d = user.to_dict()

        assert "domain_authorities" in d
        assert "level_authorities" in d
        assert "civile" in d["domain_authorities"]
        assert "reasoning" in d["level_authorities"]


class TestMultilevelEvolution:
    """Test evoluzione authority multilivello nel tempo."""

    def test_domain_authority_converges(self):
        """Verifica che authority dominio converga con feedback consistente."""
        # Usa un utente con baseline più alta per test convergenza
        user = create_user_from_profile("domain_specialist", user_id=1)

        initial_authority = user.get_domain_authority("penale")
        authorities_over_time = []

        # Simula 50 feedback ottimi su penale
        for _ in range(50):
            user.record_feedback_multilevel(
                feedback={},
                quality_score=0.95,
                domain="penale",
                level="reasoning",
                feedback_accuracy=0.95
            )
            authorities_over_time.append(user.get_domain_authority("penale"))

        # Authority deve essere aumentata significativamente
        assert user.get_domain_authority("penale") > initial_authority + 0.1

        # Ultimi 10 valori devono essere stabili (convergenza)
        last_10 = authorities_over_time[-10:]
        variance = max(last_10) - min(last_10)
        assert variance < 0.05, "Authority dovrebbe convergere"

    def test_different_domains_evolve_independently(self):
        """Verifica che domini diversi evolvano indipendentemente."""
        user = create_user_from_profile("domain_specialist", user_id=1)

        # Feedback positivi su penale
        for _ in range(10):
            user.record_feedback_multilevel(
                feedback={},
                quality_score=0.9,
                domain="penale",
                level="reasoning",
                feedback_accuracy=0.9
            )

        # Feedback negativi su civile
        for _ in range(10):
            user.record_feedback_multilevel(
                feedback={},
                quality_score=0.2,
                domain="civile",
                level="reasoning",
                feedback_accuracy=0.2
            )

        # Penale alto, civile basso
        assert user.get_domain_authority("penale") > user.get_domain_authority("civile")
