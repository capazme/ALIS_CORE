"""
Test PromptPolicy
==================

Test per soft prompt tuning via REINFORCE.
"""

import pytest
from tempfile import TemporaryDirectory
from pathlib import Path

# Skip se torch non disponibile
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch non disponibile")


class TestPromptFeedback:
    """Test per PromptFeedback dataclass."""

    def test_creation(self):
        """Test creazione PromptFeedback."""
        from merlt.rlcf.prompt_policy import PromptFeedback

        feedback = PromptFeedback(
            clarity=0.8,
            relevance=0.9,
            completeness=0.7,
        )

        assert feedback.clarity == 0.8
        assert feedback.relevance == 0.9
        assert feedback.completeness == 0.7
        # Score aggregato calcolato automaticamente
        assert 0.7 < feedback.prompt_quality_score < 0.9

    def test_default_values(self):
        """Test valori default."""
        from merlt.rlcf.prompt_policy import PromptFeedback

        feedback = PromptFeedback()

        assert feedback.clarity == 0.5
        assert feedback.relevance == 0.5
        assert feedback.completeness == 0.5

    def test_to_dict(self):
        """Test serializzazione."""
        from merlt.rlcf.prompt_policy import PromptFeedback

        feedback = PromptFeedback(clarity=0.8, relevance=0.9, completeness=0.7)
        d = feedback.to_dict()

        assert "clarity" in d
        assert "relevance" in d
        assert "completeness" in d
        assert "prompt_quality_score" in d


class TestPromptAction:
    """Test per PromptAction dataclass."""

    def test_creation(self):
        """Test creazione PromptAction."""
        from merlt.rlcf.prompt_policy import PromptAction

        action = PromptAction(
            expert_type="literal",
            prompt_version="1.0.0",
            log_prob=-0.5,
        )

        assert action.expert_type == "literal"
        assert action.prompt_version == "1.0.0"
        assert action.log_prob == -0.5
        assert action.modulation_vector is None

    def test_to_dict(self):
        """Test serializzazione."""
        from merlt.rlcf.prompt_policy import PromptAction

        action = PromptAction(
            expert_type="literal",
            prompt_version="1.0.0",
            modulation_vector=[0.1, 0.2, 0.3],
            log_prob=-0.5,
        )

        d = action.to_dict()

        assert d["action_type"] == "prompt_generation"
        assert d["expert_type"] == "literal"
        assert d["modulation_vector"] == [0.1, 0.2, 0.3]


class TestPromptPolicy:
    """Test per PromptPolicy network."""

    @pytest.fixture
    def policy(self):
        """Crea policy per test."""
        from merlt.rlcf.prompt_policy import PromptPolicy
        return PromptPolicy(input_dim=768, hidden_dim=256, prompt_dim=128)

    def test_initialization(self, policy):
        """Test inizializzazione."""
        assert policy.input_dim == 768
        assert policy.hidden_dim == 256
        assert policy.prompt_dim == 128
        assert policy.num_experts == 4

    def test_forward_shape(self, policy):
        """Test shape output forward."""
        batch_size = 4
        x = torch.randn(batch_size, 768)

        modulation, log_prob = policy(x)

        assert modulation.shape == (batch_size, 128)
        assert log_prob.shape == (batch_size,)

    def test_forward_deterministic(self, policy):
        """Test modalita' deterministica."""
        # Crea input e testa che stesso policy + stesso input = stesso output
        x = torch.randn(1, 768)

        # Eval mode per disabilitare dropout
        policy.eval()

        # Prima chiamata con deterministic=True
        with torch.no_grad():
            mod1, log_prob1 = policy(x, deterministic=True)
            # Seconda chiamata immediata con stesso input
            mod2, log_prob2 = policy(x, deterministic=True)

        # Con deterministic, log_prob dovrebbe essere 0
        assert log_prob1.item() == 0.0
        assert log_prob2.item() == 0.0

        # Output dovrebbe essere identico (stesso input, stessa policy, deterministic)
        assert torch.allclose(mod1, mod2, atol=1e-6)

        # Torna in train mode
        policy.train()

    def test_forward_stochastic(self, policy):
        """Test modalita' stocastica."""
        x = torch.randn(1, 768)

        # Con seed diversi, output diversi
        torch.manual_seed(42)
        mod1, _ = policy(x, deterministic=False)

        torch.manual_seed(123)
        mod2, _ = policy(x, deterministic=False)

        assert not torch.allclose(mod1, mod2)

    def test_expert_specific_projection(self, policy):
        """Test proiezione expert-specifica."""
        x = torch.randn(1, 768)

        mod0, _ = policy(x, expert_idx=0)
        mod1, _ = policy(x, expert_idx=1)

        # Expert diversi -> modulation diverse
        # (perche' passano per proiezioni diverse)
        assert not torch.allclose(mod0, mod1)

    def test_get_modulation(self, policy):
        """Test convenience method get_modulation."""
        x = torch.randn(768)  # Singolo embedding

        modulation, log_prob = policy.get_modulation(x, expert_type="literal")

        assert modulation.shape == (128,)
        assert isinstance(log_prob, float)

    def test_compute_loss(self, policy):
        """Test calcolo REINFORCE loss."""
        log_probs = torch.tensor([-0.5, -0.3, -0.8])
        rewards = torch.tensor([0.8, 0.9, 0.2])

        loss = policy.compute_loss(log_probs, rewards)

        assert loss.ndim == 0  # Scalare
        assert loss.requires_grad is False  # Advantages sono detached

    def test_compute_loss_with_baseline(self, policy):
        """Test loss con baseline."""
        log_probs = torch.tensor([-0.5, -0.3, -0.8])
        rewards = torch.tensor([0.8, 0.9, 0.2])
        baseline = torch.tensor([0.5, 0.5, 0.5])

        loss = policy.compute_loss(log_probs, rewards, baseline)

        assert loss.ndim == 0

    def test_save_load(self, policy):
        """Test save/load pesi."""
        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "policy.pt")

            # Salva
            policy.save(path)

            # Verifica file esiste
            assert Path(path).exists()

            # Carica in nuova policy
            from merlt.rlcf.prompt_policy import PromptPolicy
            new_policy = PromptPolicy(input_dim=768, hidden_dim=256, prompt_dim=128)
            new_policy.load(path)

            # Pesi dovrebbero essere uguali
            for p1, p2 in zip(policy.parameters(), new_policy.parameters()):
                assert torch.allclose(p1, p2)

    def test_trainable_parameters(self, policy):
        """Test che abbia parametri trainabili."""
        params = list(policy.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_device_handling(self):
        """Test gestione device."""
        from merlt.rlcf.prompt_policy import PromptPolicy

        policy = PromptPolicy(input_dim=64, hidden_dim=32, prompt_dim=16)
        x = torch.randn(1, 64)

        modulation, log_prob = policy(x)

        assert modulation.device == x.device


class TestPromptPolicyTrainer:
    """Test per PromptPolicyTrainer."""

    @pytest.fixture
    def trainer(self):
        """Crea trainer per test."""
        from merlt.rlcf.prompt_policy import PromptPolicy, PromptPolicyTrainer

        policy = PromptPolicy(input_dim=64, hidden_dim=32, prompt_dim=16)
        return PromptPolicyTrainer(policy, learning_rate=0.001)

    def test_initialization(self, trainer):
        """Test inizializzazione."""
        assert trainer.running_baseline == 0.0
        assert len(trainer.experience_buffer) == 0

    def test_record_experience(self, trainer):
        """Test registrazione esperienza."""
        query_emb = torch.randn(64)
        modulation = torch.randn(16)

        trainer.record_experience(
            query_embedding=query_emb,
            modulation=modulation,
            log_prob=-0.5,
            reward=0.8,
            expert_type="literal",
        )

        assert len(trainer.experience_buffer) == 1
        assert trainer.running_baseline > 0

    def test_update_with_insufficient_experiences(self, trainer):
        """Test che update non avvenga con poche esperienze."""
        # Aggiungi solo 3 esperienze
        for _ in range(3):
            trainer.record_experience(
                query_embedding=torch.randn(64),
                modulation=torch.randn(16),
                log_prob=-0.5,
                reward=0.8,
                expert_type="literal",
            )

        # Update richiede min 8
        result = trainer.update(min_experiences=8)

        assert result is None
        assert len(trainer.experience_buffer) == 3  # Non svuotato

    def test_update_with_sufficient_experiences(self, trainer):
        """Test update con esperienze sufficienti."""
        # Aggiungi 10 esperienze con log_probs reali
        for i in range(10):
            query_emb = torch.randn(64)
            modulation, log_prob = trainer.policy(query_emb.unsqueeze(0))

            trainer.record_experience(
                query_embedding=query_emb,
                modulation=modulation.squeeze(0),
                log_prob=log_prob.item(),  # Store scalar
                reward=0.5 + i * 0.05,
                expert_type="literal",
            )

        result = trainer.update(min_experiences=8)

        assert result is not None
        assert "loss" in result
        assert "policy_loss" in result
        assert "avg_reward" in result
        assert len(trainer.experience_buffer) == 0  # Buffer svuotato

    def test_save_load_checkpoint(self, trainer):
        """Test save/load checkpoint."""
        # Aggiungi esperienze con query_embeddings reali
        for _ in range(10):
            query_emb = torch.randn(64)
            modulation, log_prob = trainer.policy(query_emb.unsqueeze(0))

            trainer.record_experience(
                query_embedding=query_emb,
                modulation=modulation.squeeze(0),
                log_prob=log_prob.item(),
                reward=0.7,
                expert_type="literal",
            )
        trainer.update()

        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")

            # Salva
            original_baseline = trainer.running_baseline
            trainer.save_checkpoint(path)

            # Carica in nuovo trainer
            from merlt.rlcf.prompt_policy import PromptPolicy, PromptPolicyTrainer

            new_policy = PromptPolicy(input_dim=64, hidden_dim=32, prompt_dim=16)
            new_trainer = PromptPolicyTrainer(new_policy)
            new_trainer.load_checkpoint(path)

            assert new_trainer.running_baseline == original_baseline


class TestPromptPolicyGradients:
    """Test per gradients della policy."""

    def test_gradients_flow_through_modulation(self):
        """Test che gradients fluiscano attraverso il modulation output."""
        from merlt.rlcf.prompt_policy import PromptPolicy

        policy = PromptPolicy(input_dim=64, hidden_dim=32, prompt_dim=16)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

        x = torch.randn(4, 64)
        modulation, _ = policy(x, deterministic=False)

        # Loss su modulation - questo DEVE avere gradients
        loss = modulation.mean()

        optimizer.zero_grad()
        loss.backward()

        # Verifica che almeno alcuni parametri abbiano gradients non-zero
        total_grad = sum(
            p.grad.abs().sum().item()
            for p in policy.parameters()
            if p.grad is not None
        )

        assert total_grad > 0, "I parametri devono avere gradient non-zero"

    def test_policy_update_changes_output(self):
        """Test che un update cambi l'output della policy."""
        from merlt.rlcf.prompt_policy import PromptPolicy

        policy = PromptPolicy(input_dim=64, hidden_dim=32, prompt_dim=16)
        policy.eval()  # Deterministic output

        optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
        x = torch.randn(1, 64)

        # Output iniziale
        with torch.no_grad():
            mod_before, _ = policy(x, deterministic=True)

        # Update
        policy.train()
        mod, _ = policy(x, deterministic=False)
        loss = mod.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Output dopo update
        policy.eval()
        with torch.no_grad():
            mod_after, _ = policy(x, deterministic=True)

        # I due output devono essere diversi
        assert not torch.allclose(mod_before, mod_after, atol=1e-6)
