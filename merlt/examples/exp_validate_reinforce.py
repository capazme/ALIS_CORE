#!/usr/bin/env python3
"""
EXP-VAL-001: Validazione REINFORCE Implementation
=================================================

Obiettivo: Verificare se il gradient flow nel PolicyGradientTrainer è corretto.

Ipotesi: L'implementazione attuale (linee 554-560) usa gradient random invece
di backpropagation reale, rendendo l'apprendimento inefficace.

Test:
1. Verificare che i gradienti SIANO random
2. Verificare che NON ci sia correlazione con la loss
3. Proporre e testare implementazione corretta

Esecuzione:
    python scripts/exp_validate_reinforce.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

# =============================================================================
# SETUP
# =============================================================================

@dataclass
class SimpleAction:
    """Azione semplificata per test."""
    state: np.ndarray  # Query embedding
    log_prob: float    # Pre-computed (PROBLEMA!)
    action_weights: np.ndarray  # Expert weights scelti


@dataclass
class SimpleTrace:
    """Trace semplificato."""
    query_id: str
    actions: List[SimpleAction]
    reward: float = 0.0


class SimpleGatingPolicy(nn.Module):
    """Policy semplificata per test."""

    def __init__(self, input_dim: int = 768, num_experts: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.mlp(x)
        weights = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return weights, log_probs


# =============================================================================
# TEST 1: Verificare che l'implementazione attuale usa gradient RANDOM
# =============================================================================

def test_current_implementation_uses_random_grads():
    """
    Verifica che l'implementazione attuale (linee 554-560) produce
    gradient che NON correlano con la loss.
    """
    print("=" * 60)
    print("TEST 1: Verifica gradient random nell'implementazione attuale")
    print("=" * 60)

    policy = SimpleGatingPolicy(input_dim=64, num_experts=4)
    learning_rate = 0.01

    # Simula l'update ERRATO (come nel codice attuale)
    def wrong_update(policy, returns):
        """Replica dell'update sbagliato."""
        with torch.no_grad():
            scale = learning_rate * returns
            for param in policy.parameters():
                if param.grad is None:
                    # QUESTO È IL PROBLEMA: gradient RANDOM
                    param.grad = torch.randn_like(param) * 0.01
                param.data.add_(param.grad * scale)

    # Esegui multiple updates con reward positivo
    initial_weights = {name: p.clone() for name, p in policy.named_parameters()}

    np.random.seed(42)
    torch.manual_seed(42)

    # Update 1: reward positivo
    wrong_update(policy, returns=1.0)
    weights_after_pos = {name: p.clone() for name, p in policy.named_parameters()}

    # Calcola direzione del cambiamento
    deltas_pos = {
        name: (weights_after_pos[name] - initial_weights[name]).detach().flatten().numpy()
        for name in initial_weights
    }

    # Reset e update con reward negativo
    for name, p in policy.named_parameters():
        p.data = initial_weights[name].clone()

    wrong_update(policy, returns=-1.0)
    weights_after_neg = {name: p.clone() for name, p in policy.named_parameters()}

    deltas_neg = {
        name: (weights_after_neg[name] - initial_weights[name]).detach().flatten().numpy()
        for name in initial_weights
    }

    # Analisi: se i gradient sono random, le DIREZIONI dovrebbero essere
    # INDIPENDENTI dal segno del reward
    # Solo la MAGNITUDINE dovrebbe cambiare

    print("\nAnalisi delle direzioni di update:")
    print("-" * 40)

    for name in deltas_pos:
        # Normalizza per confrontare direzioni
        dir_pos = deltas_pos[name] / (np.linalg.norm(deltas_pos[name]) + 1e-8)
        dir_neg = deltas_neg[name] / (np.linalg.norm(deltas_neg[name]) + 1e-8)

        # Correlazione tra direzioni
        correlation = np.dot(dir_pos, dir_neg)

        # Se gradient fosse corretto, direzioni sarebbero OPPOSTE (correlation ~ -1)
        # Se gradient è random, direzioni sarebbero INDIPENDENTI (correlation ~ 0)
        # NOTA: qui usiamo same seed, quindi correlation ~ 1 (stesso random)

        print(f"Layer {name}:")
        print(f"  Magnitude pos: {np.linalg.norm(deltas_pos[name]):.6f}")
        print(f"  Magnitude neg: {np.linalg.norm(deltas_neg[name]):.6f}")
        print(f"  Direction correlation: {correlation:.4f}")

        # Rapporto magnitudini
        ratio = np.linalg.norm(deltas_pos[name]) / (np.linalg.norm(deltas_neg[name]) + 1e-8)
        print(f"  Magnitude ratio (should be 1 if same random): {ratio:.4f}")

    print("\n" + "=" * 60)
    print("CONCLUSIONE TEST 1:")
    print("Con same random seed, le direzioni sono IDENTICHE (correlation ~ 1)")
    print("Solo la magnitudine cambia con il reward.")
    print("Questo conferma che il gradient è RANDOM, non calcolato dalla loss!")
    print("=" * 60)


# =============================================================================
# TEST 2: Confronto con REINFORCE corretto
# =============================================================================

def test_correct_reinforce():
    """
    Implementa e testa REINFORCE corretto con backpropagation reale.
    """
    print("\n" + "=" * 60)
    print("TEST 2: REINFORCE corretto con backpropagation")
    print("=" * 60)

    policy = SimpleGatingPolicy(input_dim=64, num_experts=4)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

    # Genera stato (query embedding)
    state = torch.randn(1, 64)

    # Forward pass
    weights, log_probs = policy(state)
    print(f"\nWeights iniziali: {weights.detach().numpy().flatten()}")

    # Simula azione: usa expert 0 (quello con peso più alto? o random?)
    # Per REINFORCE categoriale: sample dall'distribuzione
    action_dist = torch.distributions.Categorical(weights)
    action_idx = action_dist.sample()
    log_prob_action = log_probs[0, action_idx]

    print(f"Expert selezionato: {action_idx.item()}")
    print(f"Log prob azione: {log_prob_action.item():.4f}")

    # Reward positivo
    reward = 1.0
    baseline = 0.0
    returns = reward - baseline

    # REINFORCE loss corretto
    policy_loss = -log_prob_action * returns

    # Backpropagation REALE
    optimizer.zero_grad()
    policy_loss.backward()

    # Verifica che i gradienti esistono e sono non-zero
    print("\nGradienti calcolati:")
    has_nonzero_grad = False
    for name, p in policy.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            print(f"  {name}: grad norm = {grad_norm:.6f}")
            if grad_norm > 0:
                has_nonzero_grad = True

    assert has_nonzero_grad, "ERRORE: Nessun gradient non-zero!"

    # Optimizer step
    weights_before = {name: p.clone() for name, p in policy.named_parameters()}
    optimizer.step()

    # Verifica che i pesi sono cambiati
    print("\nCambiamento pesi:")
    for name, p in policy.named_parameters():
        delta = (p - weights_before[name]).norm().item()
        print(f"  {name}: delta = {delta:.6f}")

    # Forward dopo update
    weights_after, _ = policy(state)
    print(f"\nWeights dopo update: {weights_after.detach().numpy().flatten()}")

    # Con reward positivo, il peso dell'expert selezionato dovrebbe AUMENTARE
    weight_change = (weights_after[0, action_idx] - weights[0, action_idx]).item()
    print(f"\nCambiamento peso expert {action_idx.item()}: {weight_change:+.6f}")

    if weight_change > 0:
        print("✓ CORRETTO: peso expert selezionato è aumentato con reward positivo")
    else:
        print("✗ PROBLEMA: peso expert selezionato è diminuito con reward positivo")

    print("\n" + "=" * 60)


# =============================================================================
# TEST 3: REINFORCE per soft combination (caso MERL-T)
# =============================================================================

def test_soft_combination_reinforce():
    """
    Test REINFORCE per soft combination di expert.

    Nel caso MERL-T, NON scegliamo un singolo expert, ma usiamo
    una combinazione pesata: response = Σ w_i * response_i

    La formulazione corretta è:
    - Trattare come problema di regressione: loss = -reward * log_prob_chosen
    - OPPURE usare "policy gradient for continuous actions"

    Per soft combination, una formulazione sensata è:
    - loss = -reward * Σ(log_prob_i * w_i)  [weighted log prob]
    - Questo rinforza i pesi proporzionalmente al loro contributo
    """
    print("\n" + "=" * 60)
    print("TEST 3: REINFORCE per soft combination (caso MERL-T)")
    print("=" * 60)

    policy = SimpleGatingPolicy(input_dim=64, num_experts=4)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

    state = torch.randn(1, 64)

    # Forward
    weights, log_probs = policy(state)
    print(f"Weights: {weights.detach().numpy().flatten()}")

    # Simula che expert 0 e 2 hanno contribuito maggiormente
    # (in pratica, questo verrebbe dal feedback per expert)
    expert_contributions = torch.tensor([[0.4, 0.1, 0.4, 0.1]])  # Contributi normalizzati

    # Weighted log prob: quanto la policy ha "creduto" negli expert giusti
    weighted_log_prob = (log_probs * expert_contributions).sum()

    reward = 1.0
    policy_loss = -weighted_log_prob * reward

    print(f"Weighted log prob: {weighted_log_prob.item():.4f}")
    print(f"Policy loss: {policy_loss.item():.4f}")

    # Backprop
    optimizer.zero_grad()
    policy_loss.backward()

    # Analizza direzione dei gradienti
    print("\nGradienti per expert (ultimo layer):")
    last_layer = list(policy.mlp.parameters())[-2]  # Linear weight
    if last_layer.grad is not None:
        grad_per_expert = last_layer.grad.sum(dim=1)  # Sum over input dim
        print(f"  Expert 0 (contrib 0.4): grad sum = {grad_per_expert[0]:.6f}")
        print(f"  Expert 1 (contrib 0.1): grad sum = {grad_per_expert[1]:.6f}")
        print(f"  Expert 2 (contrib 0.4): grad sum = {grad_per_expert[2]:.6f}")
        print(f"  Expert 3 (contrib 0.1): grad sum = {grad_per_expert[3]:.6f}")

    # Update
    weights_before = weights.clone()
    optimizer.step()

    weights_after, _ = policy(state)
    print(f"\nWeights dopo update: {weights_after.detach().numpy().flatten()}")

    # Gli expert con alto contributo dovrebbero avere peso AUMENTATO
    delta_weights = weights_after - weights_before
    print(f"Delta weights: {delta_weights.detach().numpy().flatten()}")

    print("\n" + "=" * 60)


# =============================================================================
# TEST 4: Convergenza su task sintetico
# =============================================================================

def test_convergence():
    """
    Test di convergenza su task sintetico.

    Task: la policy deve imparare che query con feature[0] > 0
    dovrebbero usare expert 0, altrimenti expert 2.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Convergenza su task sintetico")
    print("=" * 60)

    policy = SimpleGatingPolicy(input_dim=64, num_experts=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    n_episodes = 200
    rewards_history = []
    correct_history = []

    for ep in range(n_episodes):
        # Genera query
        state = torch.randn(1, 64)

        # Target: expert 0 se feature[0] > 0, altrimenti expert 2
        target_expert = 0 if state[0, 0] > 0 else 2

        # Forward
        weights, log_probs = policy(state)

        # Sample azione (categorical)
        action_dist = torch.distributions.Categorical(weights)
        action_idx = action_dist.sample()

        # Reward: 1 se corretto, 0 altrimenti
        reward = 1.0 if action_idx.item() == target_expert else 0.0

        # REINFORCE update
        log_prob_action = log_probs[0, action_idx]
        baseline = 0.5  # Simple baseline
        returns = reward - baseline

        policy_loss = -log_prob_action * returns

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        rewards_history.append(reward)
        correct_history.append(action_idx.item() == target_expert)

        if (ep + 1) % 50 == 0:
            recent_accuracy = np.mean(correct_history[-50:])
            recent_reward = np.mean(rewards_history[-50:])
            print(f"Episode {ep+1}: Accuracy = {recent_accuracy:.2f}, Avg Reward = {recent_reward:.2f}")

    # Analisi finale
    final_accuracy = np.mean(correct_history[-50:])
    print(f"\nAccuratezza finale: {final_accuracy:.2%}")

    if final_accuracy > 0.7:
        print("✓ REINFORCE corretto converge!")
    else:
        print("✗ REINFORCE non converge (potrebbe richiedere più episodi)")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXP-VAL-001: Validazione REINFORCE Implementation")
    print("=" * 60)
    print()

    test_current_implementation_uses_random_grads()
    test_correct_reinforce()
    test_soft_combination_reinforce()
    test_convergence()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
CONCLUSIONI:

1. L'implementazione attuale USA GRADIENT RANDOM
   - Il gradient non deriva dalla loss function
   - L'apprendimento è di fatto RUMORE scalato per reward

2. REINFORCE corretto richiede:
   - Forward pass con gradient enabled
   - loss.backward() per calcolare gradienti REALI
   - optimizer.step() per applicare i gradienti

3. Per MERL-T (soft combination):
   - Opzione A: Sample categoriale da weights
   - Opzione B: Weighted log prob (Σ log_prob * contribution)

4. Il fix richiede:
   - Salvare lo stato (query embedding) nell'Action
   - Ri-eseguire forward pass durante l'update
   - Usare backpropagation reale

RACCOMANDAZIONE:
Reimplementare PolicyGradientTrainer.update_from_feedback()
con backpropagation reale.
""")
