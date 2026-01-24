"""
Esempio: Integrazione GatingPolicy con MultiExpertOrchestrator
================================================================

Dimostra come usare GatingPolicy per neural routing nel sistema multi-expert.

Flow completo:
1. Crea GatingPolicy (o carica da checkpoint)
2. Setup MultiExpertOrchestrator con policy e embedding service
3. Process query → genera ExecutionTrace
4. Raccogli feedback utente
5. Update policy con PolicyGradientTrainer

Questo è il loop RLCF completo per apprendimento policy-based.
"""

import asyncio
from pathlib import Path

from merlt.experts.orchestrator import MultiExpertOrchestrator, OrchestratorConfig
from merlt.rlcf.policy_gradient import GatingPolicy, PolicyGradientTrainer
from merlt.rlcf.multilevel_feedback import MultilevelFeedback
from merlt.storage.vectors import EmbeddingService


async def main():
    """
    Esempio completo di utilizzo GatingPolicy.
    """

    # =============================================================================
    # STEP 1: Setup GatingPolicy
    # =============================================================================

    # Opzione A: Crea nuova policy (warm-start con priors)
    gating_policy = GatingPolicy(
        input_dim=768,  # Dimensione embedding (es. sentence-transformers)
        hidden_dim=256,
        num_experts=4,  # literal, systemic, principles, precedent
        device="cpu"  # Usa "cuda" se disponibile
    )

    # Opzione B: Carica policy esistente da checkpoint
    # gating_policy = GatingPolicy(input_dim=768, hidden_dim=256, num_experts=4)
    # trainer = PolicyGradientTrainer(gating_policy)
    # trainer.load_checkpoint("checkpoints/gating_policy_latest.pt")

    # =============================================================================
    # STEP 2: Setup EmbeddingService
    # =============================================================================

    # Necessario per encoding query → embedding vector
    embedding_service = EmbeddingService()
    await embedding_service.initialize()

    # =============================================================================
    # STEP 3: Setup MultiExpertOrchestrator con GatingPolicy
    # =============================================================================

    orchestrator = MultiExpertOrchestrator(
        gating_policy=gating_policy,
        embedding_service=embedding_service,
        config=OrchestratorConfig(
            selection_threshold=0.1,  # Threshold basso per neural (0.1-0.2)
            max_experts=4,
            parallel_execution=True
        )
    )

    print("=" * 80)
    print("MultiExpertOrchestrator con GatingPolicy attivo")
    print("=" * 80)

    # =============================================================================
    # STEP 4: Process Query con Neural Routing
    # =============================================================================

    query = "Cos'è la legittima difesa nel codice penale?"

    print(f"\nQuery: {query}")
    print("-" * 80)

    # Process con return_trace=True per ottenere ExecutionTrace
    response, trace = await orchestrator.process(
        query=query,
        return_trace=True  # ← Cruciale per RLCF
    )

    print(f"\nRisposta Sintetizzata:")
    print(response.synthesis)
    print(f"\nConfidence: {response.confidence:.2f}")
    print(f"Expert usati: {[r.expert_type for r in response.expert_responses]}")

    # =============================================================================
    # STEP 5: Analisi ExecutionTrace
    # =============================================================================

    print("\n" + "=" * 80)
    print("ExecutionTrace Analysis")
    print("=" * 80)

    print(f"\nTrace ID: {trace.query_id}")
    print(f"Num azioni: {trace.num_actions}")
    print(f"Total log prob: {trace.total_log_prob:.4f}")

    # Expert selections con log_prob
    expert_selections = trace.get_actions_by_type("expert_selection")
    print(f"\nExpert Selections ({len(expert_selections)}):")

    for action in expert_selections:
        expert_type = action.parameters["expert_type"]
        weight = action.parameters["weight"]
        log_prob = action.log_prob

        print(f"  - {expert_type:12s}: weight={weight:.4f}, log_prob={log_prob:.4f}")

    # =============================================================================
    # STEP 6: Simulazione Feedback Utente
    # =============================================================================

    print("\n" + "=" * 80)
    print("Feedback Utente (simulato)")
    print("=" * 80)

    # In produzione, feedback arriverebbe da interfaccia utente
    feedback = MultilevelFeedback(
        query_id=trace.query_id,
        query_text=query,
        response_text=response.synthesis
    )

    # Aggiungi rating (0-1 normalizzato)
    feedback.add_rating(
        level="overall",
        rating=0.85,  # Utente soddisfatto
        comment="Risposta completa e ben strutturata"
    )

    feedback.add_rating(
        level="correctness",
        rating=0.9,
        comment="Interpretazione giuridicamente corretta"
    )

    # Calcola overall score
    overall_score = feedback.overall_score()
    print(f"\nOverall Score: {overall_score:.2f}")

    # =============================================================================
    # STEP 7: Update Policy con REINFORCE
    # =============================================================================

    print("\n" + "=" * 80)
    print("Policy Update (REINFORCE)")
    print("=" * 80)

    # Crea trainer
    trainer = PolicyGradientTrainer(
        policy=gating_policy,
        # config=TrainerConfig(learning_rate=1e-4)
    )

    # Update da trace + feedback
    metrics = trainer.update_from_feedback(trace, feedback)

    print("\nUpdate Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Reward: {metrics['reward']:.4f}")
    print(f"  Baseline: {metrics['baseline']:.4f}")
    print(f"  Returns: {metrics['returns']:.4f}")
    print(f"  Num updates: {metrics['num_updates']}")

    # =============================================================================
    # STEP 8: Save Checkpoint
    # =============================================================================

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "gating_policy_example.pt"
    trainer.save_checkpoint(
        str(checkpoint_path),
        metadata={
            "example": "gating_policy_integration",
            "query": query,
            "overall_score": overall_score
        }
    )

    print(f"\nPolicy salvata in: {checkpoint_path}")

    # =============================================================================
    # STEP 9: Test Policy Aggiornata
    # =============================================================================

    print("\n" + "=" * 80)
    print("Test Policy Aggiornata")
    print("=" * 80)

    # Process nuova query
    new_query = "Responsabilità del debitore nell'adempimento"
    print(f"\nNuova query: {new_query}")

    response2, trace2 = await orchestrator.process(
        query=new_query,
        return_trace=True
    )

    # Mostra nuovi pesi
    expert_selections2 = trace2.get_actions_by_type("expert_selection")
    print(f"\nExpert Selections (dopo update):")

    for action in expert_selections2:
        expert_type = action.parameters["expert_type"]
        weight = action.parameters["weight"]
        print(f"  - {expert_type:12s}: weight={weight:.4f}")

    print("\n" + "=" * 80)
    print("Esempio completato!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
