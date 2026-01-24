# RLCF Framework

> **Reinforcement Learning from Community Feedback**

---

## Overview

RLCF is a novel framework for improving AI systems through feedback from expert communities. Unlike RLHF (Reinforcement Learning from Human Feedback), RLCF:

1. **Weighs feedback by authority** - Expert feedback matters more than novice feedback
2. **Preserves uncertainty** - Doesn't force consensus where genuine uncertainty exists
3. **Uses constitutional governance** - Immutable principles guide the system
4. **Employs devil's advocates** - Deliberately challenges to avoid groupthink

---

## The Four Pillars

### 1. Dynamic Authority Scoring

Users accumulate authority based on demonstrated competence:

```python
from merlt.rlcf.authority import calculate_authority, AuthorityScore

score: AuthorityScore = calculate_authority(
    user_id="user123",
    domain="diritto_civile",
    feedback_history=history
)

# score.value: 0.0 - 1.0
# score.components:
#   - background: 0.25 (declared expertise, titles)
#   - consistency: 0.25 (coherent feedback over time)
#   - consensus: 0.30 (alignment with other experts)
#   - domain_expertise: 0.20 (domain-specific competence)
```

**Domain Authority:**

```python
from merlt.rlcf.domain_authority import DomainAuthorityService

service = DomainAuthorityService()
authority = await service.get_domain_authority(
    user_id=123,
    domain="obbligazioni"
)
# Returns domain-specific authority score
```

### 2. Uncertainty Preservation

The system maintains calibrated uncertainty:

```python
@dataclass
class ConfidenceFactors:
    norm_clarity: float           # How clear is the norm [0-1]
    jurisprudence_alignment: float # Agreement with case law [0-1]
    contextual_ambiguity: float   # Situational ambiguity [0-1]
    source_availability: float    # Available sources [0-1]
```

When experts disagree, the system reports this rather than forcing consensus:

```python
from merlt.disagreement import DisagreementDetector, DisagreementExplainer

detector = DisagreementDetector()
disagreement = detector.detect(expert_responses)

if disagreement.is_significant:
    explainer = DisagreementExplainer()
    explanation = explainer.explain(disagreement)
    # Returns: "Experts disagree on interpretation of Art. X..."
```

### 3. Constitutional Governance

Immutable principles that guide the system:

```python
# These principles CANNOT be modified by learning
CONSTITUTIONAL_PRINCIPLES = [
    "Never contradict explicit constitutional provisions",
    "Acknowledge uncertainty when sources conflict",
    "Cite primary sources before secondary",
    "Distinguish ratio decidendi from obiter dicta",
    "Flag potential unconstitutionality"
]
```

### 4. Devil's Advocate System

Deliberate challenge to avoid conformism:

```python
from merlt.rlcf.devils_advocate import DevilsAdvocateAssigner

assigner = DevilsAdvocateAssigner()
assignments = assigner.assign_advocates(
    task_id="task123",
    eligible_users=users,
    num_advocates=2
)

# Selected users are asked to argue AGAINST the majority position
# Their feedback is weighted to encourage critical thinking
```

---

## Policy Gradient Training

### REINFORCE for Gating/Traversal

Single-step policy learning for expert routing:

```python
from merlt.rlcf.policy_gradient import GatingPolicy, TraversalPolicy, PolicyGradientTrainer

# Gating Policy: Which experts to invoke
gating_policy = GatingPolicy(input_dim=768, num_experts=4)

# Traversal Policy: Which graph edges to follow
traversal_policy = TraversalPolicy(input_dim=768, num_relations=10)

# Training
trainer = PolicyGradientTrainer(gating_policy)
metrics = trainer.update_from_feedback(trace, feedback)
```

### SingleStepTrainer (Optimized REINFORCE)

For routing decisions:

```python
from merlt.rlcf.single_step_trainer import SingleStepTrainer, SingleStepConfig

config = SingleStepConfig(
    learning_rate=1e-4,
    entropy_coef=0.01,      # Encourage exploration
    baseline_ema=0.99       # Exponential moving average baseline
)

trainer = SingleStepTrainer(policy, config)
metrics = trainer.update(trace, feedback)
```

### ReActPPOTrainer (Multi-Step PPO)

For complex expert reasoning:

```python
from merlt.rlcf.react_ppo_trainer import ReActPPOTrainer, ReActConfig, ReActPolicy

policy = ReActPolicy(
    state_dim=1024,
    num_actions=7           # 7 possible actions in ReAct loop
)

config = ReActConfig(
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE lambda
    clip_ratio=0.2,         # PPO clip
    value_coef=0.5,
    entropy_coef=0.01
)

trainer = ReActPPOTrainer(policy, config)

# Collect trajectory during expert reasoning
trainer.add_trajectory(trajectory)

# Update after batch
metrics = trainer.update()
```

---

## Execution Tracing

Every query creates an execution trace for feedback:

```python
from merlt.rlcf.execution_trace import ExecutionTrace, Action

trace = ExecutionTrace(
    query_id="q123",
    metadata={"expert_type": "literal", "query_text": "..."}
)

# Record actions during execution
trace.record_action(Action(
    action_type="tool_call",
    tool_name="semantic_search",
    parameters={"query": "...", "top_k": 5},
    result={"found": 5},
    duration_ms=150
))

# Finalize with response
trace.finalize(response, execution_time_ms=2500)
```

---

## Feedback Types

### MultilevelFeedback

Structured feedback at multiple levels:

```python
from merlt.rlcf.multilevel_feedback import MultilevelFeedback, FeedbackLevel

feedback = MultilevelFeedback(
    query_level=0.8,              # Overall response quality
    expert_level={
        "literal": 0.9,
        "systemic": 0.7,
        "principles": 0.8,
        "precedent": 0.6
    },
    source_level={
        "urn:nir:stato:legge:...": 1.0,   # Highly relevant
        "urn:nir:stato:decreto:...": 0.3   # Less relevant
    },
    step_level=[0.9, 0.8, 0.7, 0.9]  # Per reasoning step
)
```

### Entity Feedback

Feedback on extracted entities:

```python
from merlt.rlcf.entity_feedback import EntityFeedbackCollector

collector = EntityFeedbackCollector()
collector.collect_feedback(
    entity="Art. 1453 c.c.",
    entity_type="norm_reference",
    correct=True,
    user_correction=None
)
```

---

## Advanced Features

### Experience Replay Buffer

For sample-efficient learning:

```python
from merlt.rlcf.replay_buffer import PrioritizedReplayBuffer

buffer = PrioritizedReplayBuffer(
    capacity=10000,
    alpha=0.6               # Prioritization exponent
)

# Add experience with priority
buffer.add(
    trace=trace,
    feedback=feedback,
    reward=0.8,
    td_error=0.5            # For prioritization
)

# Sample weighted batch
batch, indices, weights = buffer.sample_with_priority(batch_size=32)

# Update priorities after learning
buffer.update_priorities(indices, new_td_errors)
```

### Curriculum Learning

Progressive difficulty training:

```python
from merlt.rlcf.curriculum_learning import CurriculumScheduler, DifficultyAssessor

scheduler = CurriculumScheduler()

# Assess query difficulty
assessment = scheduler.assess_difficulty(
    "Cos'e' la legittima difesa?"
)
# assessment.difficulty: 0.3 (relatively easy)

# Filter training batch by current curriculum
filtered_batch = scheduler.filter_batch_by_curriculum(
    queries=all_queries,
    target_size=32
)

# Advance curriculum after successful epoch
scheduler.update_after_epoch(avg_reward=0.75)
```

### Off-Policy Evaluation

Evaluate new policies before deployment:

```python
from merlt.rlcf.off_policy_eval import OPEEvaluator

evaluator = OPEEvaluator()

result = evaluator.evaluate(
    new_policy=candidate_policy,
    historical_data=logged_data,
    method="doubly_robust"      # or "importance_sampling", "direct"
)

if result.estimated_value > current_value:
    # Safe to deploy
    deploy(candidate_policy)
```

### Bias Detection

6-dimensional bias analysis:

```python
from merlt.rlcf.bias_detection import BiasDetector, BiasDimension

detector = BiasDetector()

report = await detector.calculate_total_bias(
    task_id="task123",
    feedbacks=feedback_list
)

# report.dimensions:
#   - AUTHORITY_SKEW: Are high-authority users biased?
#   - TEMPORAL: Early vs late feedback differences
#   - DOMAIN: Domain-specific biases
#   - POSITION: First-presented vs later options
#   - CONFIRMATION: Agreement with prior answers
#   - ANCHORING: Influence of initial ratings

if report.total_bias > 0.3:
    # Apply debiasing
    debiased = detector.debias(feedbacks)
```

---

## Persistence

PostgreSQL storage for RLCF data:

```python
from merlt.rlcf.persistence import RLCFPersistence, RLCFTrace, RLCFFeedback

persistence = RLCFPersistence(session)

# Save trace
await persistence.save_trace(trace)

# Save feedback
await persistence.save_feedback(feedback, trace_id)

# Load policy checkpoint
checkpoint = await persistence.load_latest_checkpoint(policy_name="gating")
```

---

## Training Scheduler

Automated training orchestration:

```python
from merlt.rlcf.training_scheduler import TrainingScheduler, get_scheduler

scheduler = get_scheduler()

# Configure training job
scheduler.configure(
    policy_type="gating",
    batch_size=32,
    epochs=10,
    eval_interval=100
)

# Start async training
task = scheduler.start_training()

# Check status
status = scheduler.get_status()
# status.current_epoch, status.metrics, etc.
```

---

## File Structure

```
merlt/merlt/rlcf/
├── __init__.py                 # Public exports, lazy loading
├── ai_service.py               # OpenRouter/LLM integration
├── aggregation.py              # Feedback aggregation engine
├── authority.py                # Authority calculation
├── authority_sync.py           # Sync with VisuaLex
├── bias_detection.py           # 6-dimensional bias detection
├── config.py                   # RLCF configuration
├── curriculum_learning.py      # Curriculum scheduler
├── database.py                 # Database session management
├── devils_advocate.py          # Critical thinking system
├── domain_authority.py         # Domain-specific authority
├── edit_merge.py               # Edit/merge operations
├── entity_feedback.py          # Entity-level feedback
├── execution_trace.py          # Execution tracing
├── external_feedback.py        # External system integration
├── metrics.py                  # Metrics tracking
├── models.py                   # SQLAlchemy models
├── multilevel_feedback.py      # Multi-level feedback
├── ner_feedback_buffer.py      # NER feedback buffer
├── ner_rlcf_integration.py     # NER integration
├── off_policy_eval.py          # Off-policy evaluation
├── orchestrator.py             # Main RLCF orchestrator
├── persistence.py              # PostgreSQL persistence
├── policy_gradient.py          # REINFORCE implementation
├── policy_manager.py           # Policy management
├── ppo_trainer.py              # PPO trainer (legacy)
├── prompt_optimizer.py         # Prompt optimization
├── prompt_policy.py            # Prompt policy learning
├── react_ppo_trainer.py        # ReAct PPO trainer
├── replay_buffer.py            # Experience replay
├── single_step_trainer.py      # Optimized REINFORCE
├── training_scheduler.py       # Training orchestration
├── simulator/                  # Feedback simulation
│   ├── config.py
│   ├── experiment.py
│   ├── feedback_synthesizer.py
│   ├── integration.py
│   ├── llm_judge.py
│   ├── objective_metrics.py
│   ├── outputs.py
│   ├── statistics.py
│   └── users.py
├── task_handlers/              # Task-specific handlers
│   ├── base.py
│   ├── classification.py
│   ├── qa.py
│   └── retrieval.py
└── validation/                 # Validation models
    ├── __init__.py
    └── models.py
```

---

## Integration with VisuaLex

External feedback from the platform:

```python
from merlt.rlcf.external_feedback import ExternalFeedbackAdapter

adapter = ExternalFeedbackAdapter()

# Receive feedback from VisuaLex Platform
await adapter.receive_feedback(
    source="visualex_platform",
    user_id=123,
    trace_id="trace_abc",
    rating=0.85,
    feedback_type="utility"
)
```

Authority synchronization:

```python
from merlt.rlcf.authority_sync import AuthoritySyncService

sync_service = AuthoritySyncService()

# Sync user authority scores between systems
await sync_service.sync_user_authority(
    user_id=123,
    visualex_session=visualex_db
)
```
