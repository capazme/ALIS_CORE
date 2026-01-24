# MERL-T Expert System

> **Multi-Expert Legal Retrieval Transformer**

---

## Overview

The MERL-T Expert System implements the hermeneutic canons of Italian law (Art. 12-14 Preleggi) as specialized AI agents. Each expert uses different tools and traversal strategies to analyze legal questions from their unique perspective.

---

## Expert Types

### 1. LiteralExpert

**Canon:** "significato proprio delle parole" (Art. 12, I)

**Focus:** Textual analysis, legal definitions, literal meaning

**Tools:**
- `semantic_search` - Vector search for definitions
- `definition_lookup` - Legal terminology database

**Traversal Weights:**
```yaml
DEFINISCE: 1.0      # Definitions
CONTIENE: 0.8       # Contains
RELATED_TO: 0.5     # Related concepts
```

**Source Types:** `["definition", "norma", "doctrine"]`

---

### 2. SystemicExpert

**Canon:** "connessione di esse" (Art. 12, I) + Art. 14 (historical)

**Focus:** Normative context, relationships between norms, system coherence

**Tools:**
- `graph_search` - Knowledge graph traversal
- `semantic_search` - Context retrieval
- `norm_hierarchy` - Hierarchical lookup

**Traversal Weights:**
```yaml
RIFERIMENTO: 1.0    # References
MODIFICA: 0.9       # Modifications
DEROGA: 0.8         # Derogations
ABROGA: 0.7         # Abrogations
CITATO_DA: 0.6      # Cited by
```

**Source Types:** `["norma", "modifica", "abrogazione"]`

---

### 3. PrinciplesExpert

**Canon:** "intenzione del legislatore" (Art. 12, II)

**Focus:** Legislative intent, constitutional principles, ratio legis

**Tools:**
- `constitutional_search` - Constitutional provisions
- `travaux_preparatoires` - Legislative history
- `principle_extraction` - Core principle identification

**Traversal Weights:**
```yaml
ATTUA: 1.0          # Implements
PRINCIPIO: 0.9      # Principle relations
DEROGA: 0.7         # Derogations
COSTITUZIONALE: 0.8 # Constitutional
```

**Source Types:** `["constitutional", "principle", "travaux"]`

---

### 4. PrecedentExpert

**Canon:** Jurisprudential practice

**Focus:** Case law, judicial interpretations, precedents

**Tools:**
- `case_law_search` - Jurisprudence database
- `semantic_search` - Similar cases
- `citation_network` - Citation analysis

**Traversal Weights:**
```yaml
CITATO_DA: 1.0      # Cited by
APPLICA: 0.9        # Applies
INTERPRETA: 0.8     # Interprets
CONFERMA: 0.7       # Confirms
OVERRULE: 0.5       # Overrules
```

**Source Types:** `["jurisprudence", "massima", "sentenza"]`

---

## Core Data Structures

### ExpertContext

Input to all experts:

```python
@dataclass
class ExpertContext:
    query_text: str                              # Original question
    query_embedding: Optional[List[float]]       # Vector embedding
    entities: Dict[str, List[str]]               # Extracted entities
    retrieved_chunks: List[Dict[str, Any]]       # Pre-retrieved content
    metadata: Dict[str, Any]                     # Additional metadata
    trace_id: str                                # Tracing ID

    @property
    def norm_references(self) -> List[str]:
        """Extracted normative references (URNs)."""

    @property
    def legal_concepts(self) -> List[str]:
        """Extracted legal concepts."""
```

### ExpertResponse

Output from all experts:

```python
@dataclass
class ExpertResponse:
    expert_type: str                             # literal, systemic, principles, precedent
    interpretation: str                          # Main interpretation (Italian)
    legal_basis: List[LegalSource]               # Cited sources with provenance
    reasoning_steps: List[ReasoningStep]         # Reasoning trace
    confidence: float                            # [0-1] confidence score
    confidence_factors: ConfidenceFactors        # Breakdown
    limitations: str                             # What couldn't be considered
    trace_id: str                                # For RLCF tracking
    execution_time_ms: float                     # Performance
    tokens_used: int                             # LLM token usage
    metadata: Dict[str, Any]                     # react_metrics, etc.
```

### LegalSource

Provenance tracking for cited sources:

```python
@dataclass
class LegalSource:
    source_type: str    # norm, jurisprudence, doctrine, constitutional
    source_id: str      # URN or unique ID
    citation: str       # Formal citation (e.g., "Art. 1321 c.c.")
    excerpt: str        # Relevant excerpt
    relevance: str      # Why this source is relevant
```

---

## Orchestration

### MultiExpertOrchestrator

Coordinates all experts for a single query:

```python
from merlt.experts import MultiExpertOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    parallel=True,              # Run experts in parallel
    timeout_seconds=30,         # Per-expert timeout
    min_experts=2,              # Minimum experts to run
    synthesis_mode="adaptive"   # How to combine responses
)

orchestrator = MultiExpertOrchestrator(
    experts=[literal, systemic, principles, precedent],
    gating_network=gating,
    synthesizer=synthesizer,
    config=config
)

result = await orchestrator.analyze(context)
```

### ExpertRouter

Determines which experts to invoke based on query characteristics:

```python
from merlt.experts import ExpertRouter, RoutingDecision

router = ExpertRouter()
decision: RoutingDecision = router.route(context)

# decision.experts: List of experts to invoke
# decision.weights: Initial weights for each expert
# decision.reasoning: Why these experts were chosen
```

### GatingNetwork

Combines expert responses using learned weights:

```python
from merlt.experts import GatingNetwork, AggregatedResponse

gating = GatingNetwork()
aggregated: AggregatedResponse = gating.aggregate(
    responses=[response1, response2, response3, response4],
    context=context
)

# aggregated.final_interpretation: Combined answer
# aggregated.expert_weights: How much each expert contributed
# aggregated.agreement_score: Inter-expert agreement [0-1]
```

---

## Neural Gating (Advanced)

Optional PyTorch-based gating for learned expert routing:

```python
from merlt.experts.neural_gating import (
    HybridExpertRouter,
    ExpertGatingMLP,
    NeuralGatingTrainer,
    GatingConfig
)

# Create neural router
config = GatingConfig(
    input_dim=768,              # Embedding dimension
    hidden_dim=256,
    num_experts=4,
    dropout=0.1
)

gating_mlp = ExpertGatingMLP(config)
router = HybridExpertRouter(
    static_router=ExpertRouter(),
    neural_gating=gating_mlp,
    blend_factor=0.7            # 70% neural, 30% static
)

# Training
trainer = NeuralGatingTrainer(gating_mlp)
trainer.train_step(query_embeddings, expert_rewards)
```

---

## ReAct Pattern

Experts can use ReAct (Reasoning + Acting) for multi-step analysis:

```python
from merlt.experts import ReActMixin, ReActResult, ThoughtActionObservation

class LiteralExpert(BaseExpert, ReActMixin):
    async def analyze(self, context: ExpertContext) -> ExpertResponse:
        # ReAct loop: Think -> Act -> Observe -> Repeat
        result: ReActResult = await self.react_loop(context, max_steps=5)
        return self._build_response(result)
```

**ReAct Steps:**
1. **Thought**: LLM reasons about what to do
2. **Action**: Execute a tool (semantic_search, graph_search, etc.)
3. **Observation**: Process tool result
4. **Repeat** until answer is complete

---

## Adaptive Synthesizer

Combines expert responses intelligently:

```python
from merlt.experts import AdaptiveSynthesizer, SynthesisMode, SynthesisConfig

config = SynthesisConfig(
    mode=SynthesisMode.WEIGHTED_CONSENSUS,  # or EXPERT_SELECTED, UNANIMOUS, etc.
    min_agreement=0.6,
    include_dissent=True
)

synthesizer = AdaptiveSynthesizer(config)
result = await synthesizer.synthesize(responses, context)
```

**Synthesis Modes:**
- `WEIGHTED_CONSENSUS`: Weight by confidence and agreement
- `EXPERT_SELECTED`: Best single expert
- `UNANIMOUS`: Only where all agree
- `ENSEMBLE`: Include all perspectives

---

## RLCF Integration

Experts track execution for feedback learning:

```python
# Initialize trace at query start
expert._init_trace(context)

# After analysis
trace = expert.get_current_trace()
exploration_metrics = expert.get_exploration_metrics()

# Record user feedback
feedback_record = await expert.record_feedback(
    response=response,
    user_rating=0.8,
    feedback_type="accuracy",
    rlcf_orchestrator=rlcf_orchestrator,
    user_id=user.id
)

# Traversal weight updates suggested
weight_updates = feedback_record["weight_update_suggestions"]
```

---

## File Structure

```
merlt/merlt/experts/
├── __init__.py              # Public exports
├── base.py                  # BaseExpert, ExpertResponse, etc. (1000+ LOC)
├── literal.py               # LiteralExpert implementation
├── systemic.py              # SystemicExpert implementation
├── principles.py            # PrinciplesExpert implementation
├── precedent.py             # PrecedentExpert implementation
├── router.py                # ExpertRouter
├── gating.py                # GatingNetwork
├── orchestrator.py          # MultiExpertOrchestrator
├── synthesizer.py           # AdaptiveSynthesizer
├── react_mixin.py           # ReAct pattern mixin
├── query_analyzer.py        # Query classification
├── prompt_loader.py         # YAML prompt loading
├── policy_metrics.py        # Policy performance metrics
├── models.py                # Pydantic models
├── config/
│   └── experts.yaml         # Expert configuration
├── prompts/
│   ├── literal.yaml         # LiteralExpert prompts
│   ├── systemic.yaml
│   ├── principles.yaml
│   └── precedent.yaml
└── neural_gating/
    ├── __init__.py
    ├── neural.py            # ExpertGatingMLP
    └── hybrid_router.py     # HybridExpertRouter
```

---

## Best Practices

1. **Never modify** `base.py` signatures without updating all experts
2. **Always trace** execution for RLCF feedback
3. **Use prompts from YAML**, not hardcoded strings
4. **Test with mocked tools** before integration testing
5. **Log with structlog** for consistent tracing
