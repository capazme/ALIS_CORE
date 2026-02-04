# Pipeline Trace Schema

## Flow Diagram

```
                                    QUERY
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. NER (Named Entity Recognition)                               [1.1ms]   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Input:  "Cos'è la risoluzione del contratto per inadempimento?"            │
│  Output: entities=[risoluzione:LEGAL_CONCEPT, inadempimento:LEGAL_CONCEPT]  │
│                                                                             │
│  Training Signal: Entity confidence scores (0.8 each)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. ROUTER (Query Classification)                                [0.1ms]   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Output: query_type=DEFINITION, primary_expert=literal, confidence=0.33     │
│                                                                             │
│  Training Signal: Routing accuracy (was literal the right choice?)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. RETRIEVER (Multi-Strategy)                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Strategy 1: Concept-based  →  Art. 1453, 1458 (via DISCIPLINA relations)   │
│  Strategy 2: Rubrica-based  →  Articles with "risoluzione" in title         │
│  Strategy 3: Text fallback  →  Keyword search in testo_vigente              │
│                                                                             │
│  Output: [Art.1453, Art.1458, Art.1749, Art.1750, Art.1751]                 │
│  Scoring: Multi-concept matching + rubrica boost                            │
│                                                                             │
│  Training Signal: relevance_boost scores, retrieval_method attribution      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│  4a. LITERAL EXPERT      [13.7s] │ │  4b. SYSTEMIC EXPERT      [36ms] │
│  ───────────────────────────────  │ │  ───────────────────────────────  │
│  Focus: Art. 12 disp. prel.       │ │  Focus: Graph relationships       │
│  Method: Text analysis + LLM      │ │  Method: Graph traversal + LLM    │
│                                   │ │                                   │
│  Retrieved: 5 norms + 5 defs      │ │  Traversed: 5 URNs                │
│  Confidence: 0.880 ✓              │ │  Relations found: 41              │
│                                   │ │  Confidence: 0.300 ⚠              │
│  Feedback Hook: F3                │ │  Feedback Hook: F4                │
│  - interpretation_quality         │ │  - graph_coverage                 │
│  - source_relevance               │ │  - systemic_insight               │
│  - confidence_calibration         │ │  - confidence_calibration         │
└───────────────────────────────────┘ └───────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. GATING NETWORK (Expert Aggregation)                         [13.3s]    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Method: weighted_average                                                   │
│                                                                             │
│  Expert Weights:                                                            │
│    literal:  0.75  ──►  weighted_confidence: 0.66                          │
│    systemic: 0.25  ──►  weighted_confidence: 0.075                         │
│                                                                             │
│  Combined Confidence: 0.735                                                 │
│  Conflicts: ["Divergenza significativa: literal (0.88) vs systemic (0.30)"]│
│                                                                             │
│  Feedback Hook: F7                                                          │
│  - weight_appropriateness                                                   │
│  - conflict_resolution                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. SYNTHESIZER                                                 [21.9s]    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Input: Expert interpretations + gating weights                             │
│  Profile: ricerca                                                           │
│                                                                             │
│  Output:                                                                    │
│    synthesis_mode: divergent                                                │
│    has_disagreement: true                                                   │
│    devils_advocate_flag: true                                               │
│    confidence_indicator: media (0.735)                                      │
│                                                                             │
│  Main Answer: "La risoluzione del contratto per inadempimento è un          │
│               rimedio giuridico che consente alla parte fedele di           │
│               sciogliere il vincolo contrattuale..."                        │
│                                                                             │
│  Sources Cited: Art. 1453, Art. 1458                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                                  RESPONSE
```

## Training Feedback Points

### Feedback Hook Types

| Hook | Stage | Purpose | Collectable Signals |
|------|-------|---------|---------------------|
| **F1** | NER | Entity recognition quality | entity_corrections, missed_entities, false_positives |
| **F2** | Router | Routing accuracy | correct_expert_choice, query_type_correction |
| **F3** | Literal Expert | Interpretation quality | interpretation_rating, source_relevance, confidence_accuracy |
| **F4** | Systemic Expert | Graph analysis quality | graph_coverage, systemic_insight, missing_connections |
| **F5** | Principles Expert | Ratio legis analysis | principle_relevance, constitutional_grounding |
| **F6** | Precedent Expert | Jurisprudence quality | precedent_relevance, case_applicability |
| **F7** | Gating | Aggregation quality | weight_corrections, conflict_resolution_feedback |
| **F8** | Synthesis | Final answer quality | answer_rating, completeness, clarity |

### Current Trace Feedback Status

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FEEDBACK HOOKS IN TRACE                          │
├──────────────────────────────────────────────────────────────────────┤
│  F3 (literal)   ✓ Present    correction_options: ✓ POPULATED        │
│    - interpretation_quality: [excellent, good, fair, poor]          │
│    - source_relevance: [all_relevant, mostly, some, mostly_irrel]   │
│    - confidence_calibration: [well_calibrated, over, under]         │
│    - textual_accuracy: [faithful, reasonable, stretched, incorrect] │
│    - missing_elements: [none, minor, key_articles, fundamental]     │
├──────────────────────────────────────────────────────────────────────┤
│  F4 (systemic)  ✓ Present    correction_options: ✓ POPULATED        │
│    - isolation_assessment: [correctly_isolated, false_isolation]    │
│    - graph_coverage: [incomplete, truly_isolated, search_failed]    │
│    - confidence_calibration: [well_calibrated, over, under]         │
├──────────────────────────────────────────────────────────────────────┤
│  F7 (gating)    ✓ Present    correction_options: ✓ POPULATED        │
│    - weight_appropriateness: [appropriate, *_overweight, *_under]   │
│    - conflict_detection: [correctly_identified, missed, false, no]  │
│    - aggregation_method: [appropriate, weighted, max, bayesian]     │
│    - combined_confidence: [well_calibrated, over, under]            │
└──────────────────────────────────────────────────────────────────────┘
```

### Context Snapshots (for training data)

Each feedback hook includes a `context_snapshot` with key information:

**F3 (Literal)**:
- query, sources_count, source_urns, confidence, confidence_factors, interpretation_preview

**F4 (Systemic)**:
- query, main_norm_count, isolated_norm, confidence, interpretation_preview

**F7 (Gating)**:
- expert_count, expert_confidences, weights_used, combined_confidence, conflicts

## Key Metrics for Training

### 1. Retrieval Quality
```
Retrieved Articles:  Art. 1453 ✓, Art. 1458 ✓, Art. 1749, Art. 1750, Art. 1751
Target Articles:     Art. 1453-1462 (risoluzione del contratto)
Precision:           2/5 = 40%  (2 correct out of 5 retrieved)
Recall:              2/10 = 20% (2 correct out of 10 target)

Training Signal: User can flag which retrieved articles were relevant
```

### 2. Expert Confidence Calibration
```
Expert          Confidence    Actual Quality (needs human label)
─────────────────────────────────────────────────────────────────
literal         0.88          [    ] Good  [    ] Overconfident
systemic        0.30          [    ] Good  [    ] Underconfident

Training Signal: Post-hoc confidence calibration
```

### 3. Synthesis Quality
```
Mode: divergent (experts disagreed)
Devil's Advocate: enabled

Quality Dimensions (need human labels):
  [ ] Completeness: Does it answer the question fully?
  [ ] Accuracy: Is the legal information correct?
  [ ] Citation Quality: Are sources properly cited?
  [ ] Clarity: Is the explanation clear?
```

## JSON Structure Summary

```json
{
  "metadata": {
    "generated_at": "ISO timestamp",
    "query": "original query",
    "user_profile": "ricerca|consulenza|analisi",
    "mode": "live_llm|mock",
    "success": true
  },

  "response": {
    "main_answer": "synthesized answer",
    "expert_accordion": [...],
    "source_links": [...],
    "confidence_indicator": "alta|media|bassa",
    "confidence_value": 0.735,
    "synthesis_mode": "convergent|divergent",
    "has_disagreement": true,
    "devils_advocate_flag": true
  },

  "trace": {
    "trace_id": "UUID",
    "ner_result": {...},
    "routing_decision": {...},
    "expert_executions": [...],
    "gating_result": {...},
    "synthesis_result": {...},
    "stage_times_ms": {...},
    "total_tokens": 2196
  },

  "metrics": {
    "total_time_ms": 48935,
    "experts_activated": ["literal", "systemic"],
    "experts_skipped": ["principles", "precedent"],
    "degraded": false
  },

  "feedback_hooks": [
    {"feedback_type": "F3", "expert_type": "literal", ...},
    {"feedback_type": "F4", "expert_type": "systemic", ...},
    {"feedback_type": "F7", "expert_type": "gating", ...}
  ]
}
```

## Identified Issues

### 1. Empty correction_options
The feedback hooks are present but `correction_options` dictionaries are empty. This needs to be populated in the expert implementations to enable training feedback collection.

### 2. Missing Experts
- `principles` and `precedent` experts were skipped (success=False)
- No F5/F6 feedback hooks generated

### 3. Low Systemic Confidence
- Systemic expert reports 0.30 confidence ("norma isolata")
- This may be correct (some norms are indeed isolated) or may indicate graph data quality issues

### 4. Retrieval Noise
- Art. 1749-1751 (about agents/representatives) are retrieved alongside the correct articles
- These match the concept "risoluzione del contratto per inadempimento" but are about specific contract types, not the general doctrine

---
*Generated: 2026-02-02*
