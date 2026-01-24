"""
Test ExecutionTrace
===================

Test per modulo execution_trace del RLCF framework.
"""

import pytest
from datetime import datetime


class TestAction:
    """Test per Action dataclass."""

    def test_action_creation(self):
        """Test creazione Action."""
        from merlt.rlcf.execution_trace import Action

        action = Action(
            action_type="expert_selection",
            parameters={"expert": "literal", "weight": 0.7},
            log_prob=-0.357
        )

        assert action.action_type == "expert_selection"
        assert action.parameters["expert"] == "literal"
        assert action.parameters["weight"] == 0.7
        assert action.log_prob == -0.357
        assert action.timestamp is not None
        assert action.metadata == {}

    def test_action_with_metadata(self):
        """Test Action con metadata."""
        from merlt.rlcf.execution_trace import Action

        action = Action(
            action_type="tool_use",
            parameters={"tool_name": "citation_chain"},
            log_prob=-0.5,
            metadata={"context": "test", "step": 1}
        )

        assert action.metadata["context"] == "test"
        assert action.metadata["step"] == 1

    def test_action_to_dict(self):
        """Test serializzazione Action."""
        from merlt.rlcf.execution_trace import Action

        action = Action(
            action_type="graph_traversal",
            parameters={"relation": "RIFERIMENTO"},
            log_prob=-0.223,
            metadata={"source": "test"}
        )

        d = action.to_dict()

        assert d["action_type"] == "graph_traversal"
        assert d["parameters"]["relation"] == "RIFERIMENTO"
        assert d["log_prob"] == -0.223
        assert "timestamp" in d
        assert d["metadata"]["source"] == "test"

    def test_action_from_dict(self):
        """Test deserializzazione Action."""
        from merlt.rlcf.execution_trace import Action

        data = {
            "action_type": "expert_selection",
            "parameters": {"expert": "systemic"},
            "log_prob": -0.5,
            "timestamp": "2025-12-29T10:00:00",
            "metadata": {"test": True}
        }

        action = Action.from_dict(data)

        assert action.action_type == "expert_selection"
        assert action.parameters["expert"] == "systemic"
        assert action.log_prob == -0.5
        assert action.timestamp == "2025-12-29T10:00:00"
        assert action.metadata["test"] is True

    def test_action_from_dict_missing_optional(self):
        """Test deserializzazione senza campi opzionali."""
        from merlt.rlcf.execution_trace import Action

        data = {
            "action_type": "tool_use",
            "parameters": {},
            "log_prob": 0.0
        }

        action = Action.from_dict(data)

        assert action.action_type == "tool_use"
        assert action.timestamp is not None  # Default
        assert action.metadata == {}  # Default

    def test_action_roundtrip(self):
        """Test serializzazione/deserializzazione roundtrip."""
        from merlt.rlcf.execution_trace import Action

        original = Action(
            action_type="prompt_generation",
            parameters={"expert_type": "literal", "version": "1.0.0"},
            log_prob=-0.3,
            timestamp="2025-12-29T12:00:00",
            metadata={"test": "roundtrip"}
        )

        serialized = original.to_dict()
        restored = Action.from_dict(serialized)

        assert restored.action_type == original.action_type
        assert restored.parameters == original.parameters
        assert restored.log_prob == original.log_prob
        assert restored.timestamp == original.timestamp
        assert restored.metadata == original.metadata


class TestExecutionTrace:
    """Test per ExecutionTrace dataclass."""

    def test_trace_creation(self):
        """Test creazione ExecutionTrace."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")

        assert trace.query_id == "q001"
        assert trace.actions == []
        assert trace.total_log_prob == 0.0
        assert trace.metadata == {}
        assert trace.reward is None
        assert trace.created_at is not None

    def test_add_action(self):
        """Test aggiunta azione."""
        from merlt.rlcf.execution_trace import ExecutionTrace, Action

        trace = ExecutionTrace(query_id="q001")
        action = Action(
            action_type="expert_selection",
            parameters={"expert": "literal"},
            log_prob=-0.5
        )

        trace.add_action(action)

        assert len(trace.actions) == 1
        assert trace.total_log_prob == -0.5

    def test_add_multiple_actions(self):
        """Test aggiunta multiple azioni."""
        from merlt.rlcf.execution_trace import ExecutionTrace, Action

        trace = ExecutionTrace(query_id="q001")

        trace.add_action(Action("a1", {}, -0.3))
        trace.add_action(Action("a2", {}, -0.2))
        trace.add_action(Action("a3", {}, -0.5))

        assert len(trace.actions) == 3
        assert abs(trace.total_log_prob - (-1.0)) < 1e-10

    def test_add_expert_selection(self):
        """Test convenience method per expert selection."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection(
            expert_type="literal",
            weight=0.7,
            log_prob=-0.357,
            metadata={"confidence": 0.9}
        )

        assert len(trace.actions) == 1
        action = trace.actions[0]
        assert action.action_type == "expert_selection"
        assert action.parameters["expert_type"] == "literal"
        assert action.parameters["weight"] == 0.7
        assert action.log_prob == -0.357
        assert action.metadata["confidence"] == 0.9

    def test_add_graph_traversal(self):
        """Test convenience method per graph traversal."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_graph_traversal(
            relation_type="RIFERIMENTO",
            weight=0.8,
            log_prob=-0.223,
            source_node="art_1372_cc"
        )

        assert len(trace.actions) == 1
        action = trace.actions[0]
        assert action.action_type == "graph_traversal"
        assert action.parameters["relation_type"] == "RIFERIMENTO"
        assert action.parameters["weight"] == 0.8
        assert action.parameters["source_node"] == "art_1372_cc"

    def test_add_tool_use(self):
        """Test convenience method per tool use."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_tool_use(
            tool_name="citation_chain",
            parameters={"depth": 3, "source_urn": "urn:nir:..."},
            log_prob=-0.4
        )

        assert len(trace.actions) == 1
        action = trace.actions[0]
        assert action.action_type == "tool_use"
        assert action.parameters["tool_name"] == "citation_chain"
        assert action.parameters["tool_parameters"]["depth"] == 3

    def test_add_prompt_action(self):
        """Test convenience method per prompt generation."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_prompt_action(
            expert_type="literal",
            prompt_version="1.0.0",
            log_prob=-0.2,
            modulation_vector=[0.1, 0.2, 0.3]
        )

        assert len(trace.actions) == 1
        action = trace.actions[0]
        assert action.action_type == "prompt_generation"
        assert action.parameters["expert_type"] == "literal"
        assert action.parameters["prompt_version"] == "1.0.0"
        assert action.parameters["modulation_vector"] == [0.1, 0.2, 0.3]

    def test_set_reward(self):
        """Test impostazione reward."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.set_reward(0.85)

        assert trace.reward == 0.85
        assert trace.has_reward is True

    def test_get_actions_by_type(self):
        """Test filtro azioni per tipo."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection("literal", 0.7, -0.3)
        trace.add_graph_traversal("RIFERIMENTO", 0.8, -0.2)
        trace.add_expert_selection("systemic", 0.5, -0.4)
        trace.add_tool_use("citation_chain", {}, -0.1)

        expert_actions = trace.get_actions_by_type("expert_selection")
        graph_actions = trace.get_actions_by_type("graph_traversal")
        tool_actions = trace.get_actions_by_type("tool_use")

        assert len(expert_actions) == 2
        assert len(graph_actions) == 1
        assert len(tool_actions) == 1

    def test_get_actions_by_type_empty(self):
        """Test filtro con tipo non presente."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection("literal", 0.7, -0.3)

        result = trace.get_actions_by_type("nonexistent")
        assert result == []

    def test_num_actions_property(self):
        """Test property num_actions."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        assert trace.num_actions == 0

        trace.add_expert_selection("literal", 0.7, -0.3)
        assert trace.num_actions == 1

        trace.add_graph_traversal("REF", 0.5, -0.2)
        assert trace.num_actions == 2

    def test_has_reward_property(self):
        """Test property has_reward."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        assert trace.has_reward is False

        trace.set_reward(0.0)  # Anche 0.0 conta come reward impostato
        assert trace.has_reward is True

    def test_average_log_prob_property(self):
        """Test property average_log_prob."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")

        # Empty trace
        assert trace.average_log_prob == 0.0

        # Con azioni
        trace.add_expert_selection("a", 0.5, -0.3)
        trace.add_expert_selection("b", 0.5, -0.5)
        trace.add_expert_selection("c", 0.5, -0.2)

        # Total: -1.0, average: -1.0/3 ≈ -0.333
        assert abs(trace.average_log_prob - (-1.0 / 3)) < 1e-10

    def test_to_dict(self):
        """Test serializzazione ExecutionTrace."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001", metadata={"source": "test"})
        trace.add_expert_selection("literal", 0.7, -0.3)
        trace.set_reward(0.85)

        d = trace.to_dict()

        assert d["query_id"] == "q001"
        assert len(d["actions"]) == 1
        assert d["total_log_prob"] == -0.3
        assert d["metadata"]["source"] == "test"
        assert d["reward"] == 0.85
        assert "created_at" in d

    def test_from_dict(self):
        """Test deserializzazione ExecutionTrace."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        data = {
            "query_id": "q002",
            "actions": [
                {
                    "action_type": "expert_selection",
                    "parameters": {"expert_type": "literal"},
                    "log_prob": -0.5
                }
            ],
            "total_log_prob": -0.5,
            "metadata": {"test": True},
            "reward": 0.9,
            "created_at": "2025-12-29T10:00:00"
        }

        trace = ExecutionTrace.from_dict(data)

        assert trace.query_id == "q002"
        assert len(trace.actions) == 1
        assert trace.total_log_prob == -0.5
        assert trace.metadata["test"] is True
        assert trace.reward == 0.9
        assert trace.created_at == "2025-12-29T10:00:00"

    def test_from_dict_missing_optional(self):
        """Test deserializzazione con campi opzionali mancanti."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        data = {
            "query_id": "q003"
        }

        trace = ExecutionTrace.from_dict(data)

        assert trace.query_id == "q003"
        assert trace.actions == []
        assert trace.total_log_prob == 0.0
        assert trace.metadata == {}
        assert trace.reward is None

    def test_trace_roundtrip(self):
        """Test serializzazione/deserializzazione roundtrip."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        original = ExecutionTrace(query_id="q001", metadata={"key": "value"})
        original.add_expert_selection("literal", 0.7, -0.3, {"meta": "data"})
        original.add_graph_traversal("REF", 0.8, -0.2)
        original.set_reward(0.85)

        serialized = original.to_dict()
        restored = ExecutionTrace.from_dict(serialized)

        assert restored.query_id == original.query_id
        assert restored.num_actions == original.num_actions
        assert abs(restored.total_log_prob - original.total_log_prob) < 1e-10
        assert restored.reward == original.reward
        assert restored.metadata == original.metadata

    def test_summary(self):
        """Test metodo summary."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection("literal", 0.7, -0.3)
        trace.add_expert_selection("systemic", 0.5, -0.2)
        trace.add_graph_traversal("REF", 0.8, -0.5)
        trace.set_reward(0.85)

        summary = trace.summary()

        assert summary["query_id"] == "q001"
        assert summary["num_actions"] == 3
        assert summary["action_types"]["expert_selection"] == 2
        assert summary["action_types"]["graph_traversal"] == 1
        assert abs(summary["total_log_prob"] - (-1.0)) < 1e-10
        assert summary["has_reward"] is True
        assert summary["reward"] == 0.85


class TestMergeTraces:
    """Test per merge_traces utility."""

    def test_merge_empty_list(self):
        """Test merge di lista vuota."""
        from merlt.rlcf.execution_trace import merge_traces

        result = merge_traces([])

        assert result.query_id == "merged_empty"
        assert result.num_actions == 0

    def test_merge_single_trace(self):
        """Test merge di singolo trace."""
        from merlt.rlcf.execution_trace import ExecutionTrace, merge_traces

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection("literal", 0.7, -0.3)
        trace.set_reward(0.8)

        result = merge_traces([trace])

        assert result.query_id == "q001"
        assert result.num_actions == 1
        assert result.reward == 0.8

    def test_merge_multiple_traces(self):
        """Test merge di multiple traces."""
        from merlt.rlcf.execution_trace import ExecutionTrace, merge_traces

        trace1 = ExecutionTrace(query_id="q001")
        trace1.add_expert_selection("literal", 0.7, -0.3)

        trace2 = ExecutionTrace(query_id="q002")
        trace2.add_graph_traversal("REF", 0.8, -0.2)
        trace2.add_tool_use("cite", {}, -0.1)

        result = merge_traces([trace1, trace2])

        assert result.query_id == "q001"  # Primo query_id
        assert result.num_actions == 3
        assert abs(result.total_log_prob - (-0.6)) < 1e-10
        assert result.metadata["merged_from"] == 2

    def test_merge_with_rewards(self):
        """Test merge con reward (fa media)."""
        from merlt.rlcf.execution_trace import ExecutionTrace, merge_traces

        trace1 = ExecutionTrace(query_id="q001")
        trace1.set_reward(0.8)

        trace2 = ExecutionTrace(query_id="q002")
        trace2.set_reward(0.6)

        trace3 = ExecutionTrace(query_id="q003")
        # No reward

        result = merge_traces([trace1, trace2, trace3])

        # Media solo di trace con reward: (0.8 + 0.6) / 2 = 0.7
        assert abs(result.reward - 0.7) < 1e-10

    def test_merge_no_rewards(self):
        """Test merge senza reward."""
        from merlt.rlcf.execution_trace import ExecutionTrace, merge_traces

        trace1 = ExecutionTrace(query_id="q001")
        trace2 = ExecutionTrace(query_id="q002")

        result = merge_traces([trace1, trace2])

        assert result.reward is None


class TestComputeReturns:
    """Test per compute_returns utility."""

    def test_compute_returns_gamma_1(self):
        """Test returns con gamma=1 (no discount)."""
        from merlt.rlcf.execution_trace import ExecutionTrace, compute_returns

        traces = [
            ExecutionTrace(query_id="q1"),
            ExecutionTrace(query_id="q2"),
            ExecutionTrace(query_id="q3"),
        ]
        traces[0].set_reward(0.8)
        traces[1].set_reward(0.6)
        traces[2].set_reward(1.0)

        returns = compute_returns(traces, gamma=1.0)

        assert returns == [0.8, 0.6, 1.0]

    def test_compute_returns_gamma_less_than_1(self):
        """Test returns con gamma < 1 (discounted)."""
        from merlt.rlcf.execution_trace import ExecutionTrace, compute_returns

        traces = [
            ExecutionTrace(query_id="q1"),
            ExecutionTrace(query_id="q2"),
        ]
        traces[0].set_reward(1.0)
        traces[1].set_reward(0.5)

        returns = compute_returns(traces, gamma=0.9)

        assert abs(returns[0] - 0.9) < 1e-10
        assert abs(returns[1] - 0.45) < 1e-10

    def test_compute_returns_missing_reward(self):
        """Test returns con reward mancante."""
        from merlt.rlcf.execution_trace import ExecutionTrace, compute_returns

        traces = [
            ExecutionTrace(query_id="q1"),
            ExecutionTrace(query_id="q2"),
        ]
        traces[0].set_reward(0.8)
        # traces[1] senza reward

        returns = compute_returns(traces)

        assert returns[0] == 0.8
        assert returns[1] == 0.0  # Default per missing

    def test_compute_returns_empty_list(self):
        """Test returns con lista vuota."""
        from merlt.rlcf.execution_trace import compute_returns

        returns = compute_returns([])
        assert returns == []


class TestComputeBaseline:
    """Test per compute_baseline utility."""

    def test_baseline_mean(self):
        """Test baseline con metodo mean."""
        from merlt.rlcf.execution_trace import ExecutionTrace, compute_baseline

        traces = [
            ExecutionTrace(query_id="q1"),
            ExecutionTrace(query_id="q2"),
            ExecutionTrace(query_id="q3"),
        ]
        traces[0].set_reward(0.6)
        traces[1].set_reward(0.8)
        traces[2].set_reward(1.0)

        baseline = compute_baseline(traces, method="mean")

        # (0.6 + 0.8 + 1.0) / 3 = 0.8
        assert abs(baseline - 0.8) < 1e-10

    def test_baseline_median_odd(self):
        """Test baseline con metodo median (numero dispari)."""
        from merlt.rlcf.execution_trace import ExecutionTrace, compute_baseline

        traces = [
            ExecutionTrace(query_id="q1"),
            ExecutionTrace(query_id="q2"),
            ExecutionTrace(query_id="q3"),
        ]
        traces[0].set_reward(0.1)
        traces[1].set_reward(0.9)
        traces[2].set_reward(0.5)

        baseline = compute_baseline(traces, method="median")

        # Sorted: [0.1, 0.5, 0.9], median = 0.5
        assert abs(baseline - 0.5) < 1e-10

    def test_baseline_median_even(self):
        """Test baseline con metodo median (numero pari)."""
        from merlt.rlcf.execution_trace import ExecutionTrace, compute_baseline

        traces = [
            ExecutionTrace(query_id="q1"),
            ExecutionTrace(query_id="q2"),
            ExecutionTrace(query_id="q3"),
            ExecutionTrace(query_id="q4"),
        ]
        traces[0].set_reward(0.2)
        traces[1].set_reward(0.4)
        traces[2].set_reward(0.6)
        traces[3].set_reward(0.8)

        baseline = compute_baseline(traces, method="median")

        # Sorted: [0.2, 0.4, 0.6, 0.8], median = (0.4 + 0.6) / 2 = 0.5
        assert abs(baseline - 0.5) < 1e-10

    def test_baseline_empty(self):
        """Test baseline con lista vuota."""
        from merlt.rlcf.execution_trace import compute_baseline

        baseline = compute_baseline([])
        assert baseline == 0.0

    def test_baseline_no_rewards(self):
        """Test baseline senza reward."""
        from merlt.rlcf.execution_trace import ExecutionTrace, compute_baseline

        traces = [
            ExecutionTrace(query_id="q1"),
            ExecutionTrace(query_id="q2"),
        ]
        # Nessun reward impostato

        baseline = compute_baseline(traces)
        assert baseline == 0.0

    def test_baseline_unknown_method(self):
        """Test baseline con metodo sconosciuto (fallback a mean)."""
        from merlt.rlcf.execution_trace import ExecutionTrace, compute_baseline

        traces = [
            ExecutionTrace(query_id="q1"),
            ExecutionTrace(query_id="q2"),
        ]
        traces[0].set_reward(0.4)
        traces[1].set_reward(0.8)

        baseline = compute_baseline(traces, method="unknown")

        # Fallback a mean: (0.4 + 0.8) / 2 = 0.6
        assert abs(baseline - 0.6) < 1e-10


class TestExecutionTraceEdgeCases:
    """Test edge cases per ExecutionTrace."""

    def test_zero_log_prob(self):
        """Test azione con log_prob = 0."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection("literal", 1.0, 0.0)

        assert trace.total_log_prob == 0.0

    def test_negative_reward(self):
        """Test reward negativo (tecnicamente possibile)."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.set_reward(-0.5)

        assert trace.reward == -0.5
        assert trace.has_reward is True

    def test_very_large_trace(self):
        """Test trace con molte azioni."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")

        for i in range(1000):
            trace.add_expert_selection(f"expert_{i}", 0.5, -0.001)

        assert trace.num_actions == 1000
        assert abs(trace.total_log_prob - (-1.0)) < 1e-6

    def test_mixed_action_types(self):
        """Test trace con tutti i tipi di azione."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection("literal", 0.7, -0.1)
        trace.add_graph_traversal("REF", 0.8, -0.2)
        trace.add_tool_use("cite", {"depth": 2}, -0.15)
        trace.add_prompt_action("literal", "1.0.0", -0.05)

        summary = trace.summary()

        assert summary["action_types"]["expert_selection"] == 1
        assert summary["action_types"]["graph_traversal"] == 1
        assert summary["action_types"]["tool_use"] == 1
        assert summary["action_types"]["prompt_generation"] == 1

    def test_modify_after_serialization(self):
        """Test che modifica dopo serializzazione non influenzi originale."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection("literal", 0.7, -0.3)

        d = trace.to_dict()
        d["query_id"] = "modified"
        d["actions"].append({"action_type": "new", "parameters": {}, "log_prob": 0})

        # Originale non modificato
        assert trace.query_id == "q001"
        assert trace.num_actions == 1

    def test_add_action_without_metadata(self):
        """Test convenience methods senza metadata."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="q001")
        trace.add_expert_selection("literal", 0.7, -0.3)  # No metadata
        trace.add_graph_traversal("REF", 0.8, -0.2)  # No metadata
        trace.add_tool_use("cite", {}, -0.1)  # No metadata
        trace.add_prompt_action("literal", "1.0.0", -0.05)  # No metadata

        for action in trace.actions:
            assert action.metadata == {}
