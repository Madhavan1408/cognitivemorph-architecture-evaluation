"""
CognitiveMorph Architecture
Hybrid: Neuromorphic Encoding + Symbolic Reasoning + Agentic Planning
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────
# LAYER 1: Neuromorphic Encoding (Mock Spiking)
# ─────────────────────────────────────────────

class NeuromorphicEncoder:
    """
    Simulates spike-timing-dependent plasticity (STDP).
    Converts continuous sensor signals into spike trains.
    """
    def __init__(self, n_neurons: int = 64, threshold: float = 0.5):
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.weights = np.random.normal(0.5, 0.15, (n_neurons, 8))  # 8 input features
        self.membrane_potential = np.zeros(n_neurons)

    def encode(self, sensor_input: np.ndarray) -> np.ndarray:
        """Convert sensor readings to spike representation."""
        normalized = (sensor_input - sensor_input.min()) / (np.ptp(sensor_input) + 1e-9)
        activation = self.weights @ normalized
        spikes = (activation > self.threshold).astype(float)
        # Temporal decay
        self.membrane_potential = 0.9 * self.membrane_potential + activation
        return spikes + 0.1 * np.random.randn(self.n_neurons)

    def update_weights(self, spikes: np.ndarray, reward: float):
        """Hebbian-style STDP weight update."""
        delta = 0.01 * reward * np.outer(spikes, np.ones(8))
        self.weights = np.clip(self.weights + delta[:, :8], 0, 1)


# ─────────────────────────────────────────────
# LAYER 2: Symbolic Reasoning Engine
# ─────────────────────────────────────────────

@dataclass
class Rule:
    condition: str
    action: str
    confidence: float


class SymbolicReasoner:
    """
    Rule-based symbolic reasoning over encoded state.
    Provides interpretable decision boundaries.
    """
    def __init__(self):
        self.rules: List[Rule] = [
            Rule("temperature > 0.8",         "reduce_load",         0.95),
            Rule("vibration > 0.7",            "maintenance_alert",   0.90),
            Rule("relational > 0.7",           "activate_collab",     0.88),
            Rule("complexity < 0.3",           "fast_path",           0.92),
            Rule("complexity > 0.8",           "decompose_task",      0.85),
            Rule("energy > 0.9",               "efficiency_mode",     0.87),
            Rule("human_level > 0.7",          "handoff_human",       0.93),
            Rule("adaptation_needed",          "trigger_plasticity",  0.91),
        ]
        self.fired_rules = []

    def reason(self, state: Dict) -> Tuple[List[str], float]:
        """Evaluate rules against current state, return actions + confidence."""
        actions = []
        total_confidence = 0.0
        self.fired_rules = []

        temp_norm   = state.get("temperature", 0) / 180.0
        vib_norm    = state.get("vibration", 0) / 2.0
        complexity  = state.get("complexity", 0.5)
        relational  = state.get("relational", 0.5)
        human_level = state.get("human_level", 0.5)
        energy_norm = state.get("energy", 0) / 300.0

        rule_checks = {
            "temperature > 0.8":    temp_norm > 0.8,
            "vibration > 0.7":      vib_norm > 0.7,
            "relational > 0.7":     relational > 0.7,
            "complexity < 0.3":     complexity < 0.3,
            "complexity > 0.8":     complexity > 0.8,
            "energy > 0.9":         energy_norm > 0.9,
            "human_level > 0.7":    human_level > 0.7,
            "adaptation_needed":    complexity > 0.6 and relational > 0.5,
        }

        for rule in self.rules:
            if rule_checks.get(rule.condition, False):
                actions.append(rule.action)
                total_confidence += rule.confidence
                self.fired_rules.append(rule)

        avg_conf = (total_confidence / len(self.fired_rules)) if self.fired_rules else 0.5
        return actions, avg_conf


# ─────────────────────────────────────────────
# LAYER 3: Agentic Planning Loop
# ─────────────────────────────────────────────

class AgenticPlanner:
    """
    Goal-directed planning with lookahead and memory.
    Integrates neuromorphic + symbolic outputs into decisions.
    """
    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self.memory: List[Dict] = []
        self.goal_stack: List[str] = []
        self.performance_history = []

    def plan(self, spikes: np.ndarray, symbolic_actions: List[str],
             symbolic_conf: float, state: Dict) -> Dict:
        """Generate action plan given current perceptions and rules."""
        # Extract signal strength from spikes
        spike_energy = np.mean(spikes)
        spike_variance = np.var(spikes)

        # Retrieve relevant memories
        relevant_mem = self._retrieve_memory(state)

        # Compute action score (higher = more deliberate action needed)
        action_urgency = (
            spike_energy * 0.4 +
            symbolic_conf * 0.4 +
            (len(symbolic_actions) / 8.0) * 0.2
        )

        # Planning output
        plan = {
            "primary_action":   symbolic_actions[0] if symbolic_actions else "continue",
            "urgency":          round(action_urgency, 4),
            "spike_energy":     round(spike_energy, 4),
            "memory_context":   len(relevant_mem),
            "confidence":       round(symbolic_conf * (0.8 + 0.2 * spike_energy), 4),
            "lookahead_steps":  min(self.horizon, max(1, int(action_urgency * 5))),
        }

        self._store_memory(state, plan)
        self.performance_history.append(plan["confidence"])
        return plan

    def _retrieve_memory(self, state: Dict) -> List[Dict]:
        """Simple episodic retrieval by complexity similarity."""
        current_c = state.get("complexity", 0.5)
        return [m for m in self.memory[-20:]
                if abs(m.get("complexity", 0.5) - current_c) < 0.15]

    def _store_memory(self, state: Dict, plan: Dict):
        self.memory.append({**state, **plan})
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]


# ─────────────────────────────────────────────
# FULL CognitiveMorph System
# ─────────────────────────────────────────────

class CognitiveMorphModel:
    def __init__(self):
        self.encoder  = NeuromorphicEncoder(n_neurons=64)
        self.reasoner = SymbolicReasoner()
        self.planner  = AgenticPlanner(horizon=5)
        self.results  = []

    def predict(self, row: Dict) -> Dict:
        sensor_vec = np.array([
            row["sensor_temperature"],
            row["pressure"],
            row["vibration"],
            row["energy_consumption"],
            row["task_complexity"],
            row["relational_dependency_score"],
            row["human_interaction_level"],
            row["time_step_sequence"],
        ])

        spikes = self.encoder.encode(sensor_vec)
        actions, conf = self.reasoner.reason({
            "temperature": row["sensor_temperature"],
            "vibration":   row["vibration"],
            "complexity":  row["task_complexity"],
            "relational":  row["relational_dependency_score"],
            "human_level": row["human_interaction_level"],
            "energy":      row["energy_consumption"],
        })
        plan = self.planner.plan(spikes, actions, conf, {
            "complexity": row["task_complexity"],
            "relational": row["relational_dependency_score"],
        })

        # Simulate predictions with architecture-informed bias
        predicted = {
            "accuracy_score":           min(0.99, plan["confidence"] * 1.05),
            "adaptation_score":         min(0.99, plan["confidence"] * 1.08),
            "task_success_rate":        min(0.99, plan["confidence"] * 1.02),
            "relational_learning_score":min(0.99, plan["confidence"] * 1.10),
            "collaboration_efficiency": min(0.99, plan["confidence"] * (0.9 + row["human_interaction_level"] * 0.15)),
        }

        self.encoder.update_weights(spikes, plan["confidence"])
        return predicted

    def evaluate(self, df) -> Dict:
        model_df = df[df["model_type"] == "CognitiveMorph"].copy()
        preds = [self.predict(row) for _, row in model_df.iterrows()]

        metrics = {}
        for key in ["accuracy_score", "adaptation_score", "task_success_rate",
                    "relational_learning_score", "collaboration_efficiency"]:
            pred_vals = np.array([p[key] for p in preds])
            true_vals = model_df[key].values
            mae = np.mean(np.abs(pred_vals - true_vals))
            metrics[key] = {
                "mean_pred": round(float(np.mean(pred_vals)), 4),
                "mean_true": round(float(np.mean(true_vals)), 4),
                "mae":       round(float(mae), 4),
            }
        return metrics


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import pandas as pd
    df = pd.read_csv("data/industrial_ai_benchmark.csv")
    model = CognitiveMorphModel()
    results = model.evaluate(df)
    print("CognitiveMorph Evaluation:")
    for k, v in results.items():
        print(f"  {k}: mean_pred={v['mean_pred']}, mean_true={v['mean_true']}, MAE={v['mae']}")