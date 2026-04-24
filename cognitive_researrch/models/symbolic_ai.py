"""
Symbolic AI System
Rule-based + knowledge graph reasoning for industrial AI
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Fact:
    name: str
    value: float
    confidence: float = 1.0


@dataclass
class InferenceRule:
    name: str
    antecedents: List[Tuple[str, str, float]]  # (feature, operator, threshold)
    consequent: Dict[str, float]
    weight: float = 1.0


class KnowledgeBase:
    """Industrial domain knowledge encoded as rules."""

    def __init__(self):
        self.rules: List[InferenceRule] = self._build_rules()
        self.facts: Dict[str, Fact] = {}

    def _build_rules(self) -> List[InferenceRule]:
        return [
            InferenceRule(
                name="normal_operation",
                antecedents=[("task_complexity", "<", 0.4),
                              ("sensor_temperature", "<", 100)],
                consequent={"accuracy": 0.78, "task_success": 0.80, "adaptation": 0.55},
                weight=0.9
            ),
            InferenceRule(
                name="high_complexity_degradation",
                antecedents=[("task_complexity", ">", 0.7)],
                consequent={"accuracy": 0.55, "task_success": 0.58, "adaptation": 0.48},
                weight=1.0
            ),
            InferenceRule(
                name="thermal_stress",
                antecedents=[("sensor_temperature", ">", 130),
                              ("vibration", ">", 0.6)],
                consequent={"accuracy": 0.62, "task_success": 0.60, "adaptation": 0.50},
                weight=0.95
            ),
            InferenceRule(
                name="relational_benefit",
                antecedents=[("relational_dependency_score", ">", 0.6)],
                consequent={"accuracy": 0.72, "task_success": 0.70, "relational": 0.75},
                weight=0.8
            ),
            InferenceRule(
                name="human_collaboration",
                antecedents=[("human_interaction_level", ">", 0.6)],
                consequent={"collaboration": 0.65, "accuracy": 0.68},
                weight=0.75
            ),
            InferenceRule(
                name="energy_efficient",
                antecedents=[("energy_consumption", "<", 200)],
                consequent={"task_success": 0.72, "accuracy": 0.71},
                weight=0.85
            ),
            InferenceRule(
                name="simple_structured_task",
                antecedents=[("task_complexity", "<", 0.25),
                              ("relational_dependency_score", "<", 0.4)],
                consequent={"accuracy": 0.82, "task_success": 0.84, "adaptation": 0.52},
                weight=0.92
            ),
        ]

    def assert_facts(self, row: Dict):
        self.facts = {k: Fact(name=k, value=v) for k, v in row.items()
                      if isinstance(v, (int, float))}

    def evaluate_condition(self, feature: str, operator: str, threshold: float) -> bool:
        fact = self.facts.get(feature)
        if fact is None:
            return False
        val = fact.value
        return {
            ">":  val > threshold,
            "<":  val < threshold,
            ">=": val >= threshold,
            "<=": val <= threshold,
            "==": abs(val - threshold) < 0.01,
        }.get(operator, False)

    def fire_rules(self) -> Dict[str, List[float]]:
        """Return weighted outputs from all matching rules."""
        fired_outputs: Dict[str, List[float]] = {
            "accuracy": [], "task_success": [], "adaptation": [],
            "relational": [], "collaboration": []
        }

        for rule in self.rules:
            all_match = all(
                self.evaluate_condition(feat, op, thresh)
                for feat, op, thresh in rule.antecedents
            )
            if all_match:
                for metric, val in rule.consequent.items():
                    if metric in fired_outputs:
                        fired_outputs[metric].append(val * rule.weight)

        return fired_outputs


class FuzzyInference:
    """
    Simple fuzzy logic layer for uncertain conditions.
    Handles partial truth values instead of hard thresholds.
    """
    @staticmethod
    def membership_low(x: float, center: float = 0.3, width: float = 0.2) -> float:
        return max(0, 1 - abs(x - center) / width)

    @staticmethod
    def membership_high(x: float, center: float = 0.7, width: float = 0.2) -> float:
        return max(0, 1 - abs(x - center) / width)

    def fuzzy_accuracy(self, row: Dict) -> float:
        c_low  = self.membership_low(row["task_complexity"])
        c_high = self.membership_high(row["task_complexity"])
        t_safe = self.membership_low(row["sensor_temperature"] / 180.0, 0.4, 0.3)
        return 0.65 + c_low * 0.15 - c_high * 0.12 + t_safe * 0.08


class SymbolicAIModel:
    """
    Full symbolic AI pipeline: KB + forward chaining + fuzzy inference.
    Strengths: interpretable, structured tasks.
    Weaknesses: brittle on complexity, no continual learning.
    """

    def __init__(self):
        self.kb = KnowledgeBase()
        self.fuzzy = FuzzyInference()

    def predict(self, row: Dict) -> Dict:
        self.kb.assert_facts(row)
        outputs = self.kb.fire_rules()

        def aggregate(values: List[float], default: float) -> float:
            if not values:
                return default
            return np.clip(np.mean(values), 0.35, 0.92)

        fuzzy_acc = self.fuzzy.fuzzy_accuracy(row)

        accuracy     = (aggregate(outputs["accuracy"], 0.65) + fuzzy_acc) / 2.0
        task_success = aggregate(outputs["task_success"], 0.63)
        adaptation   = aggregate(outputs["adaptation"], 0.50)
        relational   = aggregate(outputs["relational"], 0.60)
        collab       = aggregate(outputs["collaboration"], 0.55)

        # Symbolic systems degrade on high complexity
        penalty = max(0, row["task_complexity"] - 0.6) * 0.18

        return {
            "accuracy_score":            round(float(np.clip(accuracy - penalty, 0.35, 0.90)), 4),
            "adaptation_score":          round(float(np.clip(adaptation, 0.35, 0.75)), 4),
            "task_success_rate":         round(float(np.clip(task_success - penalty * 0.8, 0.38, 0.88)), 4),
            "relational_learning_score": round(float(np.clip(relational, 0.40, 0.82)), 4),
            "collaboration_efficiency":  round(float(np.clip(collab, 0.38, 0.76)), 4),
        }

    def evaluate(self, df) -> Dict:
        model_df = df[df["model_type"] == "Symbolic"].copy()
        preds = [self.predict(row) for _, row in model_df.iterrows()]
        target_cols = ["accuracy_score", "adaptation_score", "task_success_rate",
                       "relational_learning_score", "collaboration_efficiency"]
        metrics = {}
        for key in target_cols:
            pred_vals = np.array([p[key] for p in preds])
            true_vals = model_df[key].values
            metrics[key] = {
                "mean_pred": round(float(np.mean(pred_vals)), 4),
                "mean_true": round(float(np.mean(true_vals)), 4),
                "mae":       round(float(np.mean(np.abs(pred_vals - true_vals))), 4),
            }
        return metrics


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/industrial_ai_benchmark.csv")
    model = SymbolicAIModel()
    results = model.evaluate(df)
    print("Symbolic AI Evaluation:")
    for k, v in results.items():
        print(f"  {k}: {v}")