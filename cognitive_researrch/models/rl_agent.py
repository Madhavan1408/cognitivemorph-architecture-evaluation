"""
Reinforcement Learning Agent
Q-Learning with epsilon-greedy exploration for industrial task optimization
"""

import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class IndustrialEnvironment:
    """
    Simulates an industrial control environment.
    State = discretized sensor bins. Actions = control decisions.
    """
    ACTIONS = ["increase_load", "decrease_load", "maintain", "alert", "optimize"]

    def __init__(self, df_rows: List[Dict]):
        self.data = df_rows
        self.index = 0
        self.n_actions = len(self.ACTIONS)

    def reset(self):
        self.index = 0
        return self._get_state()

    def step(self, action_idx: int) -> Tuple:
        row = self.data[self.index % len(self.data)]
        reward = self._compute_reward(action_idx, row)
        self.index += 1
        done = self.index >= len(self.data)
        next_state = self._get_state() if not done else None
        return next_state, reward, done, row

    def _get_state(self) -> Tuple:
        row = self.data[self.index % len(self.data)]
        temp_bin   = int(row["sensor_temperature"] / 45)     # 0-3
        comp_bin   = int(row["task_complexity"] * 3)          # 0-2
        rel_bin    = int(row["relational_dependency_score"] * 2)  # 0-1
        return (min(temp_bin, 3), min(comp_bin, 2), min(rel_bin, 1))

    def _compute_reward(self, action_idx: int, row: Dict) -> float:
        action = self.ACTIONS[action_idx]
        complexity = row["task_complexity"]
        relational = row["relational_dependency_score"]
        temp = row["sensor_temperature"]

        # Reward shaping based on action appropriateness
        reward = 0.0
        if action == "maintain" and complexity < 0.5:
            reward += 0.8
        if action == "optimize" and relational > 0.6:
            reward += 0.85
        if action == "alert" and temp > 130:
            reward += 0.9
        if action == "decrease_load" and complexity > 0.7:
            reward += 0.75
        if action == "increase_load" and complexity < 0.3:
            reward += 0.7

        # Add base success signal
        reward += row["task_success_rate"] * 0.3
        return min(reward, 1.0)


class QLearningAgent:
    """
    Tabular Q-Learning with epsilon-greedy exploration.
    Good at adaptation via policy, weak at relational reasoning.
    """
    def __init__(self, n_actions: int = 5, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.3):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: Dict = defaultdict(lambda: np.zeros(n_actions))
        self.episode_rewards: List[float] = []

    def act(self, state: Tuple) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: Tuple, action: int, reward: float,
               next_state: Tuple, done: bool):
        current_q = self.Q[state][action]
        if done or next_state is None:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self, min_eps: float = 0.05, decay: float = 0.98):
        self.epsilon = max(min_eps, self.epsilon * decay)


class RLModel:
    """Full RL training + evaluation pipeline."""

    def __init__(self):
        self.agent = QLearningAgent()
        self.trained = False
        self.performance_log: List[float] = []

    def train(self, df, n_episodes: int = 40):
        model_df = df[df["model_type"] == "RL"]
        rows = model_df.to_dict("records")
        env = IndustrialEnvironment(rows)

        for ep in range(n_episodes):
            state = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, _ = env.step(action)
                self.agent.update(state, action, reward, next_state, done)
                if next_state:
                    state = next_state
                ep_reward += reward
            self.agent.decay_epsilon()
            self.performance_log.append(ep_reward / len(rows))

        self.trained = True

    def _state_to_predicted_perf(self, row: Dict) -> Dict:
        """Map Q-values for a row's state to performance estimates."""
        temp_bin = int(row["sensor_temperature"] / 45)
        comp_bin = int(row["task_complexity"] * 3)
        rel_bin  = int(row["relational_dependency_score"] * 2)
        state    = (min(temp_bin, 3), min(comp_bin, 2), min(rel_bin, 1))

        q_vals = self.agent.Q[state]
        q_norm = (q_vals - q_vals.min()) / (np.ptp(q_vals) + 1e-9)

        # RL excels at adaptation, weaker at relational + collaboration
        complexity  = row["task_complexity"]
        relational  = row["relational_dependency_score"]
        human_level = row["human_interaction_level"]

        base = 0.60 + q_norm.max() * 0.25
        return {
            "accuracy_score":            round(min(0.95, base - complexity * 0.05), 4),
            "adaptation_score":          round(min(0.97, base + 0.08), 4),
            "task_success_rate":         round(min(0.95, base + 0.02), 4),
            "relational_learning_score": round(min(0.90, base - 0.10 + relational * 0.05), 4),
            "collaboration_efficiency":  round(min(0.88, base - 0.08 + human_level * 0.06), 4),
        }

    def evaluate(self, df) -> Dict:
        self.train(df)
        model_df = df[df["model_type"] == "RL"].copy()
        preds = [self._state_to_predicted_perf(row) for _, row in model_df.iterrows()]
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
    model = RLModel()
    results = model.evaluate(df)
    print("RL Agent Evaluation:")
    for k, v in results.items():
        print(f"  {k}: {v}")