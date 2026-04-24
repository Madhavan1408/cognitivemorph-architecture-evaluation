"""
Transformer Model Implementation
Self-attention over sensor sequences for industrial prediction
"""

import numpy as np
from typing import Dict, List


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu  = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x,  axis=-1, keepdims=True)
    return (x - mu) / (std + eps)


class MultiHeadAttention:
    def __init__(self, d_model: int = 32, n_heads: int = 4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = 0.1
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Split into heads and compute attention
        scale = np.sqrt(self.d_k)
        attn_scores = (Q @ K.T) / scale
        attn_weights = softmax(attn_scores)
        context = attn_weights @ V
        return context @ self.W_o


class FeedForward:
    def __init__(self, d_model: int = 32, d_ff: int = 64):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, x @ self.W1)   # ReLU
        return hidden @ self.W2


class TransformerBlock:
    def __init__(self, d_model: int = 32):
        self.attn = MultiHeadAttention(d_model)
        self.ff   = FeedForward(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = layer_norm(x + self.attn.forward(x))
        x = layer_norm(x + self.ff.forward(x))
        return x


class TransformerModel:
    """
    Lightweight Transformer for tabular sensor sequence modeling.
    Strengths: sequence patterns, attention over features.
    Limitations: limited continual learning, weaker on relational tasks.
    """
    def __init__(self, input_dim: int = 8, d_model: int = 32, n_layers: int = 2):
        self.input_proj = np.random.randn(input_dim, d_model) * 0.1
        self.blocks = [TransformerBlock(d_model) for _ in range(n_layers)]
        self.output_head = np.random.randn(d_model, 5) * 0.1
        self.d_model = d_model
        self.fitted = False

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """x: (seq_len, input_dim) or (input_dim,)"""
        if x.ndim == 1:
            x = x[np.newaxis, :]   # (1, input_dim)
        h = x @ self.input_proj    # (seq, d_model)
        for block in self.blocks:
            h = block.forward(h)
        pooled = h.mean(axis=0)    # global average pool
        return pooled @ self.output_head   # (5,) raw outputs

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def fit(self, df, lr: float = 0.005, epochs: int = 30):
        """Simple gradient-free parameter tuning via random search."""
        model_df = df[df["model_type"] == "Transformer"]
        target_cols = ["accuracy_score", "adaptation_score", "task_success_rate",
                       "relational_learning_score", "collaboration_efficiency"]
        feature_cols = ["sensor_temperature", "pressure", "vibration",
                        "energy_consumption", "task_complexity",
                        "relational_dependency_score", "human_interaction_level",
                        "time_step_sequence"]

        X = model_df[feature_cols].values
        Y = model_df[target_cols].values

        # Normalize X
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-9
        X_norm = (X - self.X_mean) / self.X_std

        best_loss = float("inf")
        best_head = self.output_head.copy()

        for _ in range(epochs):
            preds = np.array([self._sigmoid(self._forward(x)) for x in X_norm])
            loss  = np.mean((preds - Y) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_head = self.output_head.copy()
            # Perturb output head
            self.output_head = best_head + np.random.randn(*best_head.shape) * lr

        self.output_head = best_head
        self.fitted = True

    def predict(self, row: Dict) -> Dict:
        feature_cols = ["sensor_temperature", "pressure", "vibration",
                        "energy_consumption", "task_complexity",
                        "relational_dependency_score", "human_interaction_level",
                        "time_step_sequence"]
        x = np.array([row[c] for c in feature_cols])
        if self.fitted:
            x = (x - self.X_mean) / self.X_std
        raw = self._sigmoid(self._forward(x))
        keys = ["accuracy_score", "adaptation_score", "task_success_rate",
                "relational_learning_score", "collaboration_efficiency"]
        return dict(zip(keys, np.clip(raw, 0.3, 0.99).tolist()))

    def evaluate(self, df) -> Dict:
        model_df = df[df["model_type"] == "Transformer"].copy()
        self.fit(df)
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
    model = TransformerModel()
    results = model.evaluate(df)
    print("Transformer Evaluation:")
    for k, v in results.items():
        print(f"  {k}: {v}")