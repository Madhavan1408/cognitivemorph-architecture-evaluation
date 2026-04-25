"""
Graph Neural Network Implementation
Message-passing over sensor dependency graph for industrial AI
"""

import numpy as np
from typing import Dict, List, Tuple


class GraphLayer:
    """
    Single GNN message-passing layer.
    H_new = ReLU(A_hat @ H @ W) where A_hat = normalized adjacency.
    """
    def __init__(self, in_dim: int, out_dim: int):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.bias = np.zeros(out_dim)

    def forward(self, H: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
        """H: (n_nodes, in_dim), A_hat: (n_nodes, n_nodes)"""
        agg = A_hat @ H          # Aggregate neighbor features
        out = agg @ self.W + self.bias
        return np.maximum(0, out)  # ReLU


class GNNModel:
    """
    Graph Neural Network for relational sensor dependency modeling.
    Builds a fully connected graph over 8 sensor features as nodes.
    Strengths: relational reasoning. Limitations: adaptation, time-series.
    """
    def __init__(self, n_nodes: int = 8, hidden_dim: int = 16, out_dim: int = 8):
        self.n_nodes = n_nodes
        self.layer1 = GraphLayer(1, hidden_dim)
        self.layer2 = GraphLayer(hidden_dim, out_dim)
        self.output_w = np.random.randn(n_nodes * out_dim, 5) * 0.1
        self.output_b = np.zeros(5)
        self.fitted = False

    def _build_adjacency(self, row: Dict) -> np.ndarray:
        """
        Construct adjacency matrix from sensor correlations.
        Higher relational_dependency_score → denser graph.
        """
        n = self.n_nodes
        relational = row.get("relational_dependency_score", 0.5)
        temp = row.get("sensor_temperature", 75)
        pressure = row.get("pressure", 3.2)
        vib = row.get("vibration", 0.4)

        A = np.zeros((n, n))
        # Physics-informed edges
        A[0, 2] = A[2, 0] = temp / 180.0        # temp-vibration
        A[0, 1] = A[1, 0] = temp * 0.005        # temp-pressure
        A[1, 3] = A[3, 1] = pressure * 0.1      # pressure-energy
        A[2, 3] = A[3, 2] = vib * 0.3           # vibration-energy
        A[4, 5] = A[5, 4] = relational           # complexity-relational
        A[5, 6] = A[6, 5] = relational * 0.8    # relational-human
        A[6, 7] = A[7, 6] = 0.5                 # human-time

        # Add self-loops
        np.fill_diagonal(A, 1.0)

        # Normalize: D^{-1/2} A D^{-1/2}
        D = np.diag(A.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-9))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def _extract_features(self, row: Dict) -> np.ndarray:
        """Node features: each sensor as a 1D node embedding."""
        cols = ["sensor_temperature", "pressure", "vibration", "energy_consumption",
                "task_complexity", "relational_dependency_score",
                "human_interaction_level", "time_step_sequence"]
        vals = np.array([row[c] for c in cols])
        # Normalize
        vals = (vals - vals.mean()) / (vals.std() + 1e-9)
        return vals.reshape(-1, 1)  # (n_nodes, 1)

    def _forward(self, row: Dict) -> np.ndarray:
        H = self._extract_features(row)    # (8, 1)
        A = self._build_adjacency(row)     # (8, 8)
        H1 = self.layer1.forward(H, A)    # (8, 16)
        H2 = self.layer2.forward(H1, A)   # (8, 8)
        flat = H2.flatten()               # (64,)
        out = flat[:self.n_nodes * 8] @ self.output_w + self.output_b  # (5,)
        return 1 / (1 + np.exp(-np.clip(out, -10, 10)))  # sigmoid

    def fit(self, df, epochs: int = 25):
        model_df = df[df["model_type"] == "GNN"]
        target_cols = ["accuracy_score", "adaptation_score", "task_success_rate",
                       "relational_learning_score", "collaboration_efficiency"]

        best_loss = float("inf")
        best_w = self.output_w.copy()

        for _ in range(epochs):
            total_loss = 0.0
            for _, row in model_df.iterrows():
                pred = self._forward(row)
                true = np.array([row[c] for c in target_cols])
                total_loss += np.mean((pred - true) ** 2)
            avg_loss = total_loss / len(model_df)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_w = self.output_w.copy()

            self.output_w = best_w + np.random.randn(*best_w.shape) * 0.008

        self.output_w = best_w
        self.fitted = True

    def predict(self, row: Dict) -> Dict:
        raw = self._forward(row)
        keys = ["accuracy_score", "adaptation_score", "task_success_rate",
                "relational_learning_score", "collaboration_efficiency"]
        return dict(zip(keys, np.clip(raw, 0.3, 0.99).tolist()))

    def evaluate(self, df) -> Dict:
        self.fit(df)
        model_df = df[df["model_type"] == "GNN"].copy()
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
    model = GNNModel()
    results = model.evaluate(df)
    print("GNN Evaluation:")
    for k, v in results.items():
        print(f"  {k}: {v}")