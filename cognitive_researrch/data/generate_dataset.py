"""
Dataset Generator: Physics-Constrained Industrial AI Benchmark
Generates synthetic but realistic multi-model performance dataset
"""

import numpy as np
import pandas as pd

np.random.seed(42)

MODELS = ["CognitiveMorph", "Transformer", "RL", "GNN", "Symbolic"]
N_PER_MODEL = 60  # 60 per model = 300 total rows


def generate_sensor_features(n, model_type):
    """Generate physics-realistic sensor readings per model context."""
    base_temp    = np.random.normal(75, 12, n)
    base_pressure= np.random.normal(3.2, 0.6, n)
    vibration    = np.random.exponential(0.4, n) + 0.1
    energy       = np.random.normal(220, 35, n)
    complexity   = np.random.uniform(0.2, 1.0, n)
    relational   = np.random.beta(3, 2, n)
    human_level  = np.random.beta(2, 3, n)
    time_seq     = np.arange(n) + np.random.normal(0, 2, n)

    # Add physics-constrained noise
    temp = np.clip(base_temp + vibration * 4.5, 30, 180)
    pressure = np.clip(base_pressure + energy * 0.002, 1.0, 8.0)

    return temp, pressure, vibration, energy, complexity, relational, human_level, time_seq


def compute_targets(model_type, complexity, relational, human_level, time_seq, energy, n):
    """
    Compute target metrics based on architecture strengths.
    CognitiveMorph is tuned to outperform on relational + adaptive tasks.
    """
    noise = lambda scale: np.random.normal(0, scale, n)

    if model_type == "CognitiveMorph":
        accuracy         = np.clip(0.88 + relational * 0.08 - complexity * 0.04 + noise(0.03), 0.70, 0.99)
        adaptation       = np.clip(0.91 + human_level * 0.06 - complexity * 0.03 + noise(0.025), 0.75, 0.99)
        task_success     = np.clip(0.87 + (1 - complexity) * 0.10 + relational * 0.04 + noise(0.03), 0.72, 0.98)
        relational_score = np.clip(0.90 + relational * 0.07 + noise(0.025), 0.76, 0.99)
        collab_eff       = np.clip(0.85 + human_level * 0.10 + noise(0.03), 0.70, 0.99)

    elif model_type == "Transformer":
        accuracy         = np.clip(0.82 + (1 - complexity) * 0.07 + noise(0.04), 0.60, 0.97)
        adaptation       = np.clip(0.72 + noise(0.045), 0.55, 0.90)
        task_success     = np.clip(0.79 + (1 - complexity) * 0.06 + noise(0.04), 0.60, 0.94)
        relational_score = np.clip(0.68 + relational * 0.08 + noise(0.04), 0.50, 0.88)
        collab_eff       = np.clip(0.65 + human_level * 0.06 + noise(0.04), 0.48, 0.85)

    elif model_type == "RL":
        accuracy         = np.clip(0.75 + (1 - complexity) * 0.08 + noise(0.05), 0.52, 0.92)
        adaptation       = np.clip(0.83 + noise(0.04), 0.65, 0.97)
        task_success     = np.clip(0.78 + noise(0.05), 0.58, 0.94)
        relational_score = np.clip(0.62 + relational * 0.05 + noise(0.05), 0.44, 0.82)
        collab_eff       = np.clip(0.60 + human_level * 0.07 + noise(0.045), 0.42, 0.83)

    elif model_type == "GNN":
        accuracy         = np.clip(0.80 + relational * 0.09 + noise(0.04), 0.60, 0.96)
        adaptation       = np.clip(0.70 + noise(0.045), 0.52, 0.88)
        task_success     = np.clip(0.76 + relational * 0.06 + noise(0.04), 0.58, 0.93)
        relational_score = np.clip(0.82 + relational * 0.10 + noise(0.035), 0.65, 0.97)
        collab_eff       = np.clip(0.64 + noise(0.045), 0.46, 0.84)

    elif model_type == "Symbolic":
        accuracy         = np.clip(0.70 - complexity * 0.15 + noise(0.05), 0.38, 0.88)
        adaptation       = np.clip(0.52 + noise(0.04), 0.36, 0.72)
        task_success     = np.clip(0.68 - complexity * 0.12 + noise(0.05), 0.40, 0.85)
        relational_score = np.clip(0.60 + noise(0.04), 0.42, 0.80)
        collab_eff       = np.clip(0.55 + noise(0.04), 0.38, 0.74)

    return accuracy, adaptation, task_success, relational_score, collab_eff


def generate_dataset():
    rows = []
    for model in MODELS:
        n = N_PER_MODEL
        temp, pressure, vib, energy, complexity, relational, human_level, time_seq = \
            generate_sensor_features(n, model)

        acc, adp, ts, rs, ce = compute_targets(
            model, complexity, relational, human_level, time_seq, energy, n
        )

        for i in range(n):
            rows.append({
                "model_type":               model,
                "sensor_temperature":       round(temp[i], 3),
                "pressure":                 round(pressure[i], 4),
                "vibration":                round(vib[i], 4),
                "energy_consumption":       round(energy[i], 3),
                "task_complexity":          round(complexity[i], 4),
                "relational_dependency_score": round(relational[i], 4),
                "human_interaction_level":  round(human_level[i], 4),
                "time_step_sequence":       round(time_seq[i], 2),
                "accuracy_score":           round(acc[i], 4),
                "adaptation_score":         round(adp[i], 4),
                "task_success_rate":        round(ts[i], 4),
                "relational_learning_score":round(rs[i], 4),
                "collaboration_efficiency": round(ce[i], 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/industrial_ai_benchmark.csv", index=False)
    print(f"Dataset generated: {len(df)} rows × {len(df.columns)} columns")
    print(df.groupby("model_type")[["accuracy_score","adaptation_score","task_success_rate"]].mean().round(3))
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print("\nPreview (first 5 rows):")
    print(df.head().to_string())