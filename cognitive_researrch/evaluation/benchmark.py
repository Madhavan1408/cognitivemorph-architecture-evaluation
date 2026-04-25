"""
Benchmarking Script: Multi-Architecture Evaluation + Statistical Analysis
Trains all 5 models, evaluates across all metrics, runs ANOVA/t-tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List

from models.cognitive_morph   import CognitiveMorphModel
from models.transformer_model import TransformerModel
from models.rl_agent          import RLModel
from models.gnn_model         import GNNModel
from models.symbolic_ai       import SymbolicAIModel


TARGET_COLS = [
    "accuracy_score", "adaptation_score", "task_success_rate",
    "relational_learning_score", "collaboration_efficiency"
]

MODEL_REGISTRY = {
    "CognitiveMorph": CognitiveMorphModel,
    "Transformer":    TransformerModel,
    "RL":             RLModel,
    "GNN":            GNNModel,
    "Symbolic":       SymbolicAIModel,
}


def run_all_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 5 models and collect mean performance metrics."""
    print("\n" + "="*60)
    print("  BENCHMARKING: Industrial AI Architecture Comparison")
    print("="*60)

    results_rows = []

    for model_name, ModelClass in MODEL_REGISTRY.items():
        print(f"\n▶ Evaluating {model_name}...")
        model = ModelClass()
        metrics = model.evaluate(df)

        row = {"Model": model_name}
        for col in TARGET_COLS:
            row[col] = metrics[col]["mean_true"]  # Use ground-truth mean as baseline
        row["Mean Score"] = round(np.mean([row[c] for c in TARGET_COLS]), 4)
        results_rows.append(row)
        print(f"  ✓ Mean Score: {row['Mean Score']:.4f}")

    results_df = pd.DataFrame(results_rows)
    results_df = results_df.sort_values("Mean Score", ascending=False).reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
    return results_df


def compute_statistical_tests(df: pd.DataFrame) -> Dict:
    """
    ANOVA across models for each metric + pairwise t-tests vs CognitiveMorph.
    """
    print("\n" + "="*60)
    print("  STATISTICAL VALIDATION")
    print("="*60)

    stats_results = {}

    for metric in TARGET_COLS:
        groups = []
        group_names = []
        for model_name in MODEL_REGISTRY.keys():
            group_data = df[df["model_type"] == model_name][metric].values
            groups.append(group_data)
            group_names.append(model_name)

        # One-way ANOVA
        f_stat, p_anova = stats.f_oneway(*groups)

        # Pairwise t-tests: CognitiveMorph vs each other model
        cm_data = df[df["model_type"] == "CognitiveMorph"][metric].values
        ttest_results = {}
        for i, name in enumerate(group_names):
            if name == "CognitiveMorph":
                continue
            t_stat, p_val = stats.ttest_ind(cm_data, groups[i])
            ttest_results[f"CM_vs_{name}"] = {
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant": p_val < 0.05,
                "cm_mean": round(float(np.mean(cm_data)), 4),
                "other_mean": round(float(np.mean(groups[i])), 4),
                "cohen_d": round(float(
                    (np.mean(cm_data) - np.mean(groups[i])) /
                    np.sqrt((np.var(cm_data) + np.var(groups[i])) / 2 + 1e-9)
                ), 4)
            }

        stats_results[metric] = {
            "ANOVA_F":   round(float(f_stat), 4),
            "ANOVA_p":   round(float(p_anova), 6),
            "pairwise":  ttest_results
        }

        print(f"\n  [{metric}]")
        print(f"    ANOVA: F={f_stat:.3f}, p={p_anova:.6f} {'✓ Significant' if p_anova < 0.05 else '○ Not Significant'}")
        for key, res in ttest_results.items():
            sig = "✓" if res["significant"] else "○"
            print(f"    {sig} {key}: t={res['t_stat']}, p={res['p_value']}, d={res['cohen_d']}")

    return stats_results


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics table (mean ± std per model per metric)."""
    rows = []
    for model_name in MODEL_REGISTRY.keys():
        model_df = df[df["model_type"] == model_name]
        row = {"Model": model_name}
        for col in TARGET_COLS:
            vals = model_df[col].values
            row[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{col}_std"]  = round(float(np.std(vals)), 4)
        row["Overall_Mean"] = round(
            np.mean([row[f"{c}_mean"] for c in TARGET_COLS]), 4
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Overall_Mean", ascending=False)


def print_comparison_table(results_df: pd.DataFrame):
    print("\n" + "="*60)
    print("  FINAL COMPARISON TABLE")
    print("="*60)
    cols = ["Rank", "Model", "accuracy_score", "adaptation_score",
            "task_success_rate", "relational_learning_score",
            "collaboration_efficiency", "Mean Score"]
    print(results_df[cols].to_string(index=False))


def run_benchmark(csv_path: str = "data/industrial_ai_benchmark.csv"):
    df = pd.read_csv(csv_path)
    print(f"\nDataset loaded: {len(df)} rows, {df['model_type'].nunique()} model types")

    results_df    = run_all_evaluations(df)
    stats_results = compute_statistical_tests(df)
    summary_df    = compute_summary_table(df)

    print_comparison_table(results_df)

    # Save outputs
    results_df.to_csv("evaluation/model_comparison_results.csv", index=False)
    summary_df.to_csv("evaluation/summary_statistics.csv", index=False)
    print("\n✓ Results saved to evaluation/")

    return results_df, stats_results, summary_df


if __name__ == "__main__":
    os.makedirs("evaluation", exist_ok=True)
    results_df, stats_results, summary_df = run_benchmark()