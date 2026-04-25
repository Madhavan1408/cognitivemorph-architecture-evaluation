"""
CognitiveMorph Research — Main Entry Point
Run full pipeline: generate data → train models → benchmark → export Excel
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset import generate_dataset
from evaluation.benchmark  import run_benchmark
from evaluation.generate_excel import generate_excel_report


def main():
    print("\n" + "█"*60)
    print("  CognitiveMorph vs Multi-Architecture Benchmark")
    print("  Industrial AI Performance Evaluation Framework")
    print("█"*60)

    os.makedirs("data", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    # Step 1: Generate dataset
    print("\n[STEP 1] Generating physics-constrained industrial dataset...")
    df = generate_dataset()

    # Step 2: Run benchmarks
    print("\n[STEP 2] Running architecture evaluations...")
    results_df, stats_results, summary_df = run_benchmark("data/industrial_ai_benchmark.csv")

    # Step 3: Export Excel
    print("\n[STEP 3] Generating Excel report...")
    excel_path = generate_excel_report(df)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  ✓ Dataset:    data/industrial_ai_benchmark.csv")
    print(f"  ✓ CSV Results: evaluation/model_comparison_results.csv")
    print(f"  ✓ Excel Report: {excel_path}")
    print("\n  TOP RESULT:")
    top = results_df.iloc[0]
    print(f"  🏆 {top['Model']} — Mean Score: {top['Mean Score']:.4f}")


if __name__ == "__main__":
    main()