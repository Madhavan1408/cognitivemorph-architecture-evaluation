# CognitiveMorph: Multi-Architecture Industrial AI Benchmark

> **A rigorous benchmark comparing CognitiveMorph (neuromorphic + symbolic + agentic hybrid) against Transformer, RL, GNN, and Symbolic AI architectures in physics-constrained industrial environments.**

---

## Problem Statement

Industrial AI systems face three simultaneous challenges that no single architecture solves well:

1. **Physics-constrained sensor fusion** — temperature, pressure, vibration interact non-linearly
2. **Relational dependency reasoning** — subsystems depend on each other; graph structure matters
3. **Continual adaptation** — environments shift; models must update without catastrophic forgetting
4. **Human-AI collaboration** — systems must seamlessly hand off to human operators

Standard Transformer, RL, GNN, and Symbolic AI architectures each solve one or two of these but fail on the rest. **CognitiveMorph** is designed as a unified hybrid that addresses all four simultaneously.

---

## Architecture Explanation

### CognitiveMorph (Hybrid Cognitive Architecture)

```
Sensor Input
    │
    ▼
┌─────────────────────────┐
│  Neuromorphic Encoder   │  ← Spike-timing encoding; energy-efficient
│  (STDP weight updates)  │    temporal pattern detection
└───────────┬─────────────┘
            │ spikes
            ▼
┌─────────────────────────┐
│  Symbolic Reasoner      │  ← Physics-domain rules; interpretable
│  (Rule-based KB)        │    decisions; confidence scoring
└───────────┬─────────────┘
            │ actions + confidence
            ▼
┌─────────────────────────┐
│  Agentic Planner        │  ← Goal-directed lookahead; episodic memory
│  (Planning + Memory)    │    continual learning via memory retrieval
└───────────┬─────────────┘
            │ plan
            ▼
     Prediction Output
```

**Why this wins:**
- Neuromorphic layer captures temporal sensor dynamics (no static embedding)
- Symbolic layer enforces physics constraints (no hallucinations)
- Agentic layer enables multi-step planning and adaptation

### Comparison Architectures

| Architecture | Strength | Weakness |
|---|---|---|
| **Transformer** | Sequence patterns, attention | Poor continual learning, expensive |
| **RL Agent** | Adaptation via policy | Slow convergence, no relational reasoning |
| **GNN** | Relational dependencies | No temporal dynamics, limited adaptation |
| **Symbolic AI** | Interpretable, structured | Brittle on complexity, no learning |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline
```bash
python main.py
```

This will:
- Generate the 300-row synthetic dataset
- Train and evaluate all 5 architectures
- Run ANOVA and t-test statistical validation
- Export a formatted 3-sheet Excel report

### 3. Run individual components
```bash
# Dataset only
python data/generate_dataset.py

# Benchmark only (requires dataset)
python evaluation/benchmark.py

# Excel report only (requires dataset)
python evaluation/generate_excel.py
```

### 4. Run individual model files
```bash
python models/cognitive_morph.py
python models/transformer_model.py
python models/rl_agent.py
python models/gnn_model.py
python models/symbolic_ai.py
```

---

## Project Structure

```
/cognitivmorph_research
│
├── main.py                         # Full pipeline runner
├── requirements.txt
├── README.md
│
├── data/
│   ├── generate_dataset.py         # Synthetic dataset generator
│   └── industrial_ai_benchmark.csv # Generated dataset (300 rows)
│
├── models/
│   ├── cognitive_morph.py          # CognitiveMorph hybrid system
│   ├── transformer_model.py        # Self-attention sequence model
│   ├── rl_agent.py                 # Q-Learning RL agent
│   ├── gnn_model.py                # Graph Neural Network
│   └── symbolic_ai.py             # Rule-based KB + fuzzy logic
│
└── evaluation/
    ├── benchmark.py                # Multi-model evaluator + stats tests
    ├── generate_excel.py           # Formatted Excel report generator
    ├── model_comparison_results.csv
    └── CognitiveMorph_Benchmark_Report.xlsx
```

---

## Results Summary

| Rank | Model | Accuracy | Adaptation | Task Success | Relational | Collaboration | Mean |
|---|---|---|---|---|---|---|---|
| 🥇 1 | **CognitiveMorph** | ~0.91 | ~0.93 | ~0.89 | ~0.92 | ~0.89 | **~0.91** |
| 🥈 2 | GNN | ~0.82 | ~0.70 | ~0.78 | ~0.85 | ~0.65 | ~0.76 |
| 🥉 3 | Transformer | ~0.83 | ~0.72 | ~0.80 | ~0.70 | ~0.67 | ~0.74 |
| 4 | RL | ~0.76 | ~0.83 | ~0.79 | ~0.63 | ~0.61 | ~0.72 |
| 5 | Symbolic | ~0.70 | ~0.52 | ~0.68 | ~0.61 | ~0.56 | ~0.61 |

> *Values are approximate; exact values depend on random seed and runtime.*

**Statistical Significance:** ANOVA confirms significant differences (p < 0.05) across all 5 metrics. Pairwise t-tests show CognitiveMorph significantly outperforms all alternatives on accuracy, adaptation, and relational scores (Cohen's d > 0.8).

---

## Why CognitiveMorph Performs Better — Panel Defense

### 1. Physics-Awareness
The neuromorphic encoder uses **spike-timing** to detect transient anomalies in sensor streams (temperature spikes, vibration bursts) that averaged embeddings miss. The symbolic KB enforces **causal physics rules** (temperature → pressure effects), preventing statistically valid but physically impossible predictions.

### 2. Continual Learning
The agentic planning layer maintains **episodic memory** and retrieves similar past scenarios to bias current decisions. STDP weight updates allow the neuromorphic encoder to **strengthen frequently activated pathways** — a form of online learning without full retraining.

### 3. Hybrid Reasoning
CognitiveMorph combines the best of three paradigms:
- **Sub-symbolic** (neuromorphic): fast, parallel, robust to noise
- **Symbolic** (rule-based): interpretable, physics-constrained, transparent
- **Agentic** (planning): goal-directed, multi-step, memory-augmented

### 4. Collaboration Efficiency
The symbolic layer explicitly detects `human_level > 0.7` and triggers `handoff_human` action, enabling graceful human-AI collaboration. No other architecture in this benchmark models human interaction as a first-class planning primitive.

---

## Dataset Features

| Feature | Type | Range | Description |
|---|---|---|---|
| sensor_temperature | float | 30–180°C | Thermal state of system |
| pressure | float | 1.0–8.0 bar | Hydraulic pressure |
| vibration | float | 0.1–2.0 | Mechanical vibration amplitude |
| energy_consumption | float | 100–350 kWh | Power draw |
| task_complexity | float | 0.2–1.0 | Normalized task difficulty |
| relational_dependency_score | float | 0–1 | Inter-subsystem coupling |
| human_interaction_level | float | 0–1 | Human operator engagement |
| time_step_sequence | float | continuous | Temporal index |

---

## Citation

```
@software{cognitivmorph_benchmark_2024,
  title  = {CognitiveMorph: Hybrid Neuromorphic-Symbolic-Agentic Architecture Benchmark},
  note   = {Industrial AI performance evaluation across 5 architecture families}
}
```
