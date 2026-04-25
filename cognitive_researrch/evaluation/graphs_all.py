"""
CognitiveMorph Architecture Evaluation — All Graph Types
Generates 24 chart types from industrial_ai_benchmark.csv results

Run from project root:   python evaluation/graphs_all.py
Run from evaluation/:    python graphs_all.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings("ignore")

# ─── Resolve paths relative to THIS file (works from any working directory) ──

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))          # .../evaluation/
DATA_CSV  = os.path.join(BASE_DIR, "..", "data", "industrial_ai_benchmark.csv")
OUT       = os.path.join(BASE_DIR, "graphs")
os.makedirs(OUT, exist_ok=True)

# ─── Data ────────────────────────────────────────────────────────────────────

if not os.path.exists(DATA_CSV):
    print(f"ERROR: Dataset not found at:\n  {os.path.abspath(DATA_CSV)}")
    print("Make sure industrial_ai_benchmark.csv is inside the data/ folder.")
    sys.exit(1)

df = pd.read_csv(DATA_CSV)
print(f"Loaded {len(df)} rows from {os.path.abspath(DATA_CSV)}")

MODELS   = ["CognitiveMorph", "GNN", "Transformer", "RL", "Symbolic"]
METRICS  = ["accuracy_score", "adaptation_score", "task_success_rate",
            "relational_learning_score", "collaboration_efficiency"]
LABELS   = ["Accuracy", "Adaptation", "Task Success", "Relational", "Collaboration"]

COLORS = {
    "CognitiveMorph": "#185FA5",
    "GNN":            "#1D9E75",
    "Transformer":    "#BA7517",
    "RL":             "#D4537E",
    "Symbolic":       "#888780",
}
PALETTE = [COLORS[m] for m in MODELS]

# Summary stats
summary = df.groupby("model_type")[METRICS].agg(["mean", "std"]).round(4)
means   = df.groupby("model_type")[METRICS].mean().reindex(MODELS)
stds    = df.groupby("model_type")[METRICS].std().reindex(MODELS)
overall = means.mean(axis=1).sort_values(ascending=False)

# ─── Style ───────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         130,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# OUT is set at top of file via BASE_DIR

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")

print("Generating charts…")

# ─────────────────────────────────────────────────────────────────────────────
# 1. RANKED BAR CHART — Overall Mean Score
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(MODELS))
ranked = overall.reset_index()
ranked.columns = ["Model", "Score"]
bars = ax.bar(ranked["Model"], ranked["Score"],
              color=[COLORS[m] for m in ranked["Model"]],
              width=0.55, zorder=3)
for bar, val in zip(bars, ranked["Score"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0.4, 1.02)
ax.set_ylabel("Mean Score (all metrics)")
ax.set_title("Architecture Ranking — Overall Mean Score", fontsize=14, fontweight="bold", pad=12)
ax.axhline(0.75, color="gray", lw=1, ls=":", alpha=0.6, label="0.75 baseline")
ax.legend(fontsize=9)
save(fig, "01_ranked_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GROUPED BAR CHART — All 5 metrics side by side
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
n_metrics = len(METRICS)
n_models  = len(MODELS)
x = np.arange(n_metrics)
width = 0.14
offsets = np.linspace(-(n_models-1)/2*width, (n_models-1)/2*width, n_models)
for i, model in enumerate(MODELS):
    vals = means.loc[model, METRICS].values
    errs = stds.loc[model, METRICS].values
    bars = ax.bar(x + offsets[i], vals, width, label=model,
                  color=COLORS[model], alpha=0.88, zorder=3,
                  yerr=errs, capsize=3, error_kw={"elinewidth":1.2})
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylim(0.3, 1.08)
ax.set_ylabel("Score")
ax.set_title("All Metrics — Grouped Bar Comparison", fontsize=14, fontweight="bold", pad=12)
ax.legend(ncol=5, fontsize=9, loc="upper right")
save(fig, "02_grouped_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HORIZONTAL BAR — per-metric champion view
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
for ax, metric, label in zip(axes, METRICS, LABELS):
    vals = means[metric].reindex(MODELS)
    colors = [COLORS[m] for m in MODELS]
    bars = ax.barh(MODELS, vals, color=colors, height=0.6, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.008, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0.3, 1.1)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.axvline(vals.max(), color="gray", lw=0.8, ls=":", alpha=0.5)
fig.suptitle("Per-Metric Horizontal Bar Charts", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "03_horizontal_bar_per_metric.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. RADAR / SPIDER CHART
# ─────────────────────────────────────────────────────────────────────────────
from matplotlib.patches import Circle
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
angles = np.linspace(0, 2*np.pi, len(METRICS), endpoint=False).tolist()
angles += angles[:1]
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylim(0.3, 1.0)
ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.4","0.6","0.8","1.0"], fontsize=7, color="gray")
for model in MODELS:
    vals = means.loc[model, METRICS].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, "o-", lw=2, label=model, color=COLORS[model], markersize=5)
    ax.fill(angles, vals, alpha=0.07, color=COLORS[model])
ax.set_title("Radar Chart — Metric Profiles", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
save(fig, "04_radar_spider.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HEATMAP — mean scores matrix
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
heat_data = means[METRICS].copy()
heat_data.columns = LABELS
sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Score", "shrink": 0.8},
            ax=ax, vmin=0.45, vmax=1.0)
ax.set_title("Heatmap — Mean Scores per Architecture × Metric", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Architecture")
plt.yticks(rotation=0)
save(fig, "05_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BOX PLOT — distribution of each metric per model
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
for ax, metric, label in zip(axes, METRICS, LABELS):
    data_list = [df[df["model_type"] == m][metric].values for m in MODELS]
    bp = ax.boxplot(data_list, patch_artist=True,
                    medianprops={"color":"black","linewidth":2},
                    whiskerprops={"linewidth":1.2},
                    capprops={"linewidth":1.2})
    for patch, model in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.75)
    ax.set_xticklabels([m[:4] for m in MODELS], fontsize=8)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylim(0.25, 1.05)
fig.suptitle("Box Plots — Score Distributions per Architecture", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "06_boxplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VIOLIN PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
for ax, metric, label in zip(axes, METRICS, LABELS):
    data_list = [df[df["model_type"] == m][metric].values for m in MODELS]
    parts = ax.violinplot(data_list, positions=range(len(MODELS)),
                          showmeans=True, showmedians=False)
    for pc, model in zip(parts["bodies"], MODELS):
        pc.set_facecolor(COLORS[model])
        pc.set_alpha(0.7)
    for part in ["cmeans","cbars","cmins","cmaxes"]:
        if part in parts:
            parts[part].set_color("black")
            parts[part].set_linewidth(1.2)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[:4] for m in MODELS], fontsize=8)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylim(0.25, 1.05)
fig.suptitle("Violin Plots — Score Distributions per Architecture", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "07_violin.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. LINE CHART — mean ± std per model across metrics
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(METRICS))
for model in MODELS:
    y  = means.loc[model, METRICS].values
    sd = stds.loc[model, METRICS].values
    ax.plot(x, y, "o-", lw=2.2, label=model, color=COLORS[model], markersize=7, zorder=4)
    ax.fill_between(x, y - sd, y + sd, alpha=0.12, color=COLORS[model])
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylim(0.3, 1.05)
ax.set_ylabel("Score")
ax.set_title("Line Chart — Metric Profiles with ±1 STD Bands", fontsize=14, fontweight="bold", pad=12)
ax.legend(ncol=5, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.18))
save(fig, "08_line_std_band.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SCATTER — Accuracy vs Adaptation
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for model in MODELS:
    sub = df[df["model_type"] == model]
    ax.scatter(sub["accuracy_score"], sub["adaptation_score"],
               color=COLORS[model], alpha=0.55, s=40, label=model, zorder=3)
    # Centroid marker
    mx, my = sub["accuracy_score"].mean(), sub["adaptation_score"].mean()
    ax.scatter(mx, my, color=COLORS[model], s=200, marker="D",
               edgecolors="white", linewidths=1.5, zorder=5)
    ax.annotate(model, (mx, my), textcoords="offset points",
                xytext=(6, 4), fontsize=8, fontweight="bold", color=COLORS[model])
ax.set_xlabel("Accuracy Score", fontsize=12)
ax.set_ylabel("Adaptation Score", fontsize=12)
ax.set_title("Scatter — Accuracy vs Adaptation\n(diamonds = centroid)", fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=9, markerscale=0.8)
save(fig, "09_scatter_accuracy_adaptation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. BUBBLE CHART — Accuracy × Adaptation, size = mean score
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for model in MODELS:
    row_m = means.loc[model]
    x = row_m["accuracy_score"]
    y = row_m["adaptation_score"]
    size = overall[model] * 2000
    ax.scatter(x, y, s=size, color=COLORS[model], alpha=0.75,
               edgecolors="white", linewidths=2, zorder=4)
    ax.text(x, y + 0.015, model, ha="center", fontsize=9,
            fontweight="bold", color=COLORS[model])
# Size legend
for s, lbl in zip([0.6, 0.75, 0.92], ["0.6", "0.75", "0.92"]):
    ax.scatter([], [], s=s*2000, color="gray", alpha=0.5, label=f"Mean={lbl}")
ax.set_xlabel("Accuracy Score", fontsize=12)
ax.set_ylabel("Adaptation Score", fontsize=12)
ax.set_title("Bubble Chart\nAccuracy × Adaptation (bubble size = overall mean)", fontsize=13, fontweight="bold", pad=10)
ax.legend(title="Overall Mean", fontsize=9, loc="lower right")
save(fig, "10_bubble.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. PIE CHART — Mean score share
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
# Pie
ax = axes[0]
explode = [0.06 if m == "CognitiveMorph" else 0 for m in MODELS]
wedges, texts, autotexts = ax.pie(
    overall[MODELS], labels=MODELS,
    colors=PALETTE, autopct="%1.1f%%",
    startangle=90, explode=explode,
    textprops={"fontsize": 9},
    wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
for at in autotexts:
    at.set_fontsize(8.5)
ax.set_title("Pie — Mean Score Share", fontsize=13, fontweight="bold")
# Donut
ax2 = axes[1]
wedges2, texts2, autotexts2 = ax2.pie(
    overall[MODELS], labels=MODELS,
    colors=PALETTE, autopct="%1.1f%%",
    startangle=90, pctdistance=0.78,
    textprops={"fontsize": 9},
    wedgeprops={"linewidth": 1.5, "edgecolor": "white", "width": 0.55})
circle = plt.Circle((0, 0), 0.45, color="white")
ax2.add_patch(circle)
ax2.text(0, 0, "Mean\nScore", ha="center", va="center",
         fontsize=10, fontweight="bold", color="gray")
ax2.set_title("Donut — Mean Score Share", fontsize=13, fontweight="bold")
save(fig, "11_pie_donut.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12. STACKED BAR CHART — cumulative metric contributions
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
cmap_stack = plt.cm.get_cmap("tab10", len(METRICS))
bottom = np.zeros(len(MODELS))
bar_colors = ["#185FA5", "#1D9E75", "#BA7517", "#D4537E", "#888780"]
metric_colors = ["#4472C4", "#70AD47", "#FF914D", "#7030A0", "#FF3B6E"]
for i, (metric, label) in enumerate(zip(METRICS, LABELS)):
    vals = means[metric].reindex(MODELS).values
    ax.bar(MODELS, vals, bottom=bottom, label=label,
           color=metric_colors[i], alpha=0.85, zorder=3, width=0.55)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.04:
            ax.text(j, b + v/2, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
    bottom += vals
ax.set_ylabel("Cumulative Score")
ax.set_title("Stacked Bar — Metric Contributions per Architecture", fontsize=14, fontweight="bold", pad=12)
ax.legend(ncol=5, fontsize=9, loc="upper right")
ax.set_ylim(0, 5.2)
save(fig, "12_stacked_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. ERROR BAR CHART — Mean ± 1 STD per metric
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, (metric, label) in enumerate(zip(METRICS, LABELS)):
    ax = axes[i]
    y = means[metric].reindex(MODELS).values
    e = stds[metric].reindex(MODELS).values
    x = np.arange(len(MODELS))
    ax.errorbar(x, y, yerr=e, fmt="o", capsize=6, capthick=2,
                color=[COLORS[m] for m in MODELS][0],
                ecolor="gray", markersize=8, zorder=4)
    for xi, yi, model in zip(x, y, MODELS):
        ax.scatter([xi], [yi], color=COLORS[model], s=80, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[:5] for m in MODELS], fontsize=8)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score")
axes[-1].set_visible(False)
fig.suptitle("Error Bar Charts — Mean ± STD per Metric", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "13_errorbars.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14. STRIP / SWARM PLOT — raw data points per metric
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
for ax, metric, label in zip(axes, METRICS, LABELS):
    for j, model in enumerate(MODELS):
        vals = df[df["model_type"] == model][metric].values
        jitter = np.random.uniform(-0.2, 0.2, len(vals))
        ax.scatter(jitter + j, vals, color=COLORS[model],
                   alpha=0.35, s=18, zorder=3)
        ax.hlines(vals.mean(), j-0.35, j+0.35, color=COLORS[model],
                  lw=2.5, zorder=4)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[:4] for m in MODELS], fontsize=7.5)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylim(0.25, 1.05)
fig.suptitle("Strip Plots — Raw Data Points per Architecture (line = mean)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "14_strip_swarm.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15. PAIR PLOT (scatter matrix) — CognitiveMorph vs rest
# ─────────────────────────────────────────────────────────────────────────────
pair_df = df[METRICS + ["model_type"]].copy()
pair_df["model_type"] = pair_df["model_type"].map(lambda m: m if m in MODELS else m)
fig, axes = plt.subplots(5, 5, figsize=(14, 13))
for i, m1 in enumerate(METRICS):
    for j, m2 in enumerate(METRICS):
        ax = axes[i][j]
        if i == j:
            for model in MODELS:
                sub = pair_df[df["model_type"] == model][m1]
                ax.hist(sub, bins=12, color=COLORS[model], alpha=0.55, density=True)
        else:
            for model in MODELS:
                sub = pair_df[df["model_type"] == model]
                ax.scatter(sub[m2], sub[m1], color=COLORS[model],
                           alpha=0.3, s=12)
        if i == 4:
            ax.set_xlabel(LABELS[j], fontsize=8)
        if j == 0:
            ax.set_ylabel(LABELS[i], fontsize=8)
        ax.tick_params(labelsize=6)
fig.suptitle("Pair / Scatter Matrix — All Metric Combinations",
             fontsize=14, fontweight="bold", y=1.01)
handles = [mpatches.Patch(color=COLORS[m], label=m) for m in MODELS]
fig.legend(handles=handles, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
save(fig, "15_pair_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 16. CORRELATION HEATMAP — feature × metric correlations per model
# ─────────────────────────────────────────────────────────────────────────────
features = ["sensor_temperature","pressure","vibration","energy_consumption",
            "task_complexity","relational_dependency_score","human_interaction_level"]
feat_labels = ["Temp","Pressure","Vibration","Energy","Complexity","Relational","Human"]

fig, axes = plt.subplots(1, len(MODELS), figsize=(20, 5))
for ax, model in zip(axes, MODELS):
    sub = df[df["model_type"] == model][features + METRICS]
    corr = sub.corr().loc[features, METRICS].values
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(feat_labels, fontsize=7.5)
    ax.set_title(model, fontsize=10, fontweight="bold")
    for r in range(len(features)):
        for c in range(len(METRICS)):
            ax.text(c, r, f"{corr[r,c]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")
fig.suptitle("Correlation Heatmaps — Feature × Metric per Architecture",
             fontsize=13, fontweight="bold")
plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Pearson r")
save(fig, "16_correlation_heatmaps.png")


# ─────────────────────────────────────────────────────────────────────────────
# 17. HISTOGRAM — distribution of accuracy_score for all models
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, model in enumerate(MODELS):
    ax = axes[i]
    vals = df[df["model_type"] == model]["accuracy_score"].values
    ax.hist(vals, bins=15, color=COLORS[model], alpha=0.75, edgecolor="white", zorder=3)
    ax.axvline(vals.mean(), color="black", lw=2, ls="--", label=f"Mean {vals.mean():.3f}")
    ax.set_title(model, fontsize=11, fontweight="bold")
    ax.set_xlabel("Accuracy Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
axes[-1].set_visible(False)
fig.suptitle("Histograms — Accuracy Score Distribution per Architecture",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "17_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# 18. KDE (Density) PLOT — overlaid score distributions
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)
for ax, metric, label in zip(axes, METRICS, LABELS):
    for model in MODELS:
        vals = df[df["model_type"] == model][metric].values
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, bw_method=0.3)
        xs = np.linspace(vals.min()-0.05, vals.max()+0.05, 150)
        ax.plot(xs, kde(xs), lw=2.2, label=model, color=COLORS[model])
        ax.fill_between(xs, kde(xs), alpha=0.08, color=COLORS[model])
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
axes[-1].legend(fontsize=7.5, loc="upper right")
fig.suptitle("KDE Density Plots — Score Distributions per Metric",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "18_kde_density.png")


# ─────────────────────────────────────────────────────────────────────────────
# 19. LOLLIPOP / STEM CHART — mean score per model per metric
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
for ax, metric, label in zip(axes, METRICS, LABELS):
    vals = means[metric].reindex(MODELS).values
    y = np.arange(len(MODELS))
    ax.hlines(y, 0, vals, colors=[COLORS[m] for m in MODELS], lw=2.5)
    for yi, val, model in zip(y, vals, MODELS):
        ax.scatter(val, yi, color=COLORS[model], s=100, zorder=5)
        ax.text(val + 0.015, yi, f"{val:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0.3, 1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(MODELS, fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
fig.suptitle("Lollipop / Stem Charts — Mean Score per Architecture",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "19_lollipop.png")


# ─────────────────────────────────────────────────────────────────────────────
# 20. AREA / FILLED LINE — metric trend across ordered models
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(METRICS))
for model in MODELS:
    y = means.loc[model, METRICS].values
    ax.plot(x, y, "o-", lw=2, color=COLORS[model], label=model)
    ax.fill_between(x, 0.3, y, alpha=0.07, color=COLORS[model])
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylim(0.3, 1.05)
ax.set_ylabel("Mean Score")
ax.set_title("Area / Filled Line Chart — Metric Profiles", fontsize=14, fontweight="bold", pad=12)
ax.legend(ncol=5, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.16))
save(fig, "20_area_filled_line.png")


# ─────────────────────────────────────────────────────────────────────────────
# 21. ANOVA / STATISTICAL SIGNIFICANCE BAR
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
p_vals = []
for metric in METRICS:
    groups = [df[df["model_type"] == m][metric].values for m in MODELS]
    _, p = f_oneway(*groups)
    p_vals.append(-np.log10(p))
bars = ax.bar(LABELS, p_vals, color=["#185FA5" if v > 1.3 else "#888780" for v in p_vals],
              width=0.55, zorder=3)
ax.axhline(1.3, color="red", lw=1.5, ls="--", label="p=0.05 threshold")
ax.axhline(2.0, color="orange", lw=1, ls=":", alpha=0.7, label="p=0.01 threshold")
for bar, val in zip(bars, p_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("-log10(p-value) from ANOVA")
ax.set_title("Statistical Significance (ANOVA) per Metric", fontsize=14, fontweight="bold", pad=12)
ax.legend(fontsize=9)
save(fig, "21_anova_significance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 22. PARALLEL COORDINATES
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(METRICS))
for model in MODELS:
    y = means.loc[model, METRICS].values
    ax.plot(x, y, "o-", lw=2.5, color=COLORS[model], label=model,
            markersize=7, alpha=0.85)
    ax.text(x[-1] + 0.07, y[-1], model, va="center",
            fontsize=8.5, color=COLORS[model], fontweight="bold")
for xi, label in zip(x, LABELS):
    ax.axvline(xi, color="gray", lw=0.6, ls="--", alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylim(0.3, 1.05)
ax.set_xlim(-0.1, len(METRICS) - 0.5)
ax.set_ylabel("Score")
ax.set_title("Parallel Coordinates — Architecture Profiles", fontsize=14, fontweight="bold", pad=12)
ax.legend(fontsize=8.5, loc="lower left")
save(fig, "22_parallel_coordinates.png")


# ─────────────────────────────────────────────────────────────────────────────
# 23. WATERFALL — CognitiveMorph advantage over mean of others
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
others_mean = means.drop("CognitiveMorph")[METRICS].mean()
cm_means    = means.loc["CognitiveMorph", METRICS]
delta       = (cm_means - others_mean).values
colors_wf   = ["#185FA5" if d > 0 else "#D4537E" for d in delta]
bars = ax.bar(LABELS, delta, color=colors_wf, width=0.55, zorder=3, alpha=0.85)
ax.axhline(0, color="black", lw=1)
for bar, val in zip(bars, delta):
    ypos = val + 0.003 if val >= 0 else val - 0.013
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"+{val:.3f}" if val >= 0 else f"{val:.3f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=10, fontweight="bold",
            color="#185FA5" if val >= 0 else "#D4537E")
ax.set_ylabel("Score Delta vs Average of Other Models")
ax.set_title("Waterfall — CognitiveMorph Advantage per Metric", fontsize=14, fontweight="bold", pad=12)
pos_patch = mpatches.Patch(color="#185FA5", label="CognitiveMorph leads")
neg_patch = mpatches.Patch(color="#D4537E", label="CognitiveMorph trails")
ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
save(fig, "23_waterfall_advantage.png")


# ─────────────────────────────────────────────────────────────────────────────
# 24. COMPREHENSIVE DASHBOARD (1 figure, all key charts)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# A. Ranked bar
ax_a = fig.add_subplot(gs[0, 0])
ranked = overall.reset_index()
ranked.columns = ["Model", "Score"]
bars = ax_a.bar(ranked["Model"], ranked["Score"],
                color=[COLORS[m] for m in ranked["Model"]],
                width=0.55, zorder=3)
for bar, val in zip(bars, ranked["Score"]):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
              f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
ax_a.set_ylim(0.4, 1.05)
ax_a.set_title("Overall Ranking", fontweight="bold")
ax_a.tick_params(axis='x', labelrotation=20, labelsize=8)

# B. Heatmap
ax_b = fig.add_subplot(gs[0, 1:])
heat_data = means[METRICS].copy()
heat_data.columns = LABELS
sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax_b, vmin=0.45, vmax=1.0,
            cbar_kws={"shrink": 0.7})
ax_b.set_title("Heatmap — Mean Scores", fontweight="bold")
ax_b.tick_params(axis='y', rotation=0)

# C. Radar
ax_c = fig.add_subplot(gs[1, 0], polar=True)
angles = np.linspace(0, 2*np.pi, len(METRICS), endpoint=False).tolist()
angles += angles[:1]
ax_c.set_theta_offset(np.pi/2)
ax_c.set_theta_direction(-1)
ax_c.set_xticks(angles[:-1])
ax_c.set_xticklabels(LABELS, fontsize=7.5)
ax_c.set_ylim(0.3, 1.0)
for model in MODELS:
    vals = means.loc[model, METRICS].tolist() + [means.loc[model, METRICS[0]]]
    ax_c.plot(angles, vals, "o-", lw=1.8, color=COLORS[model], markersize=4)
    ax_c.fill(angles, vals, alpha=0.06, color=COLORS[model])
ax_c.set_title("Radar Chart", fontweight="bold", pad=18)

# D. Box plots (accuracy only)
ax_d = fig.add_subplot(gs[1, 1])
data_list = [df[df["model_type"]==m]["accuracy_score"].values for m in MODELS]
bp = ax_d.boxplot(data_list, patch_artist=True, medianprops={"color":"black","lw":2})
for patch, model in zip(bp["boxes"], MODELS):
    patch.set_facecolor(COLORS[model])
    patch.set_alpha(0.75)
ax_d.set_xticklabels([m[:5] for m in MODELS], fontsize=8)
ax_d.set_title("Box Plot — Accuracy", fontweight="bold")

# E. Scatter accuracy vs adaptation
ax_e = fig.add_subplot(gs[1, 2])
for model in MODELS:
    sub = df[df["model_type"]==model]
    ax_e.scatter(sub["accuracy_score"], sub["adaptation_score"],
                 color=COLORS[model], alpha=0.45, s=20, label=model)
    mx, my = sub["accuracy_score"].mean(), sub["adaptation_score"].mean()
    ax_e.scatter(mx, my, color=COLORS[model], s=120, marker="D",
                 edgecolors="white", lw=1.5, zorder=5)
ax_e.set_xlabel("Accuracy", fontsize=9)
ax_e.set_ylabel("Adaptation", fontsize=9)
ax_e.set_title("Scatter: Accuracy × Adaptation", fontweight="bold")

# F. Line + band
ax_f = fig.add_subplot(gs[2, :2])
x = np.arange(len(METRICS))
for model in MODELS:
    y  = means.loc[model, METRICS].values
    sd = stds.loc[model, METRICS].values
    ax_f.plot(x, y, "o-", lw=2, label=model, color=COLORS[model], markersize=6)
    ax_f.fill_between(x, y-sd, y+sd, alpha=0.1, color=COLORS[model])
ax_f.set_xticks(x)
ax_f.set_xticklabels(LABELS, fontsize=9)
ax_f.set_ylim(0.3, 1.05)
ax_f.set_title("Line + STD Band — Metric Profiles", fontweight="bold")
ax_f.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.22))

# G. Waterfall
ax_g = fig.add_subplot(gs[2, 2])
bars_wf = ax_g.bar(LABELS, delta, color=colors_wf, width=0.6, zorder=3, alpha=0.85)
ax_g.axhline(0, color="black", lw=1)
ax_g.set_title("CognitiveMorph\nAdvantage", fontweight="bold")
ax_g.tick_params(axis='x', labelrotation=30, labelsize=8)

# H. ANOVA
ax_h = fig.add_subplot(gs[3, 0])
ax_h.bar(LABELS, p_vals, color=["#185FA5" if v > 1.3 else "#888780" for v in p_vals],
         width=0.55, zorder=3)
ax_h.axhline(1.3, color="red", lw=1.5, ls="--")
ax_h.set_ylabel("-log10(p)")
ax_h.set_title("ANOVA Significance", fontweight="bold")
ax_h.tick_params(axis='x', labelrotation=30, labelsize=8)

# I. Pie
ax_i = fig.add_subplot(gs[3, 1])
ax_i.pie(overall[MODELS], labels=[m[:4] for m in MODELS],
         colors=PALETTE, autopct="%1.0f%%", startangle=90,
         textprops={"fontsize": 8},
         wedgeprops={"linewidth": 1.5, "edgecolor":"white"})
ax_i.set_title("Score Share (Donut)", fontweight="bold")
circle2 = plt.Circle((0,0),0.5,color="white")
ax_i.add_patch(circle2)

# J. Lollipop (overall)
ax_j = fig.add_subplot(gs[3, 2])
y = np.arange(len(MODELS))
vals_ov = overall[MODELS].values
ax_j.hlines(y, 0, vals_ov, colors=PALETTE, lw=3)
for yi, val, model in zip(y, vals_ov, MODELS):
    ax_j.scatter(val, yi, color=COLORS[model], s=90, zorder=5)
    ax_j.text(val + 0.008, yi, f"{val:.3f}", va="center", fontsize=8.5)
ax_j.set_xlim(0.4, 1.05)
ax_j.set_yticks(y)
ax_j.set_yticklabels(MODELS, fontsize=9)
ax_j.invert_yaxis()
ax_j.set_title("Lollipop — Overall Score", fontweight="bold")

# Legend row
handles = [mpatches.Patch(color=COLORS[m], label=m) for m in MODELS]
fig.legend(handles=handles, loc="upper center", ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, 1.01),
           frameon=True, edgecolor="gray")

fig.suptitle("CognitiveMorph Architecture Evaluation — Complete Dashboard",
             fontsize=17, fontweight="bold", y=1.03)

save(fig, "00_full_dashboard.png")

print("\nAll 24 charts saved to /mnt/user-data/outputs/")