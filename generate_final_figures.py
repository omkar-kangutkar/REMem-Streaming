import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("final_figures")
OUT.mkdir(exist_ok=True)

paired = pd.read_csv("paired_streaming_vs_reindex_FINAL.csv")
summary = pd.read_csv("paired_latency_summary_FINAL.csv")

# -------------------------------------------------
# Figure 1: Mean latency across conversation history
# -------------------------------------------------
stage_order = ["25%", "50%", "75%", "100%"]
summary["stage"] = pd.Categorical(
    summary["stage"],
    categories=stage_order,
    ordered=True
)
summary = summary.sort_values("stage")

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    summary["stage"],
    summary["rebuild_mean"],
    marker="o",
    linewidth=2,
    label="Full re-index"
)

ax.plot(
    summary["stage"],
    summary["streaming_mean"],
    marker="o",
    linewidth=2,
    label="Streaming"
)

ax.set_xlabel("Conversation history checkpoint")
ax.set_ylabel("Mean indexing latency (s)")
ax.set_title("Indexing Latency vs Conversation Growth")
ax.grid(True, alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(OUT / "figure_latency_scalability.png", dpi=300)
fig.savefig(OUT / "figure_latency_scalability.pdf")
plt.close(fig)

# -------------------------------------------------
# Figure 2: Distribution of paired checkpoint latency
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))

data = [
    paired["wall_time_seconds"],
    paired["latency_seconds"]
]

ax.boxplot(
    data,
    tick_labels=["Full re-index", "Streaming"],
    showmeans=True
)

ax.set_ylabel("Latency (s)")
ax.set_title("Latency Distribution Across 40 Matched Checkpoints")
ax.grid(True, axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "figure_latency_distribution.png", dpi=300)
fig.savefig(OUT / "figure_latency_distribution.pdf")
plt.close(fig)

# -------------------------------------------------
# Figure 3: Batch vs Streaming QA quality
# -------------------------------------------------
quality = pd.DataFrame({
    "Metric": ["F1", "BLEU-1", "Exact Match"] * 2,
    "Method": (
        ["Batch", "Batch", "Batch"] +
        ["Streaming", "Streaming", "Streaming"]
    ),
    "Dataset": (
        ["LoCoMo", "LoCoMo", "LoCoMo"] +
        ["LoCoMo", "LoCoMo", "LoCoMo"]
    ),
    "Score": [
        44.15, 40.77, 27.09,
        40.20, 37.10, 26.54
    ]
})

quality_rt = pd.DataFrame({
    "Metric": ["F1", "BLEU-1", "Exact Match"] * 2,
    "Method": (
        ["Batch", "Batch", "Batch"] +
        ["Streaming", "Streaming", "Streaming"]
    ),
    "Dataset": (
        ["REALTALK", "REALTALK", "REALTALK"] +
        ["REALTALK", "REALTALK", "REALTALK"]
    ),
    "Score": [
        25.41, 17.13, 5.50,
        22.83, 15.63, 5.63
    ]
})

quality_all = pd.concat([quality, quality_rt], ignore_index=True)

# LoCoMo plot
locomo = quality_all[quality_all["Dataset"] == "LoCoMo"]

pivot = locomo.pivot(
    index="Metric",
    columns="Method",
    values="Score"
).loc[["F1", "BLEU-1", "Exact Match"]]

fig, ax = plt.subplots(figsize=(7, 4.5))
pivot.plot(kind="bar", ax=ax)

ax.set_ylabel("Score (%)")
ax.set_xlabel("")
ax.set_title("LoCoMo QA Quality: Batch vs Streaming")
ax.tick_params(axis="x", rotation=0)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(title="Method")

fig.tight_layout()
fig.savefig(OUT / "figure_quality_locomo.png", dpi=300)
fig.savefig(OUT / "figure_quality_locomo.pdf")
plt.close(fig)

# REALTALK plot
realtalk = quality_all[quality_all["Dataset"] == "REALTALK"]

pivot = realtalk.pivot(
    index="Metric",
    columns="Method",
    values="Score"
).loc[["F1", "BLEU-1", "Exact Match"]]

fig, ax = plt.subplots(figsize=(7, 4.5))
pivot.plot(kind="bar", ax=ax)

ax.set_ylabel("Score (%)")
ax.set_xlabel("")
ax.set_title("REALTALK QA Quality: Batch vs Streaming")
ax.tick_params(axis="x", rotation=0)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(title="Method")

fig.tight_layout()
fig.savefig(OUT / "figure_quality_realtalk.png", dpi=300)
fig.savefig(OUT / "figure_quality_realtalk.pdf")
plt.close(fig)

print("\nGenerated figures:")
for f in sorted(OUT.iterdir()):
    print(" ", f)
