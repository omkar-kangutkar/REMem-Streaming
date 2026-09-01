import pandas as pd
import numpy as np

stream = pd.read_csv("streaming_latency_all_realtalk_FINAL.csv")
rebuild = pd.read_csv("naive_reindex_all_realtalk_FINAL.csv")

paired = rebuild.merge(
    stream[
        [
            "chat",
            "session_number",
            "latency_seconds",
            "messages",
            "graph_nodes",
            "graph_edges",
            "chunks",
            "valid",
        ]
    ],
    left_on=["chat", "checkpoint_sessions"],
    right_on=["chat", "session_number"],
    how="left",
    suffixes=("_rebuild", "_stream"),
)

paired = paired.sort_values(["chat", "checkpoint_sessions"]).copy()
paired["stage_num"] = paired.groupby("chat").cumcount() + 1
paired["stage"] = paired["stage_num"].map(
    {1: "25%", 2: "50%", 3: "75%", 4: "100%"}
)

paired["seconds_saved"] = (
    paired["wall_time_seconds"] - paired["latency_seconds"]
)

paired["percent_reduction"] = (
    paired["seconds_saved"] / paired["wall_time_seconds"] * 100
)

paired["speedup_x"] = (
    paired["wall_time_seconds"] / paired["latency_seconds"]
)

paired.to_csv("paired_streaming_vs_reindex_FINAL.csv", index=False)

summary = paired.groupby("stage", sort=False).agg(
    n=("chat", "count"),
    rebuild_mean=("wall_time_seconds", "mean"),
    rebuild_median=("wall_time_seconds", "median"),
    streaming_mean=("latency_seconds", "mean"),
    streaming_median=("latency_seconds", "median"),
    mean_seconds_saved=("seconds_saved", "mean"),
    mean_percent_reduction=("percent_reduction", "mean"),
    mean_speedup=("speedup_x", "mean"),
).reset_index()

summary.to_csv("paired_latency_summary_FINAL.csv", index=False)

print(summary.to_string(index=False))

print("\nOverall")
print(f"Paired points: {len(paired)}")
print(f"Rebuild mean: {paired['wall_time_seconds'].mean():.2f}s")
print(f"Streaming mean: {paired['latency_seconds'].mean():.2f}s")
print(f"Mean reduction: {paired['percent_reduction'].mean():.2f}%")
print(f"Mean speedup: {paired['speedup_x'].mean():.2f}x")
print(
    f"Streaming faster: {(paired['seconds_saved'] > 0).sum()}/{len(paired)} checkpoints"
)
print(
    f"Rebuild P95: {np.percentile(paired['wall_time_seconds'], 95):.2f}s"
)
print(
    f"Streaming P95: {np.percentile(paired['latency_seconds'], 95):.2f}s"
)
