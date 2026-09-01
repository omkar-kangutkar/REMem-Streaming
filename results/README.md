# Final result files

This directory is reserved for the lightweight result files used in the final dissertation analysis.

The following original CSV files from the final experiment must be copied here from the Surrey experiment machine:

- `streaming_latency_all_realtalk_FINAL.csv`
- `naive_reindex_all_realtalk_FINAL.csv`
- `paired_streaming_vs_reindex_FINAL.csv`
- `paired_latency_summary_FINAL.csv`

Do not recreate these CSVs from the dissertation summary. The files committed here should be the original outputs used for the final analysis.

After upload, `paired_latency_analysis.py` should reproduce the final matched comparison:

- 40 matched checkpoints
- mean full-reindex latency: 68.53 s
- mean streaming latency: 28.42 s
- mean latency reduction: 57.82%
- mean speedup: 2.90x
- streaming faster at 40/40 matched checkpoints

The complete streaming benchmark should contain 219 REALTALK session updates with zero structurally invalid rows.