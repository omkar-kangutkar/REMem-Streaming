# Final submission synchronization checklist

This checklist records the files that must match the final dissertation **LLM Agents with Incremental Episodic Memory Indexing** before the repository is treated as the final submission version.

## Verified present in the repository

- `README.md` — updated to describe the final dissertation method/results
- `streaming_indexer.py` — present, but exact final version still needs synchronization check
- `evaluate_streaming.py`
- `stream_realtalk.py` — older experiment script; final dissertation refers to `stream_realtalk_v3.py`
- `src/remem/remem.py`
- `src/remem/rag_strategies/episodic_gist_strategy.py`
- `src/remem/information_extraction/episodic_gist_extraction_openai.py`
- `src/remem/llm/openai_gpt.py`
- `requirements.txt`
- `pyproject.toml`
- `reproduce/dataset/locomo/`
- `reproduce/dataset/realtalk/`
- `hnsw_synonymy.py` — exploratory only, not part of final latency claim

## Final dissertation files that are currently missing from GitHub

The final dissertation explicitly lists the following files as part of the reproducibility package. They are not currently present on the main branch and should be copied from the exact final experiment directory on the Surrey machine rather than reconstructed from memory.

### Core scripts

- [ ] `stream_realtalk_v3.py`
- [ ] `benchmark_streaming_latency_all_realtalk.py`
- [ ] `benchmark_naive_reindex_all_realtalk.py`
- [ ] `paired_latency_analysis.py`
- [ ] `generate_final_figures.py`

### Final result files

- [ ] `results/streaming_latency_all_realtalk_FINAL.csv`
- [ ] `results/naive_reindex_all_realtalk_FINAL.csv`
- [ ] `results/paired_streaming_vs_reindex_FINAL.csv`
- [ ] `results/paired_latency_summary_FINAL.csv`

## Exact code synchronization still required

The current GitHub repository was originally committed before the final corrected experiment. The exact final versions used in the accepted run should therefore be copied from the experiment machine for these files:

- [ ] `streaming_indexer.py`
- [ ] `src/remem/rag_strategies/episodic_gist_strategy.py`
- [ ] `src/remem/remem.py`
- [ ] `src/remem/information_extraction/episodic_gist_extraction_openai.py`
- [ ] `src/remem/llm/openai_gpt.py`

The final dissertation describes these behaviours and the synchronized code should visibly support them:

1. Persistent session history is retained across updates.
2. Incoming preprocessed chunk identifiers are compared with stored chunk identifiers.
3. Only unseen chunks require new gist/fact extraction.
4. Stored chunk count is checked against expected session count during the REALTALK benchmark.
5. Structured extraction failures can be retried with fresh LLM responses.
6. HNSW is not required by the live final `StreamingIndexer` benchmark.

## Final expected experiment values

These values are recorded here only as a consistency check against the actual result CSV files once uploaded.

### REALTALK streaming benchmark

- Sessions: 219
- Structurally invalid rows: 0
- Overall streaming mean: 26.44 s
- Overall minimum: 2.28 s
- Overall maximum: 142.43 s

### Matched full-reindex vs streaming comparison

- Matched checkpoints: 40
- Mean full-reindex latency: 68.53 s
- Mean streaming latency: 28.42 s
- Median full-reindex latency: 69.91 s
- Median streaming latency: 27.01 s
- Mean seconds saved: 40.10 s
- Mean latency reduction: 57.82%
- Mean speedup: 2.90x
- Full-reindex P95: 109.40 s
- Streaming P95: 50.61 s
- Streaming faster: 40/40

### QA quality

- LoCoMo batch: F1 44.15%, BLEU-1 40.77%, EM 27.09%
- LoCoMo streaming: F1 40.20%, BLEU-1 37.10%, EM 26.54%
- REALTALK batch: F1 25.41%, BLEU-1 17.13%, EM 5.50%
- REALTALK streaming: F1 22.83%, BLEU-1 15.63%, EM 5.63%

## Files that should not be presented as the final contribution

The repository contains development/debug scripts such as `test_stream.py`, `test_stream2.py`, `write_hnsw*.py`, `write_indexer.py`, `reindex_conv0.py`, and earlier REALTALK scripts. These may be retained as development history, but the README and submission documentation should direct the examiner to the final reproducibility files listed above.

## Final verification procedure

Before submission:

1. Copy the exact final files from `~/Downloads/REMem-main` on the Surrey experiment machine into the repository.
2. Place the four final CSVs under `results/`.
3. Run `python -m py_compile` on all final Python scripts.
4. Run `paired_latency_analysis.py` over the uploaded final CSVs and confirm the headline values above.
5. Check that no API key, `.env`, output embedding database, `.pkl`, or private credential is staged.
6. Commit with a message such as `Final dissertation code and reproducibility results`.
7. Re-open this checklist and mark every required item complete.

Do not fabricate or regenerate the final CSVs from the dissertation summary. The original files produced by the final experiments are the correct submission artifacts.