# LLM Agents with Incremental Episodic Memory Indexing

This repository contains the implementation and experimental code for the MSc dissertation **LLM Agents with Incremental Episodic Memory Indexing**.

The project extends REMem (Reasoning with Episodic Memory) from mainly offline episodic-memory construction to **persistent session-level streaming indexing**. The final evaluated method keeps memory from earlier conversation sessions, detects conversation chunks that have not yet been indexed, performs new gist/fact extraction only for unseen chunks, and then updates the existing REMem memory state.

## Project idea

A conversation is treated as a sequence of sessions. Each session contains multiple messages. REMem preprocesses the session history into chunks; with the configuration used in this project, chunking is session-based. The streaming method keeps previously constructed memory and compares incoming chunk identifiers with the chunk identifiers already stored. Previously processed chunks are reused, while unseen chunks proceed to new extraction.

The final system therefore changes the **indexing control flow**, not REMem's basic memory representation. Gists, facts, embeddings, the hybrid graph and the existing retrieval/QA process remain based on REMem.

The streaming update can be summarized as:

```text
new session
    -> accumulated conversation history
    -> REMem session preprocessing
    -> compare incoming and stored chunk identifiers
    -> unseen chunks only
    -> gist and fact extraction
    -> embedding/persistent stores
    -> REMem graph/synonymy update
    -> updated persistent memory
```

The system is **not claimed to be fully incremental at every stage**. Session ingestion and new extraction are incremental and persistent, but some graph/synonymy operations still depend on the accumulated memory size.

## Final evaluated configuration

- LLM for extraction: `google/gemini-2.5-flash-lite` through OpenRouter
- Embedding model: `sentence-transformers/all-mpnet-base-v2`
- REMem extraction mode: `episodic_gist`
- Conversation preprocessing: `by_session`
- Main datasets: LoCoMo and REALTALK

The final OpenRouter configuration uses the OpenAI-compatible client with `llm_base_url="https://openrouter.ai/api/v1"`. In the submitted code the credential is read from the `OPENAI_API_KEY` environment variable, so an OpenRouter key can be supplied without placing the secret in source code:

```bash
export OPENAI_API_KEY="your_openrouter_key_here"
```

Do not commit a real API key, `.env` file, token, or other credential.

## Final dissertation results

### Question-answering quality

| Dataset | Method | F1 | BLEU-1 | Exact Match |
|---|---:|---:|---:|---:|
| LoCoMo | Batch REMem | 44.15% | 40.77% | 27.09% |
| LoCoMo | Streaming REMem | 40.20% | 37.10% | 26.54% |
| REALTALK | Batch REMem | 25.41% | 17.13% | 5.50% |
| REALTALK | Streaming REMem | 22.83% | 15.63% | 5.63% |

### Streaming efficiency on REALTALK

The complete streaming benchmark covers **219 sessions** across ten REALTALK conversations and contains **zero structurally invalid rows**.

The primary efficiency comparison uses 40 matched conversation checkpoints (approximately 25%, 50%, 75% and 100% of each conversation):

| Measure | Full re-indexing | Streaming |
|---|---:|---:|
| Mean latency | 68.53 s | 28.42 s |
| Median latency | 69.91 s | 27.01 s |
| P95 latency | 109.40 s | 50.61 s |

This corresponds to a **57.82% mean latency reduction** and a **2.90x mean speedup**. Streaming was faster at all **40/40** matched checkpoints.

## What produced the speedup

The reported speedup comes primarily from:

- persistent reuse of previously constructed memory;
- explicit detection of already indexed versus unseen chunks;
- avoiding repeated gist and fact extraction for old sessions;
- reusing stored embeddings/extraction outputs where appropriate.

`hnsw_synonymy.py` and related HNSW utilities are retained as **exploratory work only**. HNSW was not part of the final live benchmarked `StreamingIndexer` path and must not be interpreted as the source of the reported 57.82% latency reduction.

## Core final-reproduction files

The dissertation identifies the following as the core implementation and analysis files:

```text
streaming_indexer.py
stream_realtalk_v3.py
evaluate_streaming.py
benchmark_streaming_latency_all_realtalk.py
benchmark_naive_reindex_all_realtalk.py
paired_latency_analysis.py
generate_final_figures.py

src/remem/remem.py
src/remem/rag_strategies/episodic_gist_strategy.py
src/remem/information_extraction/episodic_gist_extraction_openai.py
src/remem/llm/openai_gpt.py
```

The final lightweight result files are:

```text
results/streaming_latency_all_realtalk_FINAL.csv
results/naive_reindex_all_realtalk_FINAL.csv
results/paired_streaming_vs_reindex_FINAL.csv
results/paired_latency_summary_FINAL.csv
```

See `FINAL_SUBMISSION_CHECKLIST.md` for exact synchronization status.

## Reproduction logic

The streaming benchmark processes each REALTALK conversation chronologically in one persistent working directory. After each session, the stored session-chunk count is checked against the expected session number. The final accepted run contains all 219 expected updates.

For the full re-indexing baseline, each selected checkpoint starts from a fresh working directory and indexes the complete conversation prefix up to that checkpoint. The paired analysis matches each full-reindex measurement with the streaming measurement from the same conversation and exact session number.

The final paired-analysis headline values expected from the final CSVs are:

- 40 matched checkpoints;
- 68.53 s mean full-reindex latency;
- 28.42 s mean streaming latency;
- 57.82% mean latency reduction;
- 2.90x mean speedup;
- streaming faster at 40/40 checkpoints.

## Reliability and limitations

The final implementation includes retry handling for malformed or incomplete LLM extraction responses. One temporary fact-extraction failure remained unresolved after three attempts during one REALTALK update; the same chunk was successfully recovered in the following session. That update was not one of the 40 matched efficiency checkpoints.

A post-run diagnostic also found zero-valued 1536-dimensional fallback vectors in a generic entity store, while the chunk, gist and fact stores used valid 768-dimensional MPNet vectors. The active evaluated episodic-gist synonymy path used valid gist embeddings. This is documented as a limitation and should be corrected in future work.

## Installation

```bash
git clone https://github.com/omkar-kangutkar/REMem-Streaming.git
cd REMem-Streaming
python -m venv remem_env
source remem_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10+ is recommended.

## Repository structure

```text
src/remem/          REMem source plus project modifications
examples/           dataset-specific REMem scripts
reproduce/dataset/  benchmark datasets used by the repository
experiments/         experimental utilities
results/             lightweight final result files
streaming_indexer.py streaming control layer
```

The original REMem source and datasets are retained so that the relationship between the baseline system and the streaming modifications remains visible.

## Citation and upstream REMem

This work builds on:

> Shu et al., **REMem: Reasoning with Episodic Memory in Language Agents**, ICLR 2026.

Please cite the original REMem paper when using the upstream framework.

## License

The repository retains the upstream REMem Apache License 2.0. Dataset-specific licences and usage conditions remain those of the original dataset providers.