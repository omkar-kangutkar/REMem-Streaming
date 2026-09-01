import os
import sys
import json
import time
import shutil
import csv
import glob
import pickle
import math

sys.path.insert(0, "src")
sys.path.insert(0, "examples")

from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem
from realtalk import get_sessions


DATASET_DIR = "reproduce/dataset/realtalk"
OUTPUT_BASE = "outputs/benchmark_naive_reindex_all_realtalk"
CSV_FILE = "naive_reindex_all_realtalk_results.csv"

os.makedirs(OUTPUT_BASE, exist_ok=True)

chat_files = sorted(
    glob.glob(os.path.join(DATASET_DIR, "Chat_*.json"))
)

print(f"Found {len(chat_files)} REALTALK conversations")

if len(chat_files) != 10:
    raise RuntimeError(
        f"Expected 10 REALTALK conversations, found {len(chat_files)}"
    )


def checkpoints_for(n):
    raw = [
        max(1, round(n * 0.25)),
        max(1, round(n * 0.50)),
        max(1, round(n * 0.75)),
        n,
    ]
    return sorted(set(raw))


def load_chunk_count(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "hash_ids" in obj:
        return len(obj["hash_ids"])

    return -1


results = []

for ci, chat_file in enumerate(chat_files, start=1):
    chat_name = os.path.splitext(
        os.path.basename(chat_file)
    )[0]

    print("\n" + "#" * 80)
    print(f"[{ci}/10] {chat_name}")
    print("#" * 80)

    with open(chat_file, "r") as f:
        data = json.load(f)

    sessions = get_sessions(data)
    n_total = len(sessions)
    checkpoints = checkpoints_for(n_total)

    print(f"Total sessions: {n_total}")
    print(f"Checkpoints: {checkpoints}")

    for n_sessions in checkpoints:
        print("\n" + "=" * 70)
        print(
            f"FULL REBUILD: {chat_name}, "
            f"first {n_sessions}/{n_total} sessions"
        )
        print("=" * 70)

        prefix = sessions[:n_sessions]

        working_dir = os.path.join(
            OUTPUT_BASE,
            chat_name,
            f"rebuild_{n_sessions:02d}_sessions",
        )

        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)

        dataset_name = (
            f"realtalk_naive_all_{chat_name}"
        )

        config = BaseConfig(
            llm_name="google/gemini-2.5-flash-lite",
            llm_base_url="https://openrouter.ai/api/v1",
            embedding_model_name=(
                "sentence-transformers/all-mpnet-base-v2"
            ),
            extract_method="episodic_gist",
            force_index_from_scratch=True,
            force_openie_from_scratch=True,
            dataset=dataset_name,
            preprocess_chunk_func="by_session",
            retrieval_top_k=200,
            linking_top_k=5,
            qa_top_k=3,
            graph_type=(
                "facts_and_sim_passage_node_unidirectional"
            ),
            agent_max_steps=3,
        )

        remem = ReMem(
            global_config=config,
            working_dir=working_dir,
        )

        start = time.perf_counter()

        remem.index(prefix)

        elapsed = time.perf_counter() - start

        graph_nodes = remem.graph.vcount()
        graph_edges = remem.graph.ecount()

        chunk_file = os.path.join(
            working_dir,
            "chunk_embeddings",
            "vdb_chunk.pkl",
        )

        chunk_count = (
            load_chunk_count(chunk_file)
            if os.path.exists(chunk_file)
            else -1
        )

        valid = chunk_count == n_sessions

        row = {
            "chat": chat_name,
            "total_sessions": n_total,
            "checkpoint_sessions": n_sessions,
            "fraction": n_sessions / n_total,
            "wall_time_seconds": elapsed,
            "chunks": chunk_count,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "valid": valid,
        }

        results.append(row)

        print(
            f"Result: {elapsed:.2f}s | "
            f"chunks={chunk_count} | "
            f"graph={graph_nodes}/{graph_edges} | "
            f"valid={valid}"
        )

        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "chat",
                    "total_sessions",
                    "checkpoint_sessions",
                    "fraction",
                    "wall_time_seconds",
                    "chunks",
                    "graph_nodes",
                    "graph_edges",
                    "valid",
                ],
            )
            writer.writeheader()
            writer.writerows(results)


print("\n" + "=" * 80)
print("ALL REALTALK FULL-REBUILD BENCHMARK COMPLETE")
print("=" * 80)

print(f"Rows collected: {len(results)}")
print(f"CSV: {CSV_FILE}")

invalid = [r for r in results if not r["valid"]]

print(f"Invalid rows: {len(invalid)}")

for r in results:
    print(
        f'{r["chat"]} | '
        f'{r["checkpoint_sessions"]}/{r["total_sessions"]} sessions | '
        f'{r["wall_time_seconds"]:.2f}s | '
        f'chunks={r["chunks"]} | '
        f'valid={r["valid"]}'
    )
