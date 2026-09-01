import sys
import os
import shutil
import json
import glob
import csv
import pickle

sys.path.insert(0, "src")
sys.path.insert(0, "examples")

from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem
from streaming_indexer import StreamingIndexer
from realtalk import get_sessions


DATASET_DIR = "reproduce/dataset/realtalk"

OUTPUT_BASE = (
    "outputs/benchmark_streaming_latency_all_realtalk"
)

CSV_FILE = "streaming_latency_all_realtalk_FINAL.csv"

os.makedirs(OUTPUT_BASE, exist_ok=True)

chat_files = sorted(
    glob.glob(
        os.path.join(DATASET_DIR, "Chat_*.json")
    )
)

print(f"Found {len(chat_files)} REALTALK conversations")

if len(chat_files) != 10:
    raise RuntimeError(
        f"Expected 10 REALTALK chats, found {len(chat_files)}"
    )


FIELDNAMES = [
    "chat",
    "session_number",
    "total_sessions",
    "progress",
    "messages",
    "latency_seconds",
    "indexed",
    "failed",
    "graph_nodes",
    "graph_edges",
    "chunks",
    "valid",
]


def save_rows(rows):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def get_chunk_count(working_dir):
    path = os.path.join(
        working_dir,
        "chunk_embeddings",
        "vdb_chunk.pkl",
    )

    if not os.path.exists(path):
        return -1

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if (
            isinstance(obj, dict)
            and "hash_ids" in obj
        ):
            return len(obj["hash_ids"])

    except Exception as e:
        print(
            f"WARNING: could not read chunk count: {e}"
        )

    return -1


all_rows = []
conversation_summaries = []


for ci, chat_file in enumerate(
    chat_files,
    start=1,
):
    chat_name = os.path.splitext(
        os.path.basename(chat_file)
    )[0]

    print("\n" + "#" * 80)
    print(
        f"[{ci}/{len(chat_files)}] "
        f"{chat_name}"
    )
    print("#" * 80)

    with open(chat_file, "r") as f:
        data = json.load(f)

    sessions = get_sessions(data)
    total_sessions = len(sessions)

    print(f"Sessions: {total_sessions}")

    working_dir = os.path.join(
        OUTPUT_BASE,
        (
            f"{chat_name}_"
            "google_gemini-2.5-flash-lite"
        ),
    )

    if os.path.exists(working_dir):
        print(
            "Removing previous benchmark index: "
            f"{working_dir}"
        )
        shutil.rmtree(working_dir)

    # Fresh namespace for this latency benchmark.
    dataset_name = (
        f"realtalk_stream_latency_{chat_name}"
    )

    config = BaseConfig(
        llm_name="google/gemini-2.5-flash-lite",
        llm_base_url=(
            "https://openrouter.ai/api/v1"
        ),
        embedding_model_name=(
            "sentence-transformers/"
            "all-mpnet-base-v2"
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
            "facts_and_sim_passage_node_"
            "unidirectional"
        ),
        agent_max_steps=3,
    )

    remem = ReMem(
        global_config=config,
        working_dir=working_dir,
    )

    indexer = StreamingIndexer(
        remem_instance=remem,
        log_latency=True,
    )

    indexer.start()

    for si, session in enumerate(
        sessions,
        start=1,
    ):
        print(
            f"\nProcessing session "
            f"{si}/{total_sessions} "
            f"(messages={len(session)})"
        )

        previous_latency_count = len(
            indexer.latency_log
        )

        indexer.add_session(session)
        indexer.wait_until_idle()

        if (
            len(indexer.latency_log)
            <= previous_latency_count
        ):
            latency = -1.0
            print(
                "WARNING: no latency value "
                "recorded for this session"
            )
        else:
            latency = float(
                indexer.latency_log[-1]
            )

        stats = indexer.graph_size

        chunks = get_chunk_count(
            working_dir
        )

        valid = (
            chunks == si
            and indexer.num_failed == 0
            and latency >= 0
        )

        row = {
            "chat": chat_name,
            "session_number": si,
            "total_sessions": total_sessions,
            "progress": si / total_sessions,
            "messages": len(session),
            "latency_seconds": latency,
            "indexed": indexer.num_indexed,
            "failed": indexer.num_failed,
            "graph_nodes": stats["nodes"],
            "graph_edges": stats["edges"],
            "chunks": chunks,
            "valid": valid,
        }

        all_rows.append(row)

        # Save after EVERY session so results
        # survive interruption.
        save_rows(all_rows)

        print(
            f"STREAM RESULT | "
            f"{chat_name} | "
            f"{si}/{total_sessions} | "
            f"{latency:.2f}s | "
            f"chunks={chunks} | "
            f"graph="
            f'{stats["nodes"]}/'
            f'{stats["edges"]} | '
            f"valid={valid}"
        )

        if indexer.num_failed > 0:
            print(
                "ERROR: streaming failure detected."
            )
            indexer.stop()
            raise RuntimeError(
                f"Failure in {chat_name} "
                f"after session {si}"
            )

        if chunks != si:
            print(
                "ERROR: chunk/session mismatch."
            )
            indexer.stop()
            raise RuntimeError(
                f"{chat_name}: expected "
                f"{si} chunks, got {chunks}"
            )

    indexer.stop()

    chat_latencies = [
        r["latency_seconds"]
        for r in all_rows
        if r["chat"] == chat_name
    ]

    conversation_summaries.append(
        {
            "chat": chat_name,
            "sessions": total_sessions,
            "avg_latency": (
                sum(chat_latencies)
                / len(chat_latencies)
            ),
            "min_latency": min(
                chat_latencies
            ),
            "max_latency": max(
                chat_latencies
            ),
        }
    )

    print("\n--- Conversation complete ---")
    print(
        f"Average latency: "
        f"{conversation_summaries[-1]['avg_latency']:.2f}s"
    )
    print(
        f"Min latency: "
        f"{conversation_summaries[-1]['min_latency']:.2f}s"
    )
    print(
        f"Max latency: "
        f"{conversation_summaries[-1]['max_latency']:.2f}s"
    )


print("\n" + "=" * 80)
print(
    "ALL REALTALK STREAMING "
    "LATENCY BENCHMARK COMPLETE"
)
print("=" * 80)

print(f"Rows: {len(all_rows)}")

invalid = [
    r for r in all_rows
    if not r["valid"]
]

print(f"Invalid rows: {len(invalid)}")
print(f"CSV: {CSV_FILE}")

all_latencies = [
    r["latency_seconds"]
    for r in all_rows
]

if all_latencies:
    print(
        f"Overall mean latency: "
        f"{sum(all_latencies) / len(all_latencies):.2f}s"
    )
    print(
        f"Overall min latency: "
        f"{min(all_latencies):.2f}s"
    )
    print(
        f"Overall max latency: "
        f"{max(all_latencies):.2f}s"
    )

print("\nPer conversation:")

for s in conversation_summaries:
    print(
        f'{s["chat"]} | '
        f'n={s["sessions"]} | '
        f'avg={s["avg_latency"]:.2f}s | '
        f'min={s["min_latency"]:.2f}s | '
        f'max={s["max_latency"]:.2f}s'
    )
