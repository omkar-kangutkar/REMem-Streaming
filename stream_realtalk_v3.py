import sys
import os
import shutil
import json
import glob

sys.path.insert(0, "src")
sys.path.insert(0, "examples")

from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem
from streaming_indexer import StreamingIndexer
from realtalk import get_sessions


DATASET_DIR = "reproduce/dataset/realtalk"
OUTPUT_BASE = "outputs/streaming_eval_realtalk_v3"

os.makedirs(OUTPUT_BASE, exist_ok=True)

chat_files = sorted(glob.glob(os.path.join(DATASET_DIR, "Chat_*.json")))

print(f"Found {len(chat_files)} REALTALK conversations")

all_latencies = []
conversation_summaries = []

for ci, chat_file in enumerate(chat_files):
    chat_name = os.path.splitext(os.path.basename(chat_file))[0]

    print(f"\n=== [{ci + 1}/{len(chat_files)}] {chat_name} ===")

    with open(chat_file, "r") as f:
        data = json.load(f)

    sessions = get_sessions(data)
    print(f"Sessions: {len(sessions)}")

    working_dir = os.path.join(
        OUTPUT_BASE,
        f"{chat_name}_google_gemini-2.5-flash-lite",
    )

    # Rebuild only inside the new v2 experiment directory.
    if os.path.exists(working_dir):
        print(f"Removing existing v2 index: {working_dir}")
        shutil.rmtree(working_dir)

    # Give every conversation its own cache/OpenIE namespace.
    dataset_name = f"realtalk_stream_v3_{chat_name}"

    config = BaseConfig(
        llm_name="google/gemini-2.5-flash-lite",
        llm_base_url="https://openrouter.ai/api/v1",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        extract_method="episodic_gist",
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        dataset=dataset_name,
        preprocess_chunk_func="by_session",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=3,
        graph_type="facts_and_sim_passage_node_unidirectional",
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

    for si, session in enumerate(sessions):
        print(
            f"Processing session {si + 1}/{len(sessions)} "
            f"(messages={len(session)})"
        )

        indexer.add_session(session)
        indexer.wait_until_idle()

        if indexer.num_failed > 0:
            print(
                f"WARNING: failures detected after session "
                f"{si + 1}: {indexer.num_failed}"
            )

    indexer.stop()

    stats = indexer.graph_size
    latencies = list(indexer.latency_log)
    all_latencies.extend(latencies)

    avg_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    gist_file = os.path.join(
        working_dir,
        "gists_embeddings",
        "vdb_gists.pkl",
    )

    gist_ok = os.path.exists(gist_file) and os.path.getsize(gist_file) > 0

    summary = {
        "chat": chat_name,
        "sessions": len(sessions),
        "indexed": indexer.num_indexed,
        "failed": indexer.num_failed,
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "avg_latency": avg_latency,
        "gist_file_ok": gist_ok,
    }

    conversation_summaries.append(summary)

    print("\n--- Conversation Summary ---")
    print(f"Indexed: {indexer.num_indexed}")
    print(f"Failed: {indexer.num_failed}")
    print(f'Graph: {stats["nodes"]} nodes, {stats["edges"]} edges')
    print(f"Avg latency: {avg_latency:.2f}s")
    print(f"Gist file present: {gist_ok}")

    if latencies:
        print(f"Min latency: {min(latencies):.2f}s")
        print(f"Max latency: {max(latencies):.2f}s")


print("\n=== REALTALK Streaming V2 Indexing Complete ===")

total_indexed = sum(x["indexed"] for x in conversation_summaries)
total_failed = sum(x["failed"] for x in conversation_summaries)

print(f"Conversations: {len(conversation_summaries)}")
print(f"Total sessions indexed: {total_indexed}")
print(f"Total failed: {total_failed}")

if all_latencies:
    print(
        f"Overall avg latency: "
        f"{sum(all_latencies) / len(all_latencies):.2f}s"
    )
    print(f"Overall min latency: {min(all_latencies):.2f}s")
    print(f"Overall max latency: {max(all_latencies):.2f}s")

print("\n=== Per-conversation verification ===")

for s in conversation_summaries:
    print(
        f'{s["chat"]}: '
        f'indexed={s["indexed"]}, '
        f'failed={s["failed"]}, '
        f'gists={s["gist_file_ok"]}, '
        f'nodes={s["nodes"]}, '
        f'edges={s["edges"]}'
    )
