import argparse
import json
import logging
import os
import time
from typing import List

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_locomo_temporal(path: str) -> List[dict]:
    with open(path) as f:
        raw = json.load(f)
    conversations = []
    for conv in raw:
        messages = []
        conversation = conv.get("conversation", conv)
        for key in sorted(conversation.keys()):
            if key.startswith("session_") and not key.endswith("date_time"):
                date = conversation.get(f"{key}_date_time", "")
                for dialog in conversation[key]:
                    messages.append({
                        "text": dialog["text"],
                        "timestamp": date if date else "2023-01-01",
                        "speaker": dialog["speaker"],
                    })
        conversations.append({
            "id": conv.get("sample_id", len(conversations)),
            "messages": messages,
            "qa_pairs": conv.get("qa", []),
        })
    logger.info("Loaded %d conversations from %s", len(conversations), path)
    return conversations


def load_realtalk(dataset_dir: str) -> List[dict]:
    import glob
    files = sorted(glob.glob(os.path.join(dataset_dir, "Chat_*.json")))
    conversations = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        messages = []
        for session in data.get("sessions", []):
            date = session.get("date", "")
            for turn in session.get("turns", []):
                messages.append({
                    "text": turn.get("text", ""),
                    "timestamp": date if date else "2023-01-01",
                    "speaker": turn.get("speaker", "user"),
                })
        conversations.append({
            "id": os.path.basename(fpath),
            "messages": messages,
            "qa_pairs": data.get("qa_pairs", []),
        })
    logger.info("Loaded %d REALTALK conversations from %s",
                len(conversations), dataset_dir)
    return conversations


def run_streaming_eval(conversations, remem_instance, indexer, output_dir):
    all_latencies = []
    graph_growth = []
    total_messages = 0

    for ci, conv in enumerate(tqdm(conversations, desc="Conversations")):
        logger.info("Conversation %d/%d | messages=%d | questions=%d",
                    ci + 1, len(conversations),
                    len(conv["messages"]), len(conv["qa_pairs"]))

        for msg in conv["messages"]:
            t_start = time.time()
            indexer.add_message(
                text=msg["text"],
                timestamp=msg["timestamp"],
                speaker=msg.get("speaker", "user"),
            )
            indexer.wait_until_idle()
            latency = time.time() - t_start
            all_latencies.append(latency)
            total_messages += 1
            gs = indexer.graph_size
            graph_growth.append({
                "message_num": total_messages,
                "nodes": gs["nodes"],
                "edges": gs["edges"],
                "latency_s": round(latency, 3),
            })

        logger.info("Conv %d indexed | avg_latency=%.2fs | graph: %d nodes, %d edges",
                    ci + 1,
                    sum(all_latencies[-len(conv["messages"]):]) / len(conv["messages"]),
                    indexer.graph_size["nodes"],
                    indexer.graph_size["edges"])

        if conv["qa_pairs"]:
            questions = [qa["question"] for qa in conv["qa_pairs"]]
            try:
                results = remem_instance.rag_for_qa(questions)
                solutions, responses, meta = results[0], results[1], results[2]
                logger.info("Answered %d questions for conversation %d",
                            len(questions), ci + 1)
            except Exception as e:
                logger.error("QA failed for conversation %d: %s", ci + 1, e)

    os.makedirs(output_dir, exist_ok=True)
    growth_path = os.path.join(output_dir, "graph_growth.json")
    with open(growth_path, "w") as f:
        json.dump(graph_growth, f, indent=2)

    logger.info("==== Streaming Evaluation Summary ====")
    logger.info("Total messages:  %d", total_messages)
    logger.info("Avg latency/msg: %.2fs", sum(all_latencies) / len(all_latencies))
    logger.info("Min latency:     %.2fs", min(all_latencies))
    logger.info("Max latency:     %.2fs", max(all_latencies))
    logger.info("Graph growth saved to %s", growth_path)

    return {
        "total_messages": total_messages,
        "avg_latency_s": sum(all_latencies) / len(all_latencies),
        "min_latency_s": min(all_latencies),
        "max_latency_s": max(all_latencies),
        "graph_growth": graph_growth,
    }


def run_ablation(embedding_dim: int = 768, threshold: float = 0.8) -> List[dict]:
    from hnsw_synonymy import HNSWSynonymyUpdater, BruteForceBaseline

    test_sizes = [100, 250, 500, 1000, 2000, 5000]
    results = []

    logger.info("==== Ablation Study: HNSW vs Brute-Force ====")
    logger.info("%-8s %-15s %-20s %-10s", "N", "HNSW (ms)", "Brute force (ms)", "Speedup")
    logger.info("-" * 55)

    for n in test_sizes:
        embs = np.random.randn(n + 1, embedding_dim).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        ids = [f"gist_{i}" for i in range(n + 1)]

        hnsw = HNSWSynonymyUpdater(embedding_dim=embedding_dim, threshold=threshold)
        for i in range(n):
            hnsw.update(ids[i], embs[i])
        t0 = time.perf_counter()
        hnsw.update(ids[n], embs[n])
        hnsw_ms = (time.perf_counter() - t0) * 1000

        bf = BruteForceBaseline(threshold=threshold)
        for i in range(n):
            bf.update(ids[i], embs[i])
        t0 = time.perf_counter()
        bf.update(ids[n], embs[n])
        bf_ms = (time.perf_counter() - t0) * 1000

        speedup = bf_ms / hnsw_ms if hnsw_ms > 0 else float("inf")

        logger.info("%-8d %-15.4f %-20.4f %.1fx", n, hnsw_ms, bf_ms, speedup)

        results.append({
            "n_gists": n,
            "hnsw_ms": round(hnsw_ms, 4),
            "bruteforce_ms": round(bf_ms, 4),
            "speedup_x": round(speedup, 1),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="REMem-Stream Evaluation")
    parser.add_argument("--dataset", default="locomo_temporal",
                        choices=["locomo_temporal", "realtalk"])
    parser.add_argument("--llm_name", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--llm_base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--embedding_name",
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--mode", default="streaming",
                        choices=["streaming", "ablation"])
    parser.add_argument("--output_dir", default="outputs/streaming_eval")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "ablation":
        results = run_ablation()
        out_path = os.path.join(args.output_dir, "ablation_results.json")
        with open(out_path, "w") as f:
            json.dump({"ablation": results}, f, indent=2)
        logger.info("Ablation results saved to %s", out_path)
        return

    from remem.remem import ReMem
    from remem.utils.config_utils import BaseConfig
    from streaming_indexer import StreamingIndexer

    config = BaseConfig(
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        extract_method="episodic_gist",
        force_index_from_scratch=True,
        dataset=args.dataset,
        qa_top_k=3,
    )

    if args.dataset == "locomo_temporal":
        path = os.path.join(args.dataset_dir, "locomo", "locomo_temporal.json")
        conversations = load_locomo_temporal(path)
    else:
        path = os.path.join(args.dataset_dir, "realtalk")
        conversations = load_realtalk(path)

    for ci, conv in enumerate(conversations):
        working_dir = os.path.join(
            args.output_dir,
            f"conv_{ci}_{args.llm_name.replace('/', '_')}"
        )
        remem = ReMem(global_config=config, working_dir=working_dir)
        indexer = StreamingIndexer(remem_instance=remem, log_latency=True)
        indexer.start()
        run_streaming_eval([conv], remem, indexer,
                           os.path.join(args.output_dir, f"conv_{ci}"))
        indexer.stop()

    logger.info("All conversations complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
