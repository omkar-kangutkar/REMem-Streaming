import logging, os, pickle, sys
import numpy as np
sys.path.insert(0, 'src')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
import faiss

def run_global_synonymy_pass(working_dir, threshold=0.8, top_k=10):
    graph_path = os.path.join(working_dir, 'graph.pkl')
    if not os.path.exists(graph_path):
        logger.warning("No graph at %s", working_dir)
        return 0

    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    logger.info("Graph: %d nodes, %d edges", graph.vcount(), graph.ecount())

    gists_path = os.path.join(working_dir, 'gists_embeddings', 'vdb_gists.pkl')
    if not os.path.exists(gists_path):
        logger.warning("No gist embeddings")
        return 0

    with open(gists_path, 'rb') as f:
        store = pickle.load(f)

    ids = store['hash_ids']
    embeddings = np.array(store['embeddings'], dtype=np.float32)
    n = len(ids)
    logger.info("Found %d gist embeddings", n)

    if n < 2:
        return 0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.hnsw.efSearch = 64
    index.add(embeddings)

    existing = set()
    for e in graph.es:
        try:
            if graph.es[e.index]['type'] == 'synonymy':
                existing.add((graph.vs[e.source]['name'], graph.vs[e.target]['name']))
        except Exception:
            pass

    logger.info("Existing synonymy edges: %d", len(existing))

    k = min(top_k + 1, n)
    sims_all, idxs_all = index.search(embeddings, k)

    new_pairs = []
    for i, (sims, idxs) in enumerate(zip(sims_all, idxs_all)):
        for sim, j in zip(sims, idxs):
            if i == int(j):
                continue
            if float(sim) < threshold:
                continue
            gi, gj = ids[i], ids[int(j)]
            if (gi, gj) not in existing:
                new_pairs.append((gi, gj))
                existing.add((gi, gj))
                existing.add((gj, gi))

    logger.info("New synonymy pairs: %d", len(new_pairs))
    if not new_pairs:
        return 0

    name_to_idx = {v['name']: v.index for v in graph.vs if v['name']}
    edges_to_add = []
    for gi, gj in new_pairs:
        if gi in name_to_idx and gj in name_to_idx:
            edges_to_add.append((name_to_idx[gi], name_to_idx[gj]))

    if edges_to_add:
        graph.add_edges(
            edges_to_add,
            attributes={
                'type': ['synonymy'] * len(edges_to_add),
                'weight': [1.0] * len(edges_to_add)
            }
        )
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
        logger.info("Saved graph: %d nodes, %d edges",
                    graph.vcount(), graph.ecount())

    return len(edges_to_add)


if __name__ == "__main__":
    output_dir = "outputs/streaming_eval"
    total = 0
    conv_dirs = sorted([
        d for d in os.listdir(output_dir)
        if d.startswith("conv_") and "gemini" in d
    ])
    logger.info("Found %d conversations", len(conv_dirs))
    for cd in conv_dirs:
        wd = os.path.join(output_dir, cd)
        logger.info("\n=== %s ===", cd)
        try:
            new = run_global_synonymy_pass(wd)
            total += new
        except Exception as e:
            logger.error("Failed: %s", e)
    logger.info("\nTotal new synonymy edges added: %d", total)
