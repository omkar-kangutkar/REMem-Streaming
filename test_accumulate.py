import sys, os, shutil, json
sys.path.insert(0, 'src')
sys.path.insert(0, 'examples')
from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem
from streaming_indexer import StreamingIndexer
from locomo import get_sessions
import pickle, numpy as np

config = BaseConfig(
    llm_name='google/gemini-2.5-flash-lite',
    llm_base_url='https://openrouter.ai/api/v1',
    embedding_model_name='sentence-transformers/all-mpnet-base-v2',
    extract_method='episodic_gist',
    force_index_from_scratch=True,
    dataset='locomo_temporal_0',
    preprocess_chunk_func='by_session',
    qa_top_k=3,
)

if os.path.exists('outputs/test_accum'):
    shutil.rmtree('outputs/test_accum')

data = json.load(open('reproduce/dataset/locomo/locomo_temporal.json'))
sessions = get_sessions(data[0]['conversation'])
print(f'Total sessions: {len(sessions)}')

remem = ReMem(global_config=config, working_dir='outputs/test_accum')
indexer = StreamingIndexer(remem_instance=remem)
indexer.start()

for si, session in enumerate(sessions[:5]):
    indexer.add_session(session)
    indexer.wait_until_idle()
    store = pickle.load(open('outputs/test_accum/gists_embeddings/vdb_gists.pkl', 'rb'))
    print(f'After session {si+1}: {len(store["hash_ids"])} gists in store | graph: {indexer.graph_size}')

indexer.stop()
