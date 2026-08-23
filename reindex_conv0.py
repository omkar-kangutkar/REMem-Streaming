import sys, os, shutil, json
sys.path.insert(0, 'src')
sys.path.insert(0, 'examples')
from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem
from streaming_indexer import StreamingIndexer
from locomo import get_sessions

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

data = json.load(open('reproduce/dataset/locomo/locomo_temporal.json'))
conversation = data[0]['conversation']
sessions = get_sessions(conversation)
print(f'Conv 0: {len(sessions)} sessions to index')

working_dir = 'outputs/streaming_eval/conv_0_google_gemini-2.5-flash-lite'
remem = ReMem(global_config=config, working_dir=working_dir)
indexer = StreamingIndexer(remem_instance=remem)
indexer.start()

for si, session in enumerate(sessions):
    indexer.add_session(session)
    indexer.wait_until_idle()
    print(f'Session {si+1}/{len(sessions)} | graph: {indexer.graph_size}')

indexer.stop()
print('Done. Final graph:', indexer.graph_size)
