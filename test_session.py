import sys, os, shutil, json
sys.path.insert(0, 'src')
from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem
from streaming_indexer import StreamingIndexer

config = BaseConfig(
    llm_name='google/gemini-2.5-flash-lite',
    llm_base_url='https://openrouter.ai/api/v1',
    embedding_model_name='sentence-transformers/all-mpnet-base-v2',
    extract_method='episodic_gist',
    force_index_from_scratch=True,
    dataset='locomo_temporal',
    preprocess_chunk_func='by_session',
)

if os.path.exists('outputs/test_session'):
    shutil.rmtree('outputs/test_session')

remem = ReMem(global_config=config, working_dir='outputs/test_session')
indexer = StreamingIndexer(remem_instance=remem)
indexer.start()

data = json.load(open('reproduce/dataset/locomo/locomo_temporal.json'))
conv = data[0]['conversation']
date = conv.get('session_1_date_time', '2023-05-07')
session_1 = []
for dialog in conv['session_1']:
    session_1.append({
        'role': dialog['speaker'],
        'content': dialog['text'],
        'date': date,
        'dialog_id': dialog['dia_id'],
        'session_idx': 1,
        'message_idx': int(dialog['dia_id'].split(':')[1]),
    })

print(f'Session 1 has {len(session_1)} messages')
indexer.add_session(session_1)
indexer.wait_until_idle()
print('After session 1 - Graph:', indexer.graph_size)
print('Indexed:', indexer.num_indexed, 'Failed:', indexer.num_failed)
if indexer.latency_log:
    print('Latency:', round(indexer.latency_log[0], 2), 'seconds')
indexer.stop()
