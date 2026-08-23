import sys, os, shutil, json, traceback
sys.path.insert(0, 'src')
from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem

config = BaseConfig(
    llm_name='google/gemini-2.5-flash-lite',
    llm_base_url='https://openrouter.ai/api/v1',
    embedding_model_name='sentence-transformers/all-mpnet-base-v2',
    extract_method='episodic_gist',
    force_index_from_scratch=True,
    dataset='locomo_temporal',
    preprocess_chunk_func='by_session',
)

if os.path.exists('outputs/test_session2'):
    shutil.rmtree('outputs/test_session2')

remem = ReMem(global_config=config, working_dir='outputs/test_session2')

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

print(f'Indexing session with {len(session_1)} messages...')
try:
    remem.index([session_1])
    print('SUCCESS')
except Exception as e:
    traceback.print_exc()
