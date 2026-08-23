import sys, os, shutil, json, glob
sys.path.insert(0, 'src')
sys.path.insert(0, 'examples')
from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem
from streaming_indexer import StreamingIndexer
from realtalk import get_sessions

dataset_dir = 'reproduce/dataset/realtalk'
output_base = 'outputs/streaming_eval_realtalk'
os.makedirs(output_base, exist_ok=True)

chat_files = sorted(glob.glob(os.path.join(dataset_dir, 'Chat_*.json')))
print(f'Found {len(chat_files)} REALTALK conversations')

all_latencies = []

for ci, chat_file in enumerate(chat_files):
    chat_name = os.path.splitext(os.path.basename(chat_file))[0]
    print(f'\n=== [{ci+1}/{len(chat_files)}] {chat_name} ===')

    with open(chat_file) as f:
        data = json.load(f)

    sessions = get_sessions(data)
    print(f'Sessions: {len(sessions)}')

    working_dir = os.path.join(output_base, f'{chat_name}_google_gemini-2.5-flash-lite')
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    config = BaseConfig(
        llm_name='google/gemini-2.5-flash-lite',
        llm_base_url='https://openrouter.ai/api/v1',
        embedding_model_name='sentence-transformers/all-mpnet-base-v2',
        extract_method='episodic_gist',
        force_index_from_scratch=True,
        dataset=f'realtalk_{ci}',
        preprocess_chunk_func='by_session',
        qa_top_k=3,
    )

    remem = ReMem(global_config=config, working_dir=working_dir)
    indexer = StreamingIndexer(remem_instance=remem)
    indexer.start()

    for si, session in enumerate(sessions):
        indexer.add_session(session)
        indexer.wait_until_idle()

    all_latencies.extend(indexer.latency_log)
    gs = indexer.graph_size
    print(f'Indexed: {indexer.num_indexed} Failed: {indexer.num_failed}')
    print(f'Graph: {gs["nodes"]} nodes, {gs["edges"]} edges')
    avg = sum(indexer.latency_log)/len(indexer.latency_log) if indexer.latency_log else 0
    print(f'Avg latency: {avg:.2f}s')
    indexer.stop()

print('\n=== REALTALK Streaming Indexing Complete ===')
print(f'Total sessions: {len(all_latencies)}')
if all_latencies:
    print(f'Avg latency: {sum(all_latencies)/len(all_latencies):.2f}s')
    print(f'Min latency: {min(all_latencies):.2f}s')
    print(f'Max latency: {max(all_latencies):.2f}s')
