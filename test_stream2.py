import sys, os
sys.path.insert(0, 'src')
from remem.utils.config_utils import BaseConfig
from remem.remem import ReMem
from streaming_indexer import StreamingIndexer
import shutil
config = BaseConfig(
    llm_name='google/gemini-2.5-flash-lite',
    llm_base_url='https://openrouter.ai/api/v1',
    embedding_model_name='sentence-transformers/all-mpnet-base-v2',
    extract_method='episodic_gist',
    force_index_from_scratch=True,
    dataset='locomo_temporal',
)
if os.path.exists('outputs/test_two_msgs'):
    shutil.rmtree('outputs/test_two_msgs')
remem = ReMem(global_config=config, working_dir='outputs/test_two_msgs')
indexer = StreamingIndexer(remem_instance=remem)
indexer.start()
indexer.add_message('Alice went to Paris last Monday for a work conference.', '2023-05-07', 'speaker_a')
indexer.wait_until_idle()
print('After msg 1 - Graph:', indexer.graph_size)
indexer.add_message('Alice bought a new laptop from the Apple store on Tuesday.', '2023-05-08', 'speaker_a')
indexer.wait_until_idle()
print('After msg 2 - Graph:', indexer.graph_size)
indexer.stop()
print('Latencies:', [round(l,2) for l in indexer.latency_log])
