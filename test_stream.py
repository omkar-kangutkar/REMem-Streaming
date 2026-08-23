import sys, os
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
)

remem = ReMem(global_config=config, working_dir='outputs/test_stream2')
indexer = StreamingIndexer(remem_instance=remem)
indexer.start()
indexer.add_message('Alice went to the park.', '2023-05-07', 'speaker_a')
indexer.wait_until_idle()
print('SUCCESS - Indexed:', indexer.num_indexed, 'Failed:', indexer.num_failed)
print('Graph:', indexer.graph_size)
indexer.stop()
