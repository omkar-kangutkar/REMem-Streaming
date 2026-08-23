import json
import logging
import queue
import threading
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class StreamingIndexer:
    """Incremental streaming indexer for ReMem."""

    def __init__(self, remem_instance, log_latency: bool = True):
        self.remem = remem_instance
        self.log_latency = log_latency
        self._queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="StreamingIndexer-Worker",
        )
        self.num_indexed = 0
        self.num_failed = 0
        self.latency_log: List[float] = []
        self._running = False

    def start(self) -> None:
        import os
        # Clear the shared openie results file so each streaming run starts fresh
        openie_path = self.remem.openie_results_path
        if os.path.exists(openie_path):
            os.remove(openie_path)
            logger.info("Cleared openie cache: %s", openie_path)
        self._running = True
        self._worker.start()
        logger.info("StreamingIndexer started.")

    def add_message(self, text: str, timestamp: str = None, speaker: str = "user") -> None:
        if not self._running:
            raise RuntimeError("Call .start() before .add_message()")
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._queue.put({
            "text": text,
            "timestamp": timestamp,
            "speaker": speaker,
        })

    def wait_until_idle(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self._queue.join()
        self._running = False
        logger.info("StreamingIndexer stopped. Indexed: %d, Failed: %d",
                    self.num_indexed, self.num_failed)


    @property
    def graph_size(self) -> dict:
        try:
            g = self.remem.graph
            return {'nodes': g.vcount(), 'edges': g.ecount()}
        except Exception:
            return {'nodes': 0, 'edges': 0}

    def add_session(self, session: list) -> None:
        if not self._running:
            raise RuntimeError('Call .start() before .add_session()')
        self._queue.put({'__session__': session})
    def _worker_loop(self) -> None:
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                t_start = time.time()
                self._index_one_message(msg)
                elapsed = time.time() - t_start
                self.num_indexed += 1
                if self.log_latency:
                    self.latency_log.append(elapsed)
                logger.info(
                    "Indexed in %.2fs [total: %d]",
                    elapsed, self.num_indexed,
                )
            except Exception as e:
                self.num_failed += 1
                logger.error("Failed: %s | %s", str(msg)[:50], e)
            finally:
                self._queue.task_done()

    def _index_one_message(self, msg: dict) -> None:
        import os
        openie_path = self.remem.openie_results_path
        if os.path.exists(openie_path):
            os.remove(openie_path)
        self.remem.global_config.preprocess_chunk_func = "by_session"
        # Only force from scratch on first session -- after that accumulate embeddings
        if self.num_indexed > 0:
            self.remem.global_config.force_index_from_scratch = False
        if "__session__" in msg:
            session = msg["__session__"]
        else:
            session = [{
                "role": msg["speaker"],
                "content": msg["text"],
                "date": msg["timestamp"],
                "dialog_id": "D1:1",
                "session_idx": 1,
                "message_idx": 1,
            }]
        self.remem.index([session])
