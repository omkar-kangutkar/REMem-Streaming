code = open('streaming_indexer.py').read()
addition = '''
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
                    "Indexed in %.2fs [total: %d]: %s...",
                    elapsed, self.num_indexed, msg["text"][:50],
                )
            except Exception as e:
                self.num_failed += 1
                logger.error("Failed: %s | %s", msg["text"][:50], e)
            finally:
                self._queue.task_done()

    def _index_one_message(self, msg: dict) -> None:
        doc = __import__("json").dumps({
            "messages": [{
                "content": msg["text"],
                "role": msg["speaker"],
                "date": msg["timestamp"],
            }]
        })
        self.remem.index([doc])
'''
with open('streaming_indexer.py', 'w') as f:
    f.write(code + addition)
print("Done.")
