code = open('hnsw_synonymy.py').read()
addition = '''
    def update(self, new_gist_id: str, new_embedding: np.ndarray) -> List[Tuple[str, str]]:
        """Find synonymy pairs for a new gist and update the index."""
        emb = self._normalise(new_embedding)
        pairs = []
        if self._count > 0:
            t0 = time.perf_counter()
            similar_ids = self._search(emb)
            self.search_times.append(time.perf_counter() - t0)
            for existing_id in similar_ids:
                if existing_id != new_gist_id:
                    pairs.append((new_gist_id, existing_id))
        else:
            self.search_times.append(0.0)
        t0 = time.perf_counter()
        self._insert(new_gist_id, emb)
        self.insert_times.append(time.perf_counter() - t0)
        return pairs
'''
with open('hnsw_synonymy.py', 'w') as f:
    f.write(code + addition)
print("Done.")
