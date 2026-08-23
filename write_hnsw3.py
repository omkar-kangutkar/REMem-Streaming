code = open('hnsw_synonymy.py').read()
addition = '''
    @property
    def num_gists(self) -> int:
        return self._count

    @property
    def avg_search_ms(self) -> Optional[float]:
        if not self.search_times:
            return None
        return (sum(self.search_times) / len(self.search_times)) * 1000

    @property
    def avg_insert_ms(self) -> Optional[float]:
        if not self.insert_times:
            return None
        return (sum(self.insert_times) / len(self.insert_times)) * 1000

    def complexity_report(self) -> dict:
        n = self._count
        return {
            "num_gists": n,
            "method": "HNSW" if self._use_hnsw else "brute-force",
            "approx_comparisons": int(np.log2(n + 1)) * self.top_k if n > 0 else 0,
            "avg_search_ms": self.avg_search_ms,
            "avg_insert_ms": self.avg_insert_ms,
        }
'''
with open('hnsw_synonymy.py', 'w') as f:
    f.write(code + addition)
print("Done.")
