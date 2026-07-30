"""
消融实验运行脚本
对比：BM25 only / Dense only / Hybrid (RRF) / Hybrid + Reranker
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Set

from src.eval.metrics import evaluate_all


class ExperimentRunner:
    """消融实验运行器"""

    def __init__(
        self,
        bm25_index,
        milvus_index,
        reranker,
        test_queries_path: str = "data/eval/test_queries.json",
    ):
        self.bm25 = bm25_index
        self.milvus = milvus_index
        self.reranker = reranker
        self.test_queries = self._load_queries(test_queries_path)

    def _load_queries(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 支持 {"queries": [...]} 和直接 [...] 两种格式
        if isinstance(data, dict) and "queries" in data:
            return data["queries"]
        return data

    def run_all(self) -> Dict:
        """运行全部消融实验"""
        print(f"\n{'='*60}")
        print(f"消融实验：共 {len(self.test_queries)} 个测试查询")
        print(f"{'='*60}\n")

        results = {}

        # ---- 实验 A: BM25 only ----
        print("[实验 A] BM25 only...")
        results["A_BM25"] = self._run_experiment(
            name="A: BM25 only",
            retrieval_fn=lambda q: self._bm25_only(q),
        )

        # ---- 实验 B: Dense only ----
        print("[实验 B] Dense only...")
        results["B_Dense"] = self._run_experiment(
            name="B: Dense only",
            retrieval_fn=lambda q: self._dense_only(q),
        )

        # ---- 实验 C: Hybrid (RRF) ----
        print("[实验 C] Hybrid (BM25 + Dense + RRF)...")
        results["C_Hybrid"] = self._run_experiment(
            name="C: Hybrid (RRF)",
            retrieval_fn=lambda q: self._hybrid(q),
        )

        # ---- 实验 D: Hybrid + Reranker ----
        print("[实验 D] Hybrid + Reranker (微调)...")
        results["D_Hybrid_Reranker"] = self._run_experiment(
            name="D: Hybrid + Reranker",
            retrieval_fn=lambda q: self._hybrid_reranker(q),
        )

        return results

    def _run_experiment(self, name: str, retrieval_fn) -> Dict:
        """运行单组实验"""
        all_results = []  # [[chunk_id, ...], ...]
        latencies = []

        for query_item in self.test_queries:
            query = query_item["query"]

            t0 = time.time()
            chunk_ids = retrieval_fn(query)
            latency = (time.time() - t0) * 1000  # ms
            latencies.append(latency)

            all_results.append(chunk_ids)

            print(f"  Q: {query[:50]}... → {len(chunk_ids)} results, {latency:.0f}ms")

        # 提取 ground truth
        gt_binary = []
        gt_graded = []
        for qi in self.test_queries:
            gt_binary.append(set(qi.get("relevant_chunks", [])))
            gt_graded.append({
                rc["chunk_id"]: rc.get("relevance", 1)
                for rc in qi.get("relevant_chunks_with_scores", [])
            })

        metrics = evaluate_all(all_results, gt_binary, gt_graded)
        metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

        print(f"  → MRR@10={metrics.get('mrr@10', 'N/A')}, "
              f"Recall@50={metrics.get('recall@50', 'N/A')}, "
              f"延迟={metrics['avg_latency_ms']}ms\n")

        return metrics

    # ---- 各检索策略 ----

    def _bm25_only(self, query: str, top_k: int = 50) -> List[str]:
        results = self.bm25.search(query, top_k=top_k)
        return [c["chunk_id"] for c, _ in results]

    def _dense_only(self, query: str, top_k: int = 50) -> List[str]:
        results = self.milvus.search(query, top_k=top_k)
        return [c["chunk_id"] for c, _ in results]

    def _hybrid(self, query: str, top_k: int = 50) -> List[str]:
        from src.retrieval.hybrid_fusion import reciprocal_rank_fusion

        bm25_res = self.bm25.search(query, top_k=20)
        dense_res = self.milvus.search(query, top_k=20)
        fused = reciprocal_rank_fusion(bm25_res, dense_res, top_n=top_k)
        return [c["chunk_id"] for c, _ in fused]

    def _hybrid_reranker(self, query: str, top_k: int = 50) -> List[str]:
        from src.retrieval.hybrid_fusion import reciprocal_rank_fusion

        bm25_res = self.bm25.search(query, top_k=20)
        dense_res = self.milvus.search(query, top_k=20)
        fused = reciprocal_rank_fusion(bm25_res, dense_res, top_n=top_k)

        # Reranker 精排
        reranked = self.reranker.rerank_from_results(query, fused, top_k=top_k)
        return [c["chunk_id"] for c, _ in reranked]

    def print_summary(self, all_results: Dict):
        """打印汇总对比表"""
        print(f"\n{'='*80}")
        print("消融实验结果汇总")
        print(f"{'='*80}")
        header = f"{'实验':<30} {'MRR@10':<10} {'Recall@10':<12} {'Recall@50':<12} {'NDCG@10':<10} {'延迟(ms)':<10}"
        print(header)
        print("-" * 80)

        for name, metrics in all_results.items():
            print(f"{name:<30} "
                  f"{metrics.get('mrr@10', 'N/A'):<10} "
                  f"{metrics.get('recall@10', 'N/A'):<12} "
                  f"{metrics.get('recall@50', 'N/A'):<12} "
                  f"{metrics.get('ndcg@10', 'N/A'):<10} "
                  f"{metrics.get('avg_latency_ms', 'N/A'):<10}")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    print("请通过 main.py 或 search_engine.py 调用实验，确保索引已构建。")
