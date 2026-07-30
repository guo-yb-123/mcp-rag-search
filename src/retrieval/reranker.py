"""
Reranker 重排序模块
使用 BGE-Reranker-v2-m3 对候选文档精排
"""
from pathlib import Path
from typing import List, Dict, Tuple


class Reranker:
    """Cross-Encoder 重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Args:
            model_name: Reranker 模型名称或本地微调后的路径
        """
        self.model_name = model_name
        self.model = None

    def load(self):
        """加载模型"""
        from sentence_transformers import CrossEncoder

        print(f"正在加载 Reranker 模型: {self.model_name}...")
        self.model = CrossEncoder(
            self.model_name,
            max_length=512,  # Reranker 输入长度限制
        )
        print("Reranker 模型加载完成")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        对候选文档重排序

        Args:
            query: 查询文本
            candidates: 候选 chunk 列表
            top_k: 返回前 K 个结果

        Returns:
            [(chunk, rerank_score), ...] 按分数降序
        """
        if self.model is None:
            self.load()

        if len(candidates) == 0:
            return []

        # 构造 (query, document) pair
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs)

        # 按分数排序
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def rerank_from_results(
        self,
        query: str,
        hybrid_results: List[Tuple[Dict, float]],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        从混合检索结果中提取候选文档并重排序

        Args:
            query: 查询文本
            hybrid_results: 混合检索结果 [(chunk, fusion_score), ...]
            top_k: 返回前 K 个结果

        Returns:
            [(chunk, rerank_score), ...]
        """
        candidates = [chunk for chunk, _ in hybrid_results]
        return self.rerank(query, candidates, top_k)


def load_finetuned_reranker(model_path: str) -> Reranker:
    """加载微调后的 Reranker"""
    return Reranker(model_name=model_path)
