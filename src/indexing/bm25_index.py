"""
BM25 稀疏检索索引模块
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import jieba
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 关键词检索引擎"""

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunks: List[Dict] = []
        self.tokenized_chunks: List[List[str]] = []

    def build(self, chunks: List[Dict]):
        """
        构建 BM25 索引

        Args:
            chunks: chunk 列表，每个 chunk 含 content 字段
        """
        self.chunks = chunks
        self.tokenized_chunks = [
            self._tokenize(c["content"]) for c in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print(f"BM25 索引构建完成，共 {len(chunks)} 个文档")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict, float]]:
        """
        检索

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            [(chunk_dict, bm25_score), ...] 按分数降序
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 build()")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 取 top_k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:top_k]

        results = []
        for idx, score in top_indices:
            if score > 0:  # 过滤得分为 0 的结果
                results.append((self.chunks[idx], float(score)))

        return results

    def _tokenize(self, text: str) -> List[str]:
        """中文分词 + 去停用词"""
        tokens = jieba.lcut(text)
        # 过滤纯标点和空白
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens

    def save(self, output_dir: str):
        """保存索引到磁盘"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 chunks 和 tokenized_chunks
        with open(output_dir / "bm25_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        with open(output_dir / "bm25_tokenized.pkl", "wb") as f:
            pickle.dump(self.tokenized_chunks, f)

        print(f"BM25 索引已保存到 {output_dir}")

    def load(self, index_dir: str):
        """从磁盘加载索引"""
        index_dir = Path(index_dir)

        with open(index_dir / "bm25_chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        with open(index_dir / "bm25_tokenized.pkl", "rb") as f:
            self.tokenized_chunks = pickle.load(f)

        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print(f"BM25 索引已加载，共 {len(self.chunks)} 个文档")


if __name__ == "__main__":
    # 测试
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    with open(f"{data_dir}/chunks/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = BM25Index()
    index.build(chunks)

    # 测试搜索
    results = index.search("云服务器怎么扩容", top_k=5)
    for chunk, score in results:
        print(f"[{score:.4f}] {chunk['doc_title']} - {chunk['section']}")
        print(f"  {chunk['content'][:100]}...")
        print()
