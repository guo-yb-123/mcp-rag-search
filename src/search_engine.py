"""
搜索引擎核心 — 串联检索全链路
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.indexing.bm25_index import BM25Index
from src.indexing.milvus_index import MilvusIndex
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker


class SearchEngine:
    """AI搜索引擎：BM25 + Dense + RRF + Reranker"""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        data_dir: str = "data",
    ):
        self.data_dir = Path(data_dir)
        self.bm25 = BM25Index()
        self.milvus = MilvusIndex(embedding_model_name=embedding_model)
        self.reranker = Reranker(model_name=reranker_model)
        self._loaded = False

    def build_index(self, chunks: List[Dict] = None):
        """构建全部索引"""
        if chunks is None:
            chunks_path = self.data_dir / "chunks" / "chunks.json"
            if not chunks_path.exists():
                raise FileNotFoundError(
                    f"未找到 chunks 文件: {chunks_path}\n"
                    f"请先运行 crawler 和 chunker 生成数据。"
                )
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

        print(f"开始构建索引，共 {len(chunks)} 个 chunk...\n")

        # BM25
        print("[1/2] 构建 BM25 索引...")
        self.bm25.build(chunks)

        # Milvus
        print("[2/2] 构建 Milvus 向量索引...")
        self.milvus.build(chunks)

        # 保存
        self.bm25.save(str(self.data_dir / "index"))
        print("\n所有索引构建完成！")

    def load_index(self):
        """从磁盘加载索引"""
        index_dir = self.data_dir / "index"

        if not (index_dir / "bm25_chunks.json").exists():
            raise FileNotFoundError(
                f"索引文件不存在: {index_dir}\n请先运行 build_index() 构建索引。"
            )

        print("加载 BM25 索引...")
        self.bm25.load(str(index_dir))
        print("加载 Milvus 索引...")
        self.milvus.load(str(index_dir / "bm25_chunks.json"))
        print("加载 Reranker 模型...")
        self.reranker.load()
        self._loaded = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        执行搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数
            use_reranker: 是否使用 Reranker 精排
            verbose: 是否输出详细日志

        Returns:
            [{"chunk_id": ..., "doc_title": ..., "content": ..., "score": ..., "rank": ...}, ...]
        """
        if verbose:
            print(f"\n查询: {query}")
            print("-" * 50)

        # Step 1: 双路召回
        bm25_results = self.bm25.search(query, top_k=20)
        dense_results = self.milvus.search(query, top_k=20)

        if verbose:
            print(f"BM25 召回: {len(bm25_results)} 条")
            print(f"Dense 召回: {len(dense_results)} 条")

        # Step 2: RRF 融合
        fused = reciprocal_rank_fusion(bm25_results, dense_results, top_n=10)
        if verbose:
            print(f"RRF 融合: {len(fused)} 条候选")

        # Step 3: Reranker 精排
        if use_reranker:
            final = self.reranker.rerank_from_results(query, fused, top_k=top_k)
        else:
            final = fused[:top_k]

        # 格式化输出
        output = []
        for rank, (chunk, score) in enumerate(final, start=1):
            output.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "doc_title": chunk.get("doc_title", ""),
                "doc_url": chunk.get("doc_url", ""),
                "category": chunk.get("category", ""),
                "section": chunk.get("section", ""),
                "content": chunk["content"],
                "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "score": round(float(score), 4),
            })

        if verbose:
            print(f"\n最终结果: Top-{top_k}")
            for item in output:
                print(f"  [{item['rank']}] score={item['score']:.4f} | {item['doc_title']} > {item['section']}")
            print()

        return output
