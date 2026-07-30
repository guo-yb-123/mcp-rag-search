"""
Milvus 向量索引模块
使用 BGE-large-zh-v1.5 生成 Embedding，Milvus Lite 存储和检索
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm


class MilvusIndex:
    """向量检索引擎（封装 Milvus + BGE Embedding）"""

    def __init__(self, embedding_model_name: str = "BAAI/bge-small-zh-v1.5", collection_name: str = "doc_search"):
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.embedding_model = None
        self.collection = None
        self.chunks: List[Dict] = []
        self.dimension = 512  # bge-small-zh 的输出维度 (large=1024)

    def build(self, chunks: List[Dict], batch_size: int = 32):
        """
        构建向量索引

        Args:
            chunks: chunk 列表
            batch_size: embedding 批处理大小
        """
        self.chunks = chunks

        # 延迟加载，避免没装依赖时 import 报错
        from pymilvus import MilvusClient

        # 加载 embedding 模型（放到这里避免启动时就加载）
        if self.embedding_model is None:
            self._load_embedding_model()

        # 创建 Milvus Lite 客户端（本地文件存储）
        self.client = MilvusClient("./data/milvus.db")

        # 创建 collection
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            metric_type="IP",  # 内积相似度（BGE 用归一化向量，IP 等价于余弦相似度）
        )

        # 批量生成 embedding 并插入
        print(f"正在为 {len(chunks)} 个 chunk 生成 Embedding...")
        texts = [c["content"] for c in chunks]

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode(batch_texts)
            all_embeddings.append(batch_embeddings)

        embeddings = np.concatenate(all_embeddings, axis=0)

        # 批量插入 Milvus
        data = []
        for i, chunk in enumerate(chunks):
            data.append({
                "id": i,
                "vector": embeddings[i].tolist(),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "doc_title": chunk.get("doc_title", ""),
                "section": chunk.get("section", ""),
            })

        self.client.insert(collection_name=self.collection_name, data=data)
        print(f"Milvus 索引构建完成，共 {len(chunks)} 条向量，维度 {self.dimension}")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict, float]]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            [(chunk_dict, similarity_score), ...]
        """
        if self.embedding_model is None:
            self._load_embedding_model()

        query_embedding = self._encode([query])[0].tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "doc_title", "section"],
        )

        # 映射回 chunk
        # 建立 chunk_id -> chunk 的映射
        chunk_map = {c["chunk_id"]: c for c in self.chunks}

        output = []
        for hit in results[0]:
            chunk_id = hit["entity"].get("chunk_id", "")
            chunk = chunk_map.get(chunk_id)
            if chunk:
                output.append((chunk, hit["distance"]))

        return output

    def load(self, chunks_path: str = "data/index/bm25_chunks.json"):
        """从已有索引加载（不重新构建）"""
        from pymilvus import MilvusClient
        import json as _json

        # 加载 chunks 数据
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = _json.load(f)
        print(f"已加载 {len(self.chunks)} 个 chunks")

        # 连接 Milvus 并加载 collection 到内存
        self.client = MilvusClient("./data/milvus.db")
        if not self.client.has_collection(self.collection_name):
            raise RuntimeError(
                f"Milvus collection '{self.collection_name}' 不存在，请先运行 build_index()"
            )
        self.client.load_collection(self.collection_name)
        print(f"Milvus collection '{self.collection_name}' 已加载")

    def _load_embedding_model(self):
        """加载 Embedding 模型"""
        from sentence_transformers import SentenceTransformer

        print(f"正在加载 Embedding 模型: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("Embedding 模型加载完成")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """批量编码文本为向量"""
        # BGE 模型官方推荐对查询加 instruction prefix
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,  # L2 归一化，使 IP 等价于余弦相似度
            show_progress_bar=False,
        )
        return embeddings


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    with open(f"{data_dir}/chunks/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = MilvusIndex()
    index.build(chunks)

    # 测试搜索
    results = index.search("云服务器怎么扩容", top_k=5)
    for chunk, score in results:
        print(f"[{score:.4f}] {chunk['doc_title']} - {chunk['section']}")
        print(f"  {chunk['content'][:100]}...")
        print()
