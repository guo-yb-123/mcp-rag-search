"""
FastAPI 搜索接口
"""
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.search_engine import SearchEngine

# ---- 初始化 ----
app = FastAPI(
    title="AI 文档搜索引擎",
    description="基于 BGE + BM25 + RRF + Reranker 的企业知识库语义搜索",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局搜索引擎实例（启动时加载）
engine: SearchEngine = None


# ---- 数据模型 ----
class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    doc_title: str
    doc_url: str = ""
    category: str = ""
    section: str = ""
    content_preview: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total_results: int
    use_reranker: bool
    latency_ms: float
    results: List[SearchResult]


class IndexInfo(BaseModel):
    status: str
    num_chunks: int = 0
    embedding_model: str = ""
    reranker_model: str = ""


# ---- 生命周期 ----
@app.on_event("startup")
async def startup():
    global engine
    import time
    print("正在启动搜索引擎...")
    t0 = time.time()
    engine = SearchEngine(data_dir="data")
    engine.load_index()
    print(f"搜索引擎启动完成 ({time.time()-t0:.1f}s)")


# ---- API 端点 ----
@app.get("/", response_model=dict)
async def root():
    return {
        "name": "AI 文档搜索引擎",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search?q=查询&top_k=5&reranker=true",
            "info": "/info",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/info", response_model=IndexInfo)
async def info():
    if engine is None:
        return IndexInfo(status="not_loaded")

    return IndexInfo(
        status="ready",
        num_chunks=len(engine.bm25.chunks) if engine.bm25.chunks else 0,
        embedding_model=engine.milvus.embedding_model_name,
        reranker_model=engine.reranker.model_name,
    )


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="搜索查询"),
    top_k: int = Query(5, ge=1, le=50, description="返回结果数"),
    reranker: bool = Query(True, description="是否启用 Reranker 精排"),
):
    """
    主搜索接口
    """
    import time

    if engine is None:
        return SearchResponse(
            query=q,
            total_results=0,
            use_reranker=reranker,
            latency_ms=0,
            results=[],
        )

    t0 = time.time()
    results = engine.search(query=q, top_k=top_k, use_reranker=reranker)
    latency = (time.time() - t0) * 1000

    search_results = [
        SearchResult(**r) for r in results
    ]

    return SearchResponse(
        query=q,
        total_results=len(search_results),
        use_reranker=reranker,
        latency_ms=round(latency, 1),
        results=search_results,
    )


# ---- 启动 ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
