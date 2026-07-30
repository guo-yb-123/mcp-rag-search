"""
混合检索融合模块 — RRF (Reciprocal Rank Fusion)
"""
from typing import List, Dict, Tuple


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Dict, float]],
    dense_results: List[Tuple[Dict, float]],
    k: int = 60,
    top_n: int = 30,
) -> List[Tuple[Dict, float]]:
    """
    RRF 混合融合

    Args:
        bm25_results: BM25 检索结果 [(chunk, score), ...]
        dense_results: Dense 向量检索结果 [(chunk, score), ...]
        k: 平滑常数（默认60，来自原始论文）
        top_n: 返回前 N 个结果

    Returns:
        [(chunk, rrf_score), ...] 按 RRF 分数降序
    """
    rrf_scores: Dict[str, Tuple[Dict, float]] = {}

    # BM25 贡献
    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        chunk_id = chunk["chunk_id"]
        rrf_score = 1.0 / (k + rank)
        if chunk_id in rrf_scores:
            rrf_scores[chunk_id] = (chunk, rrf_scores[chunk_id][1] + rrf_score)
        else:
            rrf_scores[chunk_id] = (chunk, rrf_score)

    # Dense 贡献
    for rank, (chunk, _) in enumerate(dense_results, start=1):
        chunk_id = chunk["chunk_id"]
        rrf_score = 1.0 / (k + rank)
        if chunk_id in rrf_scores:
            rrf_scores[chunk_id] = (chunk, rrf_scores[chunk_id][1] + rrf_score)
        else:
            rrf_scores[chunk_id] = (chunk, rrf_score)

    # 按 RRF 分数排序
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def weighted_sum_fusion(
    bm25_results: List[Tuple[Dict, float]],
    dense_results: List[Tuple[Dict, float]],
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
    top_n: int = 30,
) -> List[Tuple[Dict, float]]:
    """
    加权求和融合（需要调参，建议优先用 RRF）

    Args:
        bm25_weight: BM25 分数的权重
        dense_weight: Dense 分数的权重
    """
    # 需要先对两组分数做归一化（min-max）
    if bm25_results:
        bm25_scores = [s for _, s in bm25_results]
        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        bm25_range = bm25_max - bm25_min or 1.0
    else:
        bm25_min, bm25_range = 0, 1.0

    if dense_results:
        dense_scores = [s for _, s in dense_results]
        dense_min, dense_max = min(dense_scores), max(dense_scores)
        dense_range = dense_max - dense_min or 1.0
    else:
        dense_min, dense_range = 0, 1.0

    fused_scores: Dict[str, Tuple[Dict, float]] = {}

    for chunk, score in bm25_results:
        normalized = (score - bm25_min) / bm25_range
        chunk_id = chunk["chunk_id"]
        fused_scores[chunk_id] = (chunk, bm25_weight * normalized)

    for chunk, score in dense_results:
        normalized = (score - dense_min) / dense_range
        chunk_id = chunk["chunk_id"]
        if chunk_id in fused_scores:
            fused_scores[chunk_id] = (chunk, fused_scores[chunk_id][1] + dense_weight * normalized)
        else:
            fused_scores[chunk_id] = (chunk, dense_weight * normalized)

    sorted_results = sorted(fused_scores.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]
