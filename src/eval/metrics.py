"""
评测指标模块：MRR, Recall@K, NDCG@K
"""
from typing import List, Dict, Set, Tuple

import numpy as np


def mean_reciprocal_rank(
    results: List[List[str]],
    ground_truth: List[Set[str]],
) -> float:
    """
    MRR (Mean Reciprocal Rank)

    Args:
        results: 每个query的检索结果，[[chunk_id, ...], ...]
        ground_truth: 每个query的相关chunk_id集合，[{chunk_id, ...}, ...]

    Returns:
        MRR 值
    """
    reciprocal_ranks = []
    for res, gt in zip(results, ground_truth):
        for rank, chunk_id in enumerate(res, start=1):
            if chunk_id in gt:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def recall_at_k(
    results: List[List[str]],
    ground_truth: List[Set[str]],
    k: int = 10,
) -> float:
    """
    Recall@K

    Args:
        results: 每个query的检索结果
        ground_truth: 每个query的相关chunk_id集合
        k: 取前 K 个结果

    Returns:
        平均 Recall@K
    """
    recalls = []
    for res, gt in zip(results, ground_truth):
        if not gt:
            continue
        hits = len(set(res[:k]) & gt)
        recalls.append(hits / len(gt))

    return float(np.mean(recalls)) if recalls else 0.0


def recall_at_k_values(
    results: List[List[str]],
    ground_truth: List[Set[str]],
    k_values: Tuple[int, ...] = (5, 10, 20, 50),
) -> Dict[int, float]:
    """计算多个 K 值下的 Recall"""
    return {k: recall_at_k(results, ground_truth, k) for k in k_values}


def ndcg_at_k(
    results: List[List[str]],
    ground_truth: List[Dict[str, int]],
    k: int = 10,
) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain)

    Args:
        results: 每个query的检索结果
        ground_truth: 每个query的 {chunk_id: relevance_score} 映射
                      relevance 取值：0(不相关), 1(部分相关), 2(完全相关)
        k: 取前 K 个结果

    Returns:
        平均 NDCG@K
    """
    ndcg_scores = []
    for res, gt in zip(results, ground_truth):
        if not gt:
            continue

        # DCG
        dcg = 0.0
        for i, chunk_id in enumerate(res[:k], start=1):
            rel = gt.get(chunk_id, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 1)

        # IDCG (ideal DCG)
        ideal_rels = sorted(gt.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, start=1):
            idcg += (2 ** rel - 1) / np.log2(i + 1)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def evaluate_all(
    results: List[List[str]],
    ground_truth_binary: List[Set[str]],
    ground_truth_graded: List[Dict[str, int]],
) -> Dict:
    """
    一次计算所有指标

    Returns:
        {"mrr@5": ..., "mrr@10": ..., "recall@5": ..., "ndcg@5": ..., "ndcg@10": ...}
    """
    metrics = {}

    # MRR
    for k in [5, 10]:
        results_k = [r[:k] for r in results]
        metrics[f"mrr@{k}"] = round(mean_reciprocal_rank(results_k, ground_truth_binary), 4)

    # Recall
    for k in [5, 10, 20, 50]:
        metrics[f"recall@{k}"] = round(recall_at_k(results, ground_truth_binary, k), 4)

    # NDCG
    for k in [5, 10]:
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(results, ground_truth_graded, k), 4)

    return metrics
