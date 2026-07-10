"""
Метрики для оценки рекомендательных систем
==========================================

Используем метрики для top-K рекомендаций:
  - Precision@K  - доля релевантных среди K рекомендованных
  - Recall@K     - доля найденных релевантных от всех релевантных
  - NDCG@K       - нормализованный дисконтированный кумулятивный выигрыш
  - HitRate@K    - хотя бы один релевантный в топ-K (бинарно)
  - MRR          - Mean Reciprocal Rank (позиция первого релевантного)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


def precision_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10,
) -> float:
    """Precision@K = |Relevant ∩ Recommended| / K

    Args:
        recommendations: DataFrame [user_id, movie_id, rank]
        ground_truth: DataFrame [user_id, movie_id] - релевантные айтемы
        k: размер списка рекомендаций
    """
    recs_k = recommendations[recommendations["rank"] <= k]
    relevant = set(zip(ground_truth["user_id"], ground_truth["movie_id"]))
    hits = recs_k.apply(
        lambda row: (row["user_id"], row["movie_id"]) in relevant, axis=1
    ).sum()
    n_users = recs_k["user_id"].nunique()
    return hits / (n_users * k) if n_users > 0 else 0.0


def recall_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10,
) -> float:
    """Recall@K = |Relevant INTERSECT Recommended| / |Relevant|"""
    recs_k = recommendations[recommendations["rank"] <= k]

    gt_per_user = ground_truth.groupby("user_id")["movie_id"].apply(set)
    rec_per_user = recs_k.groupby("user_id")["movie_id"].apply(set)

    users = gt_per_user.index.intersection(rec_per_user.index)
    if len(users) == 0:
        return 0.0

    recalls = []
    for u in users:
        gt_set = gt_per_user[u]
        rec_set = rec_per_user[u]
        recalls.append(len(gt_set & rec_set) / len(gt_set))

    return float(np.mean(recalls))


def ndcg_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10,
) -> float:
    """NDCG@K учитывает позицию релевантного документа в списке.

    DCG@K = sum_i( rel_i / log2(i+1) )  для i = 1..K
    NDCG@K = DCG@K / IDCG@K  где IDCG — идеальный порядок
    """
    recs_k = recommendations[recommendations["rank"] <= k].copy()

    gt_per_user = ground_truth.groupby("user_id")["movie_id"].apply(set)
    rec_per_user = recs_k.groupby("user_id")[["movie_id", "rank"]].apply(
        lambda df: df.set_index("movie_id")["rank"].to_dict()
    )

    users = gt_per_user.index.intersection(rec_per_user.index)
    if len(users) == 0:
        return 0.0

    ndcgs = []
    for u in users:
        gt_set = gt_per_user[u]
        rank_dict = rec_per_user[u]

        # DCG
        dcg = 0.0
        for movie_id, rank in rank_dict.items():
            if movie_id in gt_set:
                dcg += 1.0 / np.log2(rank + 1)

        # IDCG (идеальный случай: все релевантные на первых позициях)
        n_relevant = min(len(gt_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))


def hit_rate_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10,
) -> float:
    """HitRate@K - доля пользователей, для которых хотя бы 1 рекомендация попала в цель."""
    recs_k = recommendations[recommendations["rank"] <= k]
    relevant = set(zip(ground_truth["user_id"], ground_truth["movie_id"]))

    users_with_hits = recs_k[
        recs_k.apply(lambda row: (row["user_id"], row["movie_id"]) in relevant, axis=1)
    ]["user_id"].nunique()

    n_users = recommendations["user_id"].nunique()
    return users_with_hits / n_users if n_users > 0 else 0.0


def mrr_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10,
) -> float:
    """MRR@K = Mean Reciprocal Rank - средний обратный ранг первого попадания."""
    recs_k = recommendations[recommendations["rank"] <= k].sort_values("rank")
    relevant = set(zip(ground_truth["user_id"], ground_truth["movie_id"]))

    gt_users = ground_truth["user_id"].unique()
    rrs = []

    for u in gt_users:
        user_recs = recs_k[recs_k["user_id"] == u]
        rr = 0.0
        for _, row in user_recs.iterrows():
            if (u, row["movie_id"]) in relevant:
                rr = 1.0 / row["rank"]
                break
        rrs.append(rr)

    return float(np.mean(rrs))


def evaluate_all(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k_values: List[int] = [10, 20, 50],
) -> pd.DataFrame:
    """Считает все метрики для нескольких значений K.

    Returns:
        DataFrame с метриками по строкам и K по столбцам
    """
    results = {}
    for k in k_values:
        results[f"@{k}"] = {
            "Precision": precision_at_k(recommendations, ground_truth, k),
            "Recall":    recall_at_k(recommendations, ground_truth, k),
            "NDCG":      ndcg_at_k(recommendations, ground_truth, k),
            "HitRate":   hit_rate_at_k(recommendations, ground_truth, k),
            "MRR":       mrr_at_k(recommendations, ground_truth, k),
        }

    df = pd.DataFrame(results).round(4)
    return df


def coverage(
    recommendations: pd.DataFrame,
    all_items: np.ndarray,
    k: int = 10,
) -> float:
    """Catalog coverage@K - доля уникальных айтемов в топ-K рекомендациях.

    Низкое coverage = система рекомендует одно и то же всем (popularity bias).
    """
    recs_k = recommendations[recommendations["rank"] <= k]
    unique_recommended = recs_k["movie_id"].nunique()
    return unique_recommended / len(all_items)
