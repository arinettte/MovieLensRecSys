"""
Стадия 1: Генерация кандидатов
================================
Три подхода:
  1. PopularityRecommender   - топ-N популярных (бейзлайн)
  2. SVDRecommender          - матричная факторизация через sklearn
  3. ItemKNNRecommender      - item-based коллаборативная фильтрация

Каждый класс реализует единый интерфейс:
  .fit(train_df)
  .recommend(user_ids, top_k) -> pd.DataFrame[user_id, movie_id, score]
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    def __init__(self, top_k: int = 100):
        self.top_k = top_k
        self._is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseRecommender":
        """Обучить модель на тренировочных данных.

        Args:
            train_df: DataFrame с колонками [user_id, movie_id, rating, timestamp]
        """

    @abstractmethod
    def recommend(
        self,
        user_ids: List[int],
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Сгенерировать кандидатов для списка пользователей.

        Returns:
            DataFrame с колонками [user_id, movie_id, score, rank]
        """

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)
        logger.info(f"Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BaseRecommender":
        return joblib.load(path)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Вызовите .fit() перед .recommend()")

    def _build_result(
        self,
        user_ids: List[int],
        scores_matrix: np.ndarray,
        all_movie_ids: np.ndarray,
        top_k: int,
        seen_items: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Конвертировать матрицу скоров в DataFrame.

        Args:
            scores_matrix: shape (n_users, n_items)
            all_movie_ids: movie_id для каждой колонки матрицы
            seen_items: {user_id: set(movie_ids)} - исключить из кандидатов
        """
        records = []
        for i, uid in enumerate(user_ids):
            scores = scores_matrix[i].copy()

            # Убираем уже просмотренные фильмы
            if seen_items and uid in seen_items:
                seen_indices = np.where(np.isin(all_movie_ids, list(seen_items[uid])))[0]
                scores[seen_indices] = -np.inf

            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            for rank, idx in enumerate(top_indices, 1):
                records.append({
                    "user_id": uid,
                    "movie_id": int(all_movie_ids[idx]),
                    "score": float(scores[idx]),
                    "rank": rank,
                })

        return pd.DataFrame(records)


class PopularityRecommender(BaseRecommender):
    """Рекомендует топ-N самых популярных фильмов.

    Скор = взвешенный рейтинг (формула IMDb):
        score = (v / (v + m)) * R + (m / (v + m)) * C
        где v - кол-во голосов, m - минимум голосов, R - средний рейтинг, C - глобальный средний
    """

    def __init__(self, top_k: int = 100, min_ratings: int = 50):
        super().__init__(top_k)
        self.min_ratings = min_ratings
        self._popular_items: pd.DataFrame = None

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        stats = train_df.groupby("movie_id").agg(
            n_ratings=("rating", "count"),
            mean_rating=("rating", "mean"),
        ).reset_index()

        C = stats["mean_rating"].mean()
        m = self.min_ratings
        v = stats["n_ratings"]
        R = stats["mean_rating"]

        stats["score"] = (v / (v + m)) * R + (m / (v + m)) * C

        self._popular_items = (
            stats[stats["n_ratings"] >= self.min_ratings]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        self._seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

        self._is_fitted = True
        logger.info(
            f"PopularityRecommender: {len(self._popular_items)} фильмов "
            f"(мин. {self.min_ratings} оценок)"
        )
        return self

    def recommend(
        self,
        user_ids: List[int],
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        self._check_fitted()
        k = top_k or self.top_k
        records = []

        for uid in user_ids:
            seen = self._seen.get(uid, set())
            candidates = self._popular_items[
                ~self._popular_items["movie_id"].isin(seen)
            ].head(k)

            for rank, (_, row) in enumerate(candidates.iterrows(), 1):
                records.append({
                    "user_id": uid,
                    "movie_id": int(row["movie_id"]),
                    "score": float(row["score"]),
                    "rank": rank,
                })

        return pd.DataFrame(records)


class SVDRecommender(BaseRecommender):
    """Матричная факторизация через TruncatedSVD (sklearn).

    Алгоритм:
    1. Строим разреженную матрицу user-item (значения - рейтинги)
    2. Центрируем по пользователям (вычитаем средний рейтинг)
    3. SVD: R = U * S * V^T, берём k компонент
    4. Восстанавливаем матрицу -> скоры для ранжирования
    """

    def __init__(self, top_k: int = 100, n_components: int = 50, random_state: int = 42):
        super().__init__(top_k)
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, train_df: pd.DataFrame) -> "SVDRecommender":
        self._user_ids = train_df["user_id"].unique()
        self._item_ids = train_df["movie_id"].unique()

        self._user2idx = {u: i for i, u in enumerate(self._user_ids)}
        self._item2idx = {m: i for i, m in enumerate(self._item_ids)}

        n_users = len(self._user_ids)
        n_items = len(self._item_ids)

        # Разреженная матрица рейтингов
        row = train_df["user_id"].map(self._user2idx).values
        col = train_df["movie_id"].map(self._item2idx).values
        data = train_df["rating"].values.astype(np.float32)

        R = csr_matrix((data, (row, col)), shape=(n_users, n_items))

        # Центрируем по пользователям
        self._user_mean = np.array(R.sum(axis=1)).flatten() / np.array(
            (R > 0).sum(axis=1)
        ).flatten()

        # Вычитаем среднее
        R_dense = R.toarray()
        mask = R_dense > 0
        R_centered = R_dense.copy()
        user_means_filled = np.broadcast_to(self._user_mean[:, np.newaxis], R_dense.shape)
        R_centered[mask] -= user_means_filled[mask]
        # SVD
        self._svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self._U = self._svd.fit_transform(R_centered)   # (n_users, k)
        self._Vt = self._svd.components_                # (k, n_items)
        self._Sigma = self._svd.singular_values_        # (k,)

        # Восстановленная матрица скоров
        self._scores = self._U @ self._Vt + self._user_mean[:, np.newaxis]

        # Что пользователь уже видел
        self._seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

        self._is_fitted = True
        logger.info(
            f"SVDRecommender: {n_users} пользователей, {n_items} фильмов, "
            f"{self.n_components} компонент"
        )
        return self

    def recommend(
        self,
        user_ids: List[int],
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        self._check_fitted()
        k = top_k or self.top_k

        valid_users = [u for u in user_ids if u in self._user2idx]
        if len(valid_users) < len(user_ids):
            missing = set(user_ids) - set(valid_users)
            logger.warning(f"Cold-start пользователи (нет в train): {missing}")

        indices = [self._user2idx[u] for u in valid_users]
        scores_matrix = self._scores[indices]

        return self._build_result(
            user_ids=valid_users,
            scores_matrix=scores_matrix,
            all_movie_ids=self._item_ids,
            top_k=k,
            seen_items=self._seen,
        )

    @property
    def explained_variance_ratio(self) -> float:
        """Доля объяснённой дисперсии показывает, насколько хорошо SVD аппроксимирует матрицу."""
        return self._svd.explained_variance_ratio_.sum()


class ItemKNNRecommender(BaseRecommender):
    """Item-based коллаборативная фильтрация.

    Идея: если пользователь высоко оценил фильм A, рекомендуем похожие на A фильмы.
    Похожесть фильмов = косинусное сходство их векторов рейтингов.

    Используем IDF-взвешивание.
    """

    def __init__(self, top_k: int = 100, n_neighbors: int = 20):
        super().__init__(top_k)
        self.n_neighbors = n_neighbors

    def fit(self, train_df: pd.DataFrame) -> "ItemKNNRecommender":
        self._user_ids = train_df["user_id"].unique()
        self._item_ids = train_df["movie_id"].unique()
        self._user2idx = {u: i for i, u in enumerate(self._user_ids)}
        self._item2idx = {m: i for i, m in enumerate(self._item_ids)}

        n_users = len(self._user_ids)
        n_items = len(self._item_ids)

        row = train_df["user_id"].map(self._user2idx).values
        col = train_df["movie_id"].map(self._item2idx).values
        data = train_df["rating"].values.astype(np.float32)

        R_T = csr_matrix((data, (col, row)), shape=(n_items, n_users))

        # IDF-взвешивание по пользователям
        n_ratings_per_user = np.array((R_T > 0).sum(axis=0)).flatten()
        idf = np.log((n_items + 1) / (n_ratings_per_user + 1)) + 1
        R_T_weighted = R_T.multiply(idf)

        R_T_normed = normalize(R_T_weighted, norm='l2', axis=1)
        self._item_sim = (R_T_normed @ R_T_normed.T).toarray()

        # Для каждого айтема оставляем только top-K соседей
        np.fill_diagonal(self._item_sim, 0)

        # Матрица user x item для подсчёта скоров
        self._R = csr_matrix((data, (row, col)), shape=(n_users, n_items))
        self._seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

        self._is_fitted = True
        logger.info(f"ItemKNNRecommender: {n_items} фильмов, {self.n_neighbors} соседей")
        return self

    def recommend(
        self,
        user_ids: List[int],
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        self._check_fitted()
        k = top_k or self.top_k

        valid_users = [u for u in user_ids if u in self._user2idx]
        records = []

        for uid in valid_users:
            uidx = self._user2idx[uid]
            user_ratings = self._R[uidx].toarray().flatten()   # (n_items,)

            # Взвешенная сумма рейтингов соседей
            rated_mask = user_ratings > 0
            if rated_mask.sum() == 0:
                continue

            sim_to_rated = self._item_sim[:, rated_mask]        # (n_items, n_rated)
            numerator = sim_to_rated @ user_ratings[rated_mask] # (n_items,)
            denominator = np.abs(sim_to_rated).sum(axis=1) + 1e-8
            scores = numerator / denominator

            # Исключаем просмотренные
            seen = self._seen.get(uid, set())
            seen_indices = [self._item2idx[m] for m in seen if m in self._item2idx]
            scores[seen_indices] = -np.inf

            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            for rank, idx in enumerate(top_indices, 1):
                records.append({
                    "user_id": uid,
                    "movie_id": int(self._item_ids[idx]),
                    "score": float(scores[idx]),
                    "rank": rank,
                })

        return pd.DataFrame(records)
