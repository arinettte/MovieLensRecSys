"""
Stage 2: Ранжирование кандидатов
==================================
CatBoostRanker с YetiRank loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from features import FEATURE_COLUMNS, CAT_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class TwoStageRanker:
    """Обёртка над CatBoostRanker для второй стадии RecSys."""
    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
        verbose: int = 100,
    ):
        self.model = CatBoostRanker(
            loss_function="YetiRank",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=verbose,
        )
        self._is_fitted = False

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "TwoStageRanker":
        """Обучить ранкер.

        Args:
            train_df: DataFrame с фичами + колонками [user_id, label]
            val_df:   опциональный val для early stopping
        """
        train_pool = self._make_pool(train_df)

        eval_set = None
        if val_df is not None:
            eval_set = self._make_pool(val_df)

        self.model.fit(train_pool, eval_set=eval_set)
        self._is_fitted = True
        logger.info("TwoStageRanker обучен")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Предсказать скоры релевантности."""
        pool = self._make_pool(df, with_label=False)
        return self.model.predict(pool)

    def recommend(
        self,
        candidates: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Ранжировать кандидатов и вернуть топ-K.

        Args:
            candidates: DataFrame с фичами для всех кандидатов
            top_k:      размер финального списка рекомендаций

        Returns:
            DataFrame [user_id, movie_id, score, rank]
        """
        candidates = candidates.copy()
        candidates["score"] = self.predict(candidates)

        result = (
            candidates.sort_values(["user_id", "score"], ascending=[True, False])
            .groupby("user_id")
            .head(top_k)
            .copy()
        )
        result["rank"] = result.groupby("user_id")["score"].rank(
            ascending=False, method="first"
        ).astype(int)

        return result[["user_id", "movie_id", "score", "rank"]]

    def _make_pool(self, df: pd.DataFrame, with_label: bool = True) -> Pool:
        """Создать CatBoost Pool из DataFrame."""
        # group_id нужен для YetiRank - группирует кандидатов по пользователю
        group_id = df["user_id"].values
        label = df["label"].values if with_label else None

        X = df[FEATURE_COLUMNS].fillna(0)

        return Pool(
            data=X,
            label=label,
            group_id=group_id,
            cat_features=CAT_FEATURE_COLUMNS,
            feature_names=FEATURE_COLUMNS,
        )

    def get_feature_importance(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Важность признаков. Для YetiRank нужен train pool."""
        train_pool = self._make_pool(train_df)
        importance = self.model.get_feature_importance(data=train_pool)
        names = self.model.feature_names_
        return (
            pd.DataFrame({"feature": names, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
    )

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TwoStageRanker":
        return joblib.load(path)
