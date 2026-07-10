"""
Матричная факторизация с BPR loss на PyTorch
=============================================
BPR (Bayesian Personalized Ranking):
  для каждого пользователя u учим, что
  просмотренный фильм i лучше непросмотренного j:
  p(i >_u j) = sigmoid(score(u,i) - score(u,j))

Архитектура:
  Embedding(user) + Embedding(item) -> dot product -> BPR loss
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class BPRDataset(Dataset):
    """Датасет для BPR: триплеты (user, pos_item, neg_item).

    Для каждого взаимодействия (u, i) сэмплируем случайный
    непросмотренный фильм j как негативный пример.
    """

    def __init__(self, train_df: pd.DataFrame, n_items: int, n_neg: int = 1):
        """
        Args:
            train_df: DataFrame [user_id, movie_id]
            n_items:  общее число фильмов
            n_neg:    число негативных примеров на один позитив
        """
        self.n_items = n_items
        self.n_neg = n_neg

        # Индексированные взаимодействия
        self.users = train_df["user_idx"].values
        self.items = train_df["item_idx"].values

        # Множество просмотренных фильмов для каждого пользователя
        self.user_seen = (
            train_df.groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

    def __len__(self) -> int:
        return len(self.users) * self.n_neg

    def __getitem__(self, idx: int):
        base_idx = idx // self.n_neg
        u = self.users[base_idx]
        i = self.items[base_idx]

        # Сэмплируем негативный пример
        seen = self.user_seen.get(u, set())
        j = np.random.randint(self.n_items)
        while j in seen:
            j = np.random.randint(self.n_items)

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long),
        )


class MatrixFactorization(nn.Module):
    """MF модель: два embedding слоя для пользователей и фильмов."""

    def __init__(self, n_users: int, n_items: int, n_factors: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Скалярное произведение эмбеддингов пользователя и фильма."""
        u = self.user_emb(user_ids)   # (batch, n_factors)
        i = self.item_emb(item_ids)   # (batch, n_factors)
        return (u * i).sum(dim=1)     # (batch,)

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """BPR loss: -mean( log sigmoid( score(u,i) - score(u,j) ) )"""
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return loss


class MFRecommender:
    """Обёртка для обучения MF и генерации кандидатов."""

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        top_k: int = 100,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "MFRecommender":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Кодируем user_id и movie_id в индексы
        self._user_ids = train_df["user_id"].unique()
        self._item_ids = train_df["movie_id"].unique()
        self._user2idx = {u: i for i, u in enumerate(self._user_ids)}
        self._item2idx = {m: i for i, m in enumerate(self._item_ids)}

        n_users = len(self._user_ids)
        n_items = len(self._item_ids)

        train_indexed = train_df.copy()
        train_indexed["user_idx"] = train_indexed["user_id"].map(self._user2idx)
        train_indexed["item_idx"] = train_indexed["movie_id"].map(self._item2idx)

        dataset = BPRDataset(train_indexed, n_items)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.model = MatrixFactorization(n_users, n_items, self.n_factors).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self._losses = []
        self.model.train()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for users, pos_items, neg_items in loader:
                users     = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                optimizer.zero_grad()
                loss = self.model.bpr_loss(users, pos_items, neg_items)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self._losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {avg_loss:.4f}")
                print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {avg_loss:.4f}")

        self._seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        self._is_fitted = True
        return self

    def recommend(
        self,
        user_ids: List[int],
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Генерация кандидатов через скалярное произведение эмбеддингов."""
        if not self._is_fitted:
            raise RuntimeError("Вызовите .fit() перед .recommend()")

        k = top_k or self.top_k
        self.model.eval()

        # Все эмбеддинги фильмов сразу
        all_item_indices = torch.arange(len(self._item_ids), device=self.device)
        with torch.no_grad():
            item_embs = self.model.item_emb(all_item_indices).cpu().numpy()  # (n_items, n_factors)

        records = []
        valid_users = [u for u in user_ids if u in self._user2idx]

        with torch.no_grad():
            for uid in valid_users:
                uidx = torch.tensor(self._user2idx[uid], device=self.device)
                user_emb = self.model.user_emb(uidx).cpu().numpy()  # (n_factors,)

                scores = item_embs @ user_emb  # (n_items,)

                # Фильтруем просмотренные
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

    def get_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает эмбеддинги пользователей и фильмов.
        Можно использовать как фичи для ранкера Stage 2.
        """
        self.model.eval()
        with torch.no_grad():
            user_embs = self.model.user_emb.weight.cpu().numpy()
            item_embs = self.model.item_emb.weight.cpu().numpy()
        return user_embs, item_embs

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "MFRecommender":
        return joblib.load(path)
