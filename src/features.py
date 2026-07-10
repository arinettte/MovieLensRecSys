"""
Feature Engineering для ранжирования
===============================================
Три группы признаков:
  1. Признаки пользователя (история взаимодействий)
  2. Признаки айтема (характеристики фильма)
  3. Признаки взаимодействия (пересечение пользователя и айтема)
"""

from __future__ import annotations

import pandas as pd


def build_user_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Признаки пользователя из истории оценок.

    Args:
        train_df: DataFrame [user_id, movie_id, rating, timestamp]

    Returns:
        DataFrame [user_id, user_*]
    """
    features = (
        train_df.groupby("user_id")["rating"]
        .agg(
            user_n_ratings="count",
            user_mean_rating="mean",
            user_std_rating="std",
            user_min_rating="min",
            user_max_rating="max",
        )
        .reset_index()
    )

    # Доля высоких оценок (>= 4)
    high_rated = train_df[train_df["rating"] >= 4].groupby("user_id").size().rename("user_n_high")
    features = features.merge(high_rated, on="user_id", how="left")
    features["user_high_rate"] = features["user_n_high"] / features["user_n_ratings"]
    features = features.drop(columns=["user_n_high"])

    return features


def build_item_features(train_df: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Признаки айтема: статистика оценок + метаданные.

    Args:
        train_df: DataFrame [user_id, movie_id, rating, timestamp]
        movies:   DataFrame [movie_id, title, genres]

    Returns:
        DataFrame [movie_id, item_*]
    """
    features = (
        train_df.groupby("movie_id")["rating"]
        .agg(
            item_n_ratings="count",
            item_mean_rating="mean",
            item_std_rating="std",
        )
        .reset_index()
    )

    C = features["item_mean_rating"].mean()
    m = 50
    v = features["item_n_ratings"]
    R = features["item_mean_rating"]
    features["item_weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

    movies_copy = movies.copy()
    movies_copy["item_year"] = movies_copy["title"].str.extract(r"\((\d{4})\)").astype(float)
    movies_copy["item_n_genres"] = movies_copy["genres"].str.split("|").apply(len)

    features = features.merge(
        movies_copy[["movie_id", "item_year", "item_n_genres", "genres"]],
        on="movie_id",
        how="left",
    )

    features = features.rename(columns={"genres": "item_genres"})
    features["item_std_rating"] = features["item_std_rating"].fillna(0)
    return features


def build_interaction_features(
    candidates: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    df = candidates.copy()
    df = df.merge(user_features, on="user_id", how="left")
    df = df.merge(item_features, on="movie_id", how="left")

    df["interact_rating_diff"] = df["user_mean_rating"] - df["item_mean_rating"]

    genre_pref = _build_genre_preference(train_df, item_features, candidates)
    df = df.merge(genre_pref, on=["user_id", "movie_id"], how="left")
    df["user_genre_affinity"] = df["user_genre_affinity"].fillna(0)

    return df


def _build_genre_preference(
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Для каждой пары (user, movie) из кандидатов считаем
    средний рейтинг пользователя по жанрам этого фильма."""

    # Средний рейтинг пользователя по каждому жанру
    rated = train_df.merge(
        item_features[["movie_id", "item_genres"]], on="movie_id", how="left"
    ).dropna(subset=["item_genres"])

    rated_exploded = rated.copy()
    rated_exploded["genre"] = rated_exploded["item_genres"].str.split("|")
    rated_exploded = rated_exploded.explode("genre")

    user_genre_rating = (
        rated_exploded.groupby(["user_id", "genre"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "genre_mean_rating"})
    )

    # Разворачиваем жанры только для фильмов из кандидатов
    candidate_movies = candidates[["movie_id"]].drop_duplicates()
    item_genres_exp = candidate_movies.merge(
        item_features[["movie_id", "item_genres"]], on="movie_id", how="left"
    )
    item_genres_exp["genre"] = item_genres_exp["item_genres"].str.split("|")
    item_genres_exp = item_genres_exp.explode("genre")

    # Джойним только реальные пары (user, movie) из кандидатов
    candidate_pairs = candidates[["user_id", "movie_id"]].drop_duplicates()
    candidate_pairs = candidate_pairs.merge(
        item_genres_exp[["movie_id", "genre"]], on="movie_id", how="left"
    )
    candidate_pairs = candidate_pairs.merge(
        user_genre_rating, on=["user_id", "genre"], how="left"
    )

    affinity = (
        candidate_pairs.groupby(["user_id", "movie_id"])["genre_mean_rating"]
        .mean()
        .reset_index()
        .rename(columns={"genre_mean_rating": "user_genre_affinity"})
    )

    return affinity


FEATURE_COLUMNS = [
    # Пользователь
    "user_n_ratings",
    "user_mean_rating",
    "user_std_rating",
    "user_min_rating",
    "user_max_rating",
    "user_high_rate",
    # Айтем
    "item_n_ratings",
    "item_mean_rating",
    "item_std_rating",
    "item_weighted_rating",
    "item_year",
    "item_n_genres",
    # Взаимодействие
    "interact_rating_diff",
    "user_genre_affinity",
    # Из Stage 1
    "candidate_score",
    "candidate_rank",
    "in_popularity_top",
]

CAT_FEATURE_COLUMNS = []
