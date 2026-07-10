"""
FastAPI сервис рекомендаций
============================
Endpoints:
  GET /recommend?user_id=123&top_k=10   - топ-K рекомендаций для пользователя
  GET /health                           - healthcheck
  GET /                                 - информация о сервисе
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import joblib

from mf_model import MFRecommender
from ranker import TwoStageRanker
from features import build_user_features, build_item_features, build_interaction_features


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: int
    top_k: int
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модели при старте сервиса."""
    print("Загрузка моделей...")

    app_state["mf"]     = MFRecommender.load(MODELS_DIR / "mf.pkl")
    app_state["ranker"] = TwoStageRanker.load(MODELS_DIR / "ranker_mf.pkl")

    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    movies   = pd.read_parquet(DATA_DIR / "movies.parquet")

    app_state["user_features"] = build_user_features(train_df)
    app_state["item_features"] = build_item_features(train_df, movies)
    app_state["train_df"]      = train_df
    app_state["movies"]        = movies[["movie_id", "title", "genres"]].set_index("movie_id")
    app_state["known_users"]   = set(train_df["user_id"].unique())

    print("Модели загружены")
    yield
    app_state.clear()


app = FastAPI(
    title="MovieLens RecSys API",
    description="Двустадийная рекомендательная система: MF (BPR) + CatBoostRanker",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["info"])
def root():
    return {
        "service": "MovieLens RecSys",
        "stage_1": "Matrix Factorization (PyTorch, BPR loss)",
        "stage_2": "CatBoostRanker (YetiRank)",
        "endpoints": ["/recommend", "/health"],
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    models_loaded = "mf" in app_state and "ranker" in app_state
    return HealthResponse(status="ok", models_loaded=models_loaded)


@app.get("/recommend", response_model=RecommendationResponse, tags=["recommend"])
def recommend(
    user_id: int = Query(..., description="ID пользователя из MovieLens 1M"),
    top_k: int   = Query(10, ge=1, le=50, description="Размер списка рекомендаций"),
):
    if user_id not in app_state["known_users"]:
        raise HTTPException(
            status_code=404,
            detail=f"Пользователь {user_id} не найден в обучающей выборке (cold start не поддерживается)"
        )

    mf     = app_state["mf"]
    ranker = app_state["ranker"]

    # Stage 1: генерация кандидатов
    # Stage 1: генерация кандидатов
    candidates = mf.recommend([user_id], top_k=100)
    candidates.columns = ["user_id", "movie_id", "candidate_score", "candidate_rank"]
    candidates["label"] = 0
    candidates["in_popularity_top"] = 0

    # Stage 2: feature engineering + ранжирование
    dataset = build_interaction_features(
        candidates,
        app_state["user_features"],
        app_state["item_features"],
        app_state["train_df"],
    )
    recs = ranker.recommend(dataset, top_k=top_k)

    # Добавляем метаданные фильмов
    movies = app_state["movies"]
    result = []
    for _, row in recs.iterrows():
        mid = int(row["movie_id"])
        movie_info = movies.loc[mid] if mid in movies.index else None
        result.append(RecommendationItem(
            movie_id=mid,
            title=movie_info["title"] if movie_info is not None else "Unknown",
            genres=movie_info["genres"] if movie_info is not None else "Unknown",
            score=float(row["score"]),
            rank=int(row["rank"]),
        ))

    return RecommendationResponse(
        user_id=user_id,
        top_k=top_k,
        recommendations=result,
    )
