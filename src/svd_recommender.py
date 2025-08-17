import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, Trainset
import streamlit as st


@st.cache_resource
def train_svd_model(ratings_df: pd.DataFrame) -> SVD:
    reader: Reader = Reader(rating_scale=(1, 5))
    data: Dataset = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]], reader
    )
    trainset: Trainset = data.build_full_trainset()

    svd: SVD = SVD()
    svd.fit(trainset)
    return svd


def get_top_n_recommendations_svd(
    user_id: int, ratings_df: pd.DataFrame, svd_model: SVD, top_n: int = 10
) -> pd.DataFrame:
    # Get all movies that the user has not rate
    seen: np.ndarray = ratings_df[ratings_df["user_id"] == user_id]["movie_id"].unique()
    all_movies: np.ndarray = ratings_df["movie_id"].unique()
    unseen: list = [movie for movie in all_movies if movie not in seen]

    predictions: list = [
        (movie, round(svd_model.predict(user_id, movie).est, 2)) for movie in unseen
    ]

    top_n: int = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    return pd.DataFrame(top_n, columns=["movie_id", "predicted_rating"])
