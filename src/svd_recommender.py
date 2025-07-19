import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import streamlit as st

@st.cache_resource
def train_svd_model(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    svd = SVD()
    svd.fit(trainset)
    return svd

def get_top_n_recommendations_svd(user_id, ratings_df, svd_model, top_n=10):
    # Get all movies that the user has not rate
    seen = ratings_df[ratings_df["user_id"] == user_id]["movie_id"].unique()
    all_movies = ratings_df["movie_id"].unique()
    unseen = [movie for movie in all_movies if movie not in seen]

    predictions = [
        (movie, round(svd_model.predict(user_id, movie).est, 2))
        for movie in unseen
    ]

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    return pd.DataFrame(top_n, columns=["movie_id", "predicted_rating"])

