import sqlite3
import pandas as pd
import numpy as np
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


# chemin du dossier courant (src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# on remonte d'un cran (..), puis vers data/
DB_PATH = os.path.join(BASE_DIR, "..", "data", "movielens_100k.db")
DB_PATH = os.path.abspath(DB_PATH)  # pour être sûr que c’est absolu


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def get_movies() -> pd.DataFrame:
    """load movies table (movie_id, title)."""
    conn: sqlite3.Connection = get_connection()
    query: str = "SELECT movie_id, title FROM movies"
    df: pd.DataFrame = pd.read_sql(query, conn)
    conn.close()
    return df


def get_ratings() -> pd.DataFrame:
    """load ratings table (user_id, movie_id, rating)."""
    conn: sqlite3.Connection = get_connection()
    query: str = "SELECT user_id, movie_id, rating FROM ratings"
    df: pd.DataFrame = pd.read_sql(query, conn)
    conn.close()
    return df


def get_users() -> pd.DataFrame:
    """Optional : get available users."""
    conn: sqlite3.Connection = get_connection()
    query: str = "SELECT DISTINCT user_id FROM ratings"
    df: pd.DataFrame = pd.read_sql(query, conn)
    conn.close()
    return df


def get_user_movie_matrix() -> pd.DataFrame:
    conn: sqlite3.Connection = sqlite3.connect(DB_PATH)

    # Charger les notations (ratings)
    ratings: pd.DataFrame = pd.read_sql(
        "SELECT user_id, movie_id, rating FROM ratings", conn
    )

    # Matrice user x movie
    user_movie_matrix: pd.DataFrame = ratings.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    )

    conn.close()
    return user_movie_matrix


def get_user_movies(user_id) -> pd.DataFrame:
    conn: sqlite3.Connection = sqlite3.connect(DB_PATH)
    query: str = f"""
    SELECT m.movie_id, m.title, r.rating
    FROM ratings r
    JOIN movies m ON r.movie_id = m.movie_id
    WHERE r.user_id = {user_id}
    """
    df: pd.DataFrame = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_ratings_matrix() -> pd.DataFrame:
    conn: sqlite3.Connection = sqlite3.connect(DB_PATH)
    ratings_df: pd.DataFrame = pd.read_sql_query("SELECT * FROM ratings", conn)
    conn.close()
    return ratings_df.pivot(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)


def collaborative_recommendations(user_id: int, n_neighbors=5) -> list[int]:
    ratings_matrix: pd.DataFrame = get_ratings_matrix()
    model_knn: NearestNeighbors = NearestNeighbors(metric="cosine", algorithm="brute")
    model_knn.fit(ratings_matrix.values)
    distances: np.ndarray
    indices: np.ndarray
    distances, indices = model_knn.kneighbors(
        [ratings_matrix.loc[user_id].values], n_neighbors=n_neighbors + 1
    )
    similar_users: pd.Index = ratings_matrix.index[indices.flatten()[1:]]  # ignore user
    return similar_users.tolist()


def content_recommendations(movie_title: str, top_n: int = 5) -> list[str]:
    conn: sqlite3.Connection = sqlite3.connect(DB_PATH)
    movies_df: pd.DataFrame = pd.read_sql_query("SELECT * FROM movies", conn)
    conn.close()

    fav_movie: pd.DataFrame = movies_df[movies_df["title"] == movie_title]
    fav_vector: np.ndarray = fav_movie.iloc[:, 5:].values  # colonnes genre_0 à genre_18
    similarities: np.ndarray = cosine_similarity(
        fav_vector, movies_df.iloc[:, 5:].values
    )
    similar_indices: np.ndarray = similarities[0].argsort()[::-1][1 : top_n + 1]
    return movies_df.iloc[similar_indices]["title"].tolist()


def load_content_based_data_sql() -> pd.DataFrame:
    conn: sqlite3.Connection = sqlite3.connect(DB_PATH)
    df: pd.DataFrame = pd.read_sql_query(
        "SELECT movie_id, title, genres FROM movies", conn
    )
    conn.close()
    return df


def build_user_movie_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    build a matrix userId x movieId from the ratings DataFrame.

    - ratings : DataFrame with columns (userId, movieId, rating)
    """
    user_movie_matrix: pd.DataFrame = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    return user_movie_matrix
