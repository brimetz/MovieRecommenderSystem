import pandas as pd
import streamlit as st


# Colonnes de genres utilisées
GENRE_COLUMNS = [
    "unkown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


@st.cache_data
def load_movies(path="data/u.item"):
    movies = pd.read_csv(
        path,
        sep="|",
        encoding="latin-1",
        header=None,
        names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"]
        + GENRE_COLUMNS,
    )
    return movies


@st.cache_data
def load_ratings(path="data/u.data"):
    ratings = pd.read_csv(
        path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
    )
    return ratings


@st.cache_data
def get_genre_matrix(movies_df):
    return movies_df[["movie_id", "title"] + GENRE_COLUMNS].copy()


def merge_ratings_with_titles(ratings_df, movies_df):
    return ratings_df.merge(movies_df[["movie_id", "title"]], on="movie_id")


@st.cache_data
def get_user_movie_matrix(ratings_df):
    return ratings_df.pivot_table(index="user_id", columns="title", values="rating")


@st.cache_data
def load_data(movies_path="data/u.item", ratings_path="data/u.data"):
    movies = load_movies(movies_path)
    ratings = load_ratings(ratings_path)
    movie_genres = get_genre_matrix(movies)
    ratings = merge_ratings_with_titles(ratings, movies)
    user_movie_matrix = get_user_movie_matrix(ratings)

    density = compute_density(user_movie_matrix)
    sparsity = compute_sparsity(user_movie_matrix)
    print(f"Taux de remplissage (density) : {density:.4f}")
    print(f"Sparsité (sparsity) : {sparsity:.4f}")

    return movie_genres, ratings, user_movie_matrix, movies


def compute_density(user_movie_matrix):
    total_cells = user_movie_matrix.size
    known_ratings = user_movie_matrix.count().sum()
    density = known_ratings / total_cells
    return density


def compute_sparsity(user_movie_matrix):
    return 1 - compute_density(user_movie_matrix)


@st.cache_data
def compute_item_similarity_matrix(user_item_matrix):
    return user_item_matrix.corr(method="pearson", min_periods=5)
