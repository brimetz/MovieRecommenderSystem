import pandas as pd
import streamlit as st


# Columnns of genres used
GENRE_COLUMNS: list[str] = [
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
def load_movies(path: str = "data/u.item") -> pd.DataFrame:
    movies: pd.DataFrame = pd.read_csv(
        path,
        sep="|",
        encoding="latin-1",
        header=None,
        names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"]
        + GENRE_COLUMNS,
    )
    return movies


@st.cache_data
def load_ratings(path: str = "data/u.data") -> pd.DataFrame:
    ratings: pd.DataFrame = pd.read_csv(
        path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
    )
    return ratings


@st.cache_data
def get_genre_matrix(movies_df: pd.DataFrame) -> pd.DataFrame:
    return movies_df[["movie_id", "title"] + GENRE_COLUMNS].copy()


def merge_ratings_with_titles(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> pd.DataFrame:
    return ratings_df.merge(movies_df[["movie_id", "title"]], on="movie_id")


@st.cache_data
def get_user_movie_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return ratings_df.pivot_table(index="user_id", columns="title", values="rating")


@st.cache_data
def load_data(
    movies_path: str = "data/u.item", ratings_path: str = "data/u.data"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies: pd.DataFrame = load_movies(movies_path)
    ratings: pd.DataFrame = load_ratings(ratings_path)
    movie_genres: pd.DataFrame = get_genre_matrix(movies)
    ratings: pd.DataFrame = merge_ratings_with_titles(ratings, movies)
    user_movie_matrix: pd.DataFrame = get_user_movie_matrix(ratings)
    return movie_genres, ratings, user_movie_matrix, movies


def compute_density(user_movie_matrix: pd.DataFrame) -> float:
    total_cells: int = user_movie_matrix.size
    known_ratings: int = user_movie_matrix.count().sum()
    density: float = known_ratings / total_cells
    return density


def compute_sparsity(user_movie_matrix: pd.DataFrame) -> float:
    return 1 - compute_density(user_movie_matrix)


@st.cache_data
def compute_item_similarity_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    return user_item_matrix.corr(method="pearson", min_periods=5)
