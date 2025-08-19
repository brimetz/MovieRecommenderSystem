import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend_similar_movies(
    movie_id: int,
    movie_genres: pd.DataFrame,
    genre_columns: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend similar movies from genres vector

    Args:
        movie_id (int): id of the movie
        movie_genres (pd.DataFrame): DataFrame with a title column and
                                    a column for each genres in binary value.
        genre_columns (list[str]): list of columns used for genres.
        top_n (int, optional): numbers of movie to recommend. default 5.

    Returns:
        pd.DataFrame: DataFrame with similar movies and their similarity score.
    """
    # DataFrame with only the target movie
    selected: pd.DataFrame = movie_genres[movie_genres["movie_id"] == movie_id]
    if selected.empty:
        return pd.DataFrame(columns=["movie_id", "similarity"])

    similarities: np.ndarray = cosine_similarity(
        selected[genre_columns], movie_genres[genre_columns]
    )

    # Add a similarity column in the movie_genres copy
    movie_genres = movie_genres.copy()
    movie_genres["similarity"] = similarities[0]

    # Remove the targetted movie
    similar_movies = movie_genres[movie_genres["movie_id"] != movie_id]

    # return a dataframe only with the best similarity
    return similar_movies.sort_values(by="similarity", ascending=False)[
        ["movie_id", "similarity"]
    ].head(top_n)


def recommend_by_user_ratings(
    movie_id: int,
    user_movie_matrix: pd.DataFrame,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_ratings: int = 50,
    top_n: int = 5,
) -> pd.DataFrame:
    # --- Verify if the matrix is index by ids or by title ---
    if movie_id in user_movie_matrix.columns:
        col_key = movie_id
    else:
        raise ValueError(f"No rating find for (id={movie_id}).")

    # --- ratings of movie_id ---
    target_ratings: pd.Series[int] = user_movie_matrix[col_key]

    # --- Compute correlation with other movies ---
    other_movies: pd.DataFrame = user_movie_matrix.drop(columns=[col_key])
    similar_scores: pd.Series[float] = other_movies.corrwith(target_ratings)

    # --- DataFrame clean ---
    corr_df: pd.DataFrame = pd.DataFrame(similar_scores, columns=["correlation"])
    corr_df.dropna(inplace=True)

    # --- Add numbers of rate per movie ---
    rating_counts: pd.Series[int] = ratings.groupby("movie_id")["rating"].count()

    corr_df["num_ratings"] = rating_counts
    # --- Filtering ---
    filtered_corr: pd.DataFrame = corr_df[corr_df["num_ratings"] >= min_ratings]

    # --- Sorting ---
    filtered_corr: pd.DataFrame = filtered_corr.sort_values(
        by="correlation", ascending=False
    )

    filtered_corr: pd.DataFrame = filtered_corr.merge(
        movies, left_index=True, right_on="movie_id"
    )

    # --- Remove the chosen movie ---
    filtered_corr: pd.DataFrame = filtered_corr[filtered_corr["movie_id"] != movie_id]

    return filtered_corr[["movie_id", "correlation", "num_ratings"]].head(top_n)
