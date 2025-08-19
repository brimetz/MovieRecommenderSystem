import pandas as pd
import numpy as np
import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_similar_movies_pearson(
    target_id: int,
    user_movie_matrix: pd.DataFrame,
    min_common_ratings: int = 10,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    will recommend movie similar to `target_id`
    using the pearson correlation

    Args:
        target_id (int): movie choose by the user
        user_movie_matrix (panda dataframe): Matrix Users x movies
        min_common_ratings (integer): minimal threshold of shared ratings
        top_n (integer): number of movies to recommend

    Returns:
        panda dataframe: movie array with their correlation scoring
    """
    if target_id not in user_movie_matrix.columns:
        return pd.DataFrame(columns=["movie_id", "correlation", "num_common_ratings"])

    # scoring vectors for the target movie
    target_ratings: pd.Series[int] = user_movie_matrix[target_id]

    # correlation compute with other movies
    correlations: pd.Series[float] = user_movie_matrix.corrwith(target_ratings)

    # create a dataframe with the result
    corr_df: pd.DataFrame = pd.DataFrame(correlations, columns=["correlation"])
    corr_df.dropna(inplace=True)

    # add shared scoring number to filter movies lesser score
    num_common: int = user_movie_matrix.apply(
        lambda x: target_ratings.notna() & x.notna(), axis=0
    ).sum()
    corr_df["num_common_ratings"] = num_common

    corr_df = corr_df[corr_df["num_common_ratings"] >= min_common_ratings].copy()

    # delete target movie from the result dataframe
    if target_id in corr_df.index:
        corr_df = corr_df.drop(index=target_id)

    # filters and rename
    corr_df = corr_df.sort_values(by="correlation", ascending=False)
    corr_df.reset_index(inplace=True)
    corr_df.rename(columns={"index": "movie_id"}, inplace=True)

    return corr_df.head(top_n)


def get_similar_movies_cosine(
    target_id: int,
    user_movie_matrix: pd.DataFrame,
    min_common_ratings: int = 10,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommand similar movies of 'target_id' using the cosine similarity

    Args:
        target_id (int): movie chosen by the user
        user_movie_matrix (panda dataframe): Users x movies matrix
        min_common_ratings (integer): minimal numbers of shared ratings
        top_n (integer): number of movie to return

    Returns:
        pd.Dataframe: movie array with similarity score
    """
    if target_id is None:
        return pd.DataFrame(columns=["movie_id", "similarity", "num_common_ratings"])

    # If the movie is not in the matrix
    if target_id not in user_movie_matrix.columns:
        return pd.DataFrame(columns=["movie_id", "similarity", "num_common_ratings"])

    # Transpose movie matrix
    movie_matrix: pd.DataFrame = user_movie_matrix.T

    if movie_matrix.loc[target_id].isna().all():
        return pd.DataFrame(columns=["movie_id", "similarity", "num_common_ratings"])

    # replace NaN value by O
    filled: pd.DataFrame = movie_matrix.fillna(0)
    sim_mat = cosine_similarity(filled)
    sim_df: pd.DataFrame = pd.DataFrame(
        sim_mat, index=filled.index, columns=filled.index
    )

    # remove the movie target
    sim_scores: pd.DataFrame = sim_df[target_id].drop(target_id)

    # Compute shared ratings with other movies
    target_ratings: pd.Series[int] = movie_matrix.loc[target_id]
    num_common: int = movie_matrix.apply(
        lambda x: (target_ratings.notna() & x.notna()).sum(), axis=1
    ).drop(target_id)

    # join each array to be sure to have the same number of element
    result: pd.DataFrame = pd.DataFrame(
        {"similarity": sim_scores, "num_common_ratings": num_common}
    )

    # filters and sorting
    result: pd.DataFrame = result[result["num_common_ratings"] >= min_common_ratings]
    result: pd.DataFrame = result.sort_values(by="similarity", ascending=False)

    result: pd.DataFrame = result.reset_index().rename(columns={"index": "movie_id"})

    return result[["movie_id", "similarity", "num_common_ratings"]].head(top_n)


def predict_rating(user_id: int, movie_id: int, ratings_matrix: pd.DataFrame) -> float:
    # Get movie ratings
    target_movie_ratings: pd.Series[int] = ratings_matrix[movie_id]

    # Compute correlation with other movies
    movie_corr: pd.Series[float] = ratings_matrix.corrwith(target_movie_ratings)

    # Get movies rate by the user
    user_ratings: pd.Series[int] = ratings_matrix.loc[user_id].dropna()

    # Keep only movies with known correlation
    valid_corrs: pd.Series[float] = movie_corr[user_ratings.index].dropna()

    # weighting
    numerator: float = sum(valid_corrs * user_ratings[valid_corrs.index])
    denominator: float = sum(np.abs(valid_corrs))

    if denominator == 0:
        return np.nan
    return numerator / denominator


def predict_rating_fast(
    user_id: int,
    movie_id: int,
    ratings_matrix: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
) -> float:
    if movie_id not in similarity_matrix.columns:
        return np.nan

    # user ratings
    user_ratings: pd.DataFrame = ratings_matrix.loc[user_id].dropna()

    # Similaries between target movie and movies rated by the user
    similarities: pd.DataFrame = similarity_matrix.loc[
        movie_id, user_ratings.index
    ].dropna()

    # same filtering of movies rating + similar movies
    user_ratings: pd.DataFrame = user_ratings[similarities.index]

    numerator: float = sum(similarities * user_ratings)
    denominator: float = sum(np.abs(similarities))

    if denominator == 0:
        return np.nan
    return numerator / denominator


def predict_mean_rating(movie_id: int, train_matrix: pd.DataFrame) -> float:
    if movie_id not in train_matrix.columns:
        return np.nan
    return train_matrix[movie_id].mean()


def predict_random_rating(min: int = 1, max: int = 5) -> float:
    return random.uniform(min, max)


def predict_rating_knn_item(
    user_id: int,
    movie_id: int,
    ratings_matrix: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    k: int = 5,
) -> float:
    if movie_id not in similarity_matrix.columns:
        return np.nan

    # Similar movies sorted by similarities (except target movie)
    similar_movies: pd.DataFrame = similarity_matrix[movie_id].drop(movie_id).dropna()
    similar_movies: pd.DataFrame = similar_movies.sort_values(ascending=False)

    # Movies that user has rated
    user_ratings: pd.DataFrame = ratings_matrix.loc[user_id].dropna()

    # Keep movies that are similar to the target movie and rated by the user
    similar_and_rated: pd.DataFrame = similar_movies[
        similar_movies.index.isin(user_ratings.index)
    ]

    if similar_and_rated.empty:
        return np.nan

    # Top k neighbours
    top_k: pd.DataFrame = similar_and_rated.head(k)

    # Weight user's ratings per similarity
    numerator: float = sum(top_k * user_ratings[top_k.index])
    denominator: float = sum(np.abs(top_k))

    if denominator == 0:
        return np.nan

    return numerator / denominator


def evaluate_model(
    predict_fn: int, test_df: pd.DataFrame, train_matrix: pd.DataFrame
) -> tuple[float, float]:
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        user_id: int = row["user_id"]
        movie_id: str = row["movie_id"]
        true_rating: float = row["rating"]
        pred_rating: float = predict_fn(user_id, movie_id, train_matrix)

        if not np.isnan(pred_rating):
            y_true.append(true_rating)
            y_pred.append(pred_rating)

    mse: float = mean_squared_error(y_true, y_pred)
    rmse: float = np.sqrt(mse)
    mae: float = mean_absolute_error(y_true, y_pred)

    return rmse, mae


def get_top_n_recommendations_knn(
    user_id: int,
    ratings_matrix: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    k: int = 5,
    N: int = 10,
) -> pd.DataFrame:
    # Get movies not rated by the user
    user_ratings: pd.DataFrame = ratings_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings.isna()].index

    recommendations: pd.DataFrame = []

    for movie in unseen_movies:
        pred: float = predict_rating_knn_item(
            user_id, movie, ratings_matrix, similarity_matrix, k
        )
        if not np.isnan(pred):
            recommendations.append((movie, pred))

    # Sort prediction per descending grade
    recommendations.sort(key=lambda x: x[1], reverse=True)

    column_name: str = "Note prédite (estimation d'appréciation)"
    # Build DataFrame
    recommendations: pd.DataFrame = pd.DataFrame(
        recommendations, columns=["movie_id", column_name]
    )

    recommendations[column_name] = pd.to_numeric(
        recommendations[column_name], errors="coerce"
    )
    recommendations[column_name] = recommendations[column_name].round(2)
    return recommendations.head(N)
