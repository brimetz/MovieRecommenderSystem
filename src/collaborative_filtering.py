import pandas as pd
import numpy as np
import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_similar_movies_pearson(
    target_title, user_movie_matrix, min_common_ratings=10, top_n=10
):
    """
    will recommend movie similar to `target_title`
    using the pearson correlation

    Args:
        target_title (string): movie choose by the user
        user_movie_matrix (panda dataframe): Matrix Users x movies
        min_common_ratings (integer): minimal threshold of shared ratings
        top_n (integer): number of movies to recommend

    Returns:
        panda dataframe: movie array with their correlation scoring
    """
    if target_title not in user_movie_matrix.columns:
        return pd.DataFrame(columns=["title", "correlation", "num_common_ratings"])

    # scoring vectors for the target movie
    target_ratings = user_movie_matrix[target_title]

    # correlation compute with other movies
    correlations = user_movie_matrix.corrwith(target_ratings)

    # create a dataframe with the result
    corr_df = pd.DataFrame(correlations, columns=["correlation"])
    corr_df.dropna(inplace=True)

    # add shared scoring number to filter movies lesser score
    num_common = user_movie_matrix.apply(
        lambda x: target_ratings.notna() & x.notna(), axis=0
    ).sum()
    corr_df["num_common_ratings"] = num_common

    corr_df = corr_df[corr_df["num_common_ratings"] >= min_common_ratings].copy()

    # delete target movie from the result dataframe
    if target_title in corr_df.index:
        corr_df = corr_df.drop(index=target_title)

    # filters and rename
    corr_df = corr_df.sort_values(by="correlation", ascending=False)
    corr_df.reset_index(inplace=True)
    corr_df.rename(columns={"index": "title"}, inplace=True)

    return corr_df.head(top_n)


def get_similar_movies_cosine(
    target_title: str, user_movie_matrix: pd.DataFrame, min_common_ratings=10, top_n=10
):
    """
    Recommand similar movies of 'target_title' using the cosine similarity

    Args:
        target_title (string): movie chosen by the user
        user_movie_matrix (panda dataframe): Users x movies matrix
        min_common_ratings (integer): minimal numbers of shared ratings
        top_n (integer): number of movie to return

    Returns:
        pd.Dataframe: movie array with similarity score
    """
    if target_title not in user_movie_matrix.columns:
        return pd.DataFrame(columns=["title", "similarity", "num_common_ratings"])

    # Transpose movie matrix
    movie_matrix = user_movie_matrix.T
    # if target movie has no ratings we stop
    if movie_matrix.loc[target_title].isna().all():
        return pd.DataFrame(columns=["title", "similarity", "num_common_ratings"])

    # replace NaN value by O
    filled = movie_matrix.fillna(0)
    sim_mat = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim_mat, index=filled.index, columns=filled.index)

    # remove the movie target
    sim_scores = sim_df[target_title].drop(target_title)

    # Compute shared ratings with other movies
    target_ratings = movie_matrix.loc[target_title]
    num_common = movie_matrix.apply(
        lambda x: (target_ratings.notna() & x.notna()).sum(), axis=1
    ).drop(target_title)

    # join each array to be sure to have the same number of element
    result = pd.DataFrame({"similarity": sim_scores, "num_common_ratings": num_common})

    # filters and sorting
    result = result[result["num_common_ratings"] >= min_common_ratings]
    result = result.sort_values(by="similarity", ascending=False)

    result = result.reset_index().rename(columns={"index": "title"})
    return result.head(top_n)


def predict_rating(user_id, movie_title: str, ratings_matrix: pd.DataFrame):
    # Obtenir les notes du film
    target_movie_ratings = ratings_matrix[movie_title]

    # Calculer la corrélation entre ce film et tous les autres
    movie_corr = ratings_matrix.corrwith(target_movie_ratings)

    # Récupérer les films notés par l’utilisateur
    user_ratings = ratings_matrix.loc[user_id].dropna()

    # Ne garder que les films avec corrélation connue
    valid_corrs = movie_corr[user_ratings.index].dropna()

    # Pondération
    numerateur = sum(valid_corrs * user_ratings[valid_corrs.index])
    denominateur = sum(np.abs(valid_corrs))

    if denominateur == 0:
        return np.nan
    return numerateur / denominateur


def predict_rating_fast(
    user_id, movie_title: str, ratings_matrix: pd.DataFrame, similarity_matrix
):
    if movie_title not in similarity_matrix.columns:
        return np.nan

    # Notes de l'utilisateur
    user_ratings = ratings_matrix.loc[user_id].dropna()

    # Similitudes entre le film cible et ceux notés par l'utilisateur
    similarities = similarity_matrix.loc[movie_title, user_ratings.index].dropna()

    # Même filtrage des films notés + corrélés
    user_ratings = user_ratings[similarities.index]

    numerateur = sum(similarities * user_ratings)
    denominateur = sum(np.abs(similarities))

    if denominateur == 0:
        return np.nan
    return numerateur / denominateur


def predict_mean_rating(movie_title, train_matrix):
    if movie_title not in train_matrix.columns:
        return np.nan
    return train_matrix[movie_title].mean()


def predict_random_rating(min=1, max=5):
    return random.uniform(min, max)


def predict_rating_knn_item(
    user_id, movie_title, ratings_matrix, similarity_matrix, k=5
):
    if movie_title not in similarity_matrix.columns:
        return np.nan

    # Films similaires triés par similarité (sauf lui-même)
    similar_movies = similarity_matrix[movie_title].drop(movie_title).dropna()
    similar_movies = similar_movies.sort_values(ascending=False)

    # Films que l'utilisateur a notés
    user_ratings = ratings_matrix.loc[user_id].dropna()

    # On conserve ceux que l'utilisateur a notés ET
    # qui sont similaires au film cible
    similar_and_rated = similar_movies[similar_movies.index.isin(user_ratings.index)]

    if similar_and_rated.empty:
        return np.nan

    # Top k voisins
    top_k = similar_and_rated.head(k)

    # Pondérer les notes de l'utilisateur par la similarité
    numerator = sum(top_k * user_ratings[top_k.index])
    denominator = sum(np.abs(top_k))

    if denominator == 0:
        return np.nan

    return numerator / denominator


def evaluate_model(predict_fn, test_df, train_matrix):
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        user_id = row["user_id"]
        title = row["title"]
        true_rating = row["rating"]
        pred_rating = predict_fn(user_id, title, train_matrix)

        if not np.isnan(pred_rating):
            y_true.append(true_rating)
            y_pred.append(pred_rating)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return rmse, mae


def get_top_n_recommendations_knn(
    user_id, ratings_matrix, similarity_matrix, k=5, N=10
):
    # Récupérer les films non notés par l'utilisateur
    user_ratings = ratings_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings.isna()].index

    recommendations = []

    for movie in unseen_movies:
        pred = predict_rating_knn_item(
            user_id, movie, ratings_matrix, similarity_matrix, k
        )
        if not np.isnan(pred):
            recommendations.append((movie, pred))

    # Trier les prédictions par note décroissante
    recommendations.sort(key=lambda x: x[1], reverse=True)

    column_name = "Note prédite (estimation d'appréciation)"
    # Construction du DataFrame avec arrondi
    recommendations = pd.DataFrame(
        recommendations, columns=["Film recommandé", column_name]
    )
    recommendations[column_name] = pd.to_numeric(recommendations[column_name], errors="coerce")
    recommendations[column_name] = recommendations[column_name].round(2)

    return recommendations.head(N)
