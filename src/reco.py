import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# === Fonction de reco ===
def recommend_similar_movies(film_title, movie_genres, genre_columns, top_n=5):
    selected = movie_genres[movie_genres["title"] == film_title]
    if selected.empty:
        return pd.DataFrame()

    similarities = cosine_similarity(
        selected[genre_columns], movie_genres[genre_columns]
    )

    movie_genres["similarity"] = similarities[0]
    similar_movies = movie_genres[movie_genres["title"] != film_title]
    return similar_movies.sort_values(by="similarity", ascending=False)[
        ["title", "similarity"]
    ].head(top_n)


def recommend_by_user_ratings(
    film_title, user_movie_matrix, ratings, min_ratings=50, top_n=5
):
    # Get the column of the selected movie
    target_ratings = user_movie_matrix[film_title]

    # Remove the Current film selected
    user_movie_matrix = user_movie_matrix.drop(columns=[film_title])

    # compute the correlation with other movies
    similar_scores = user_movie_matrix.corrwith(target_ratings)

    # Création d’un DataFrame propre
    corr_df = pd.DataFrame(similar_scores, columns=["correlation"])
    corr_df.dropna(inplace=True)

    # Ajout du nombre de notes par film
    rating_counts = ratings.groupby("title")["rating"].count()
    corr_df["num_ratings"] = rating_counts

    # Filtrage : au moins `min_ratings` pour plus de fiabilité
    filtered_corr = corr_df[corr_df["num_ratings"] >= min_ratings]

    filtered_corr = filtered_corr.sort_values(by="correlation", ascending=False)
    filtered_corr = filtered_corr[filtered_corr.index != film_title]
    # Top recommandations
    return filtered_corr.head(top_n).reset_index()
