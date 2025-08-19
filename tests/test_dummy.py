import sys
import os
import pandas as pd
import numpy as np
import pytest

from src.reco import recommend_similar_movies, recommend_by_user_ratings
from src.svd_recommender import train_svd_model, get_top_n_recommendations_svd
from src.content_based_reco import get_nlp_content_based_recommendations


# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


@pytest.fixture
def sample_movies():
    # Mapping simple titres -> ids (1..4)
    return pd.DataFrame(
        {"movie_id": [1, 2, 3], "title": ["Film A", "Film B", "Film C"]}
    )


def test_content_recommendation():
    df = pd.DataFrame(
        {
            "movie_id": [1, 2],
            "text_features": ["toy story adventure", "jumanji adventure"],
        }
    )
    matrix = np.identity(2)
    result = get_nlp_content_based_recommendations(1, matrix, df, top_n=1)
    assert len(result) == 1


def test_train_svd_model():
    dummy_data = pd.DataFrame(
        {"user_id": [1, 2, 3], "movie_id": [10, 20, 30], "rating": [4.0, 5.0, 3.0]}
    )
    model = train_svd_model(dummy_data)
    assert model is not None


def test_get_recommendations():
    dummy_data = pd.DataFrame(
        {"user_id": [1, 2, 3], "movie_id": [10, 20, 30], "rating": [4.0, 5.0, 3.0]}
    )

    # Entraîne le modèle
    svd_model = train_svd_model(dummy_data)

    # On choisit un user_id présent dans les données de test
    user_id = 1

    # Appel de la fonction avec tous les arguments nécessaires
    recommendations = get_top_n_recommendations_svd(user_id, dummy_data, svd_model)

    # Assertions
    assert isinstance(recommendations, pd.DataFrame)
    assert "movie_id" in recommendations.columns
    assert "predicted_rating" in recommendations.columns
    assert len(recommendations) <= 10  # si top_n=10 par défaut


def test_recommend_similar_movies():
    # Jeu de données minimal
    data = {
        "movie_id": [1, 2, 3],
        "Action": [1, 0, 1],
        "Comedy": [0, 1, 1],
    }
    movie_genres = pd.DataFrame(data)
    genre_columns = ["Action", "Comedy"]

    # ---- Cas normal ----
    result = recommend_similar_movies(1, movie_genres.copy(), genre_columns, top_n=2)
    assert not result.empty
    assert len(result) <= 2
    assert "movie_id" in result.columns
    assert "similarity" in result.columns
    assert all(result["movie_id"] != 1)  # ne recommande pas le même film

    # ---- Cas titre inexistant ----
    result_empty = recommend_similar_movies(99, movie_genres.copy(), genre_columns)
    assert isinstance(result_empty, pd.DataFrame)
    assert result_empty.empty

    # ---- Cas top_n plus grand que dataset ----
    result_large_n = recommend_similar_movies(
        1, movie_genres.copy(), genre_columns, top_n=10
    )
    assert len(result_large_n) <= 2  # Seulement 2 autres films dispo

    # ---- Cas un seul film dans le dataset ----
    one_film_df = pd.DataFrame({"movie_id": [1], "Action": [1], "Comedy": [0]})
    result_one = recommend_similar_movies(1, one_film_df.copy(), genre_columns)
    assert result_one.empty


def test_recommend_by_user_ratings(sample_movies):
    # Jeu de données minimal : 3 films, 3 utilisateurs
    ratings_data = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movie_id": [1, 2, 1, 3, 2, 3],
            "rating": [5, 4, 4, 5, 3, 4],
        }
    )

    # Matrice utilisateur-film
    user_movie_matrix = ratings_data.pivot_table(
        index="userId", columns="movie_id", values="rating"
    )

    # ---- Cas normal ----
    result = recommend_by_user_ratings(
        1,
        user_movie_matrix.copy(),
        ratings_data,
        sample_movies,
        min_ratings=1,
        top_n=2,
    )
    assert not result.empty
    assert "movie_id" in result.columns or result.columns[0] == "movie_id"
    assert "correlation" in result.columns
    assert "num_ratings" in result.columns
    assert all(result["num_ratings"] >= 1)
    assert all(result["movie_id"] != 1)

    # ---- Cas titre inexistant dans la matrice ----
    # Ici, on doit vérifier qu'un KeyError peut survenir
    try:
        recommend_by_user_ratings(
            99, user_movie_matrix.copy(), ratings_data, sample_movies
        )
        assert False, "Un ValueError aurait dû être levé"
    except ValueError:
        pass  # comportement attendu

    # ---- Cas min_ratings élevé (aucun film ne passe le filtre) ----
    result_empty = recommend_by_user_ratings(
        1, user_movie_matrix.copy(), ratings_data, sample_movies, min_ratings=10
    )
    assert result_empty.empty

    # ---- Cas top_n plus grand que disponible ----
    result_large_n = recommend_by_user_ratings(
        1,
        user_movie_matrix.copy(),
        ratings_data,
        sample_movies,
        min_ratings=1,
        top_n=10,
    )
    assert len(result_large_n) <= 2  # Seulement 2 autres films dispo
