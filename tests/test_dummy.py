import sys
import os
import pandas as pd
import numpy as np

# from src.reco import recommend_similar_movies, recommend_by_user_ratings
# from src.data_preprocessing import (
#                       load_data, GENRE_COLUMNS,
#                       compute_item_similarity_matrix)
# from src.collaborative_filtering import (
#                       get_similar_movies_cosine,
#                       get_top_n_recommendations_knn)
from src.svd_recommender import train_svd_model, get_top_n_recommendations_svd
# from src.content_based_reco import (
#                       get_nlp_content_based_recommendations,
#                       merge_movies_overviews,
#                       load_content_based_data)
from src.content_based_reco import get_nlp_content_based_recommendations
# uncommented import when we will add more tests


def test_basic_math():
    assert 1+1 == 2


# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


def test_content_recommendation():
    df = pd.DataFrame({
        'title': ['Toy Story (1995)', 'Jumanji (1995)'],
        'title_norm': ['toy story (1995)', 'jumanji (1995)'],
        'text_features': ['toy story adventure', 'jumanji adventure']
    })
    matrix = np.identity(2)
    result = get_nlp_content_based_recommendations('Toy Story (1995)',
                                                   matrix, df, top_n=1)
    print(len(result))
    assert len(result) == 1


def test_train_svd_model():
    dummy_data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'movie_id': [10, 20, 30],
        'rating': [4.0, 5.0, 3.0]
    })
    model = train_svd_model(dummy_data)
    assert model is not None


def test_get_recommendations():
    dummy_data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'movie_id': [10, 20, 30],
        'rating': [4.0, 5.0, 3.0]
    })

    # Entraîne le modèle
    svd_model = train_svd_model(dummy_data)

    # On choisit un user_id présent dans les données de test
    user_id = 1

    # Appel de la fonction avec tous les arguments nécessaires
    recommendations = get_top_n_recommendations_svd(user_id,
                                                    dummy_data, svd_model)

    # Assertions
    assert isinstance(recommendations, pd.DataFrame)
    assert "movie_id" in recommendations.columns
    assert "predicted_rating" in recommendations.columns
    assert len(recommendations) <= 10  # si top_n=10 par défaut
