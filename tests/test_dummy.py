def test_basic_math():
    assert 1+1 == 2

from pathlib import Path
import sys
import os

# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.reco import recommend_similar_movies, recommend_by_user_ratings
from src.data_preprocessing import load_data, GENRE_COLUMNS, compute_item_similarity_matrix
from src.collaborative_filtering import get_similar_movies_cosine, get_top_n_recommendations_knn
from src.svd_recommender import train_svd_model, get_top_n_recommendations_svd
from src.content_based_reco import get_nlp_content_based_recommendations, merge_movies_overviews, load_content_based_data

import pandas as pd
import numpy as np

def test_content_recommendation():
    df = pd.DataFrame({
        'title': ['Toy Story (1995)', 'Jumanji (1995)'],
        'title_norm': ['toy story (1995)', 'jumanji (1995)'],
        'text_features': ['toy story adventure', 'jumanji adventure']
    })
    matrix = np.identity(2)
    result = get_nlp_content_based_recommendations('Toy Story (1995)', matrix, df, top_n=1)
    print(len(result))
    assert len(result) == 1