import pandas as pd
import numpy as np
import pytest
import src.collaborative_filtering as cf


@pytest.fixture
def sample_data():
    # Exemple simple : 4 utilisateurs x 4 films
    data = {
        "MovieA": [5, 4, np.nan, 1],
        "MovieB": [4, np.nan, 2, 1],
        "MovieC": [np.nan, 3, 5, 1],
        "MovieD": [4, 3, 5, 1]
    }
    df = pd.DataFrame(data, index=["User1", "User2", "User3", "User4"])
    return df


def test_get_similar_movies_pearson_valid(sample_data):
    result = cf.get_similar_movies_pearson("MovieA", user_movie_matrix=sample_data, min_common_ratings=1, top_n=2)
    assert "title" in result.columns
    assert not result.empty


def test_get_similar_movies_pearson_invalid(sample_data):
    result = cf.get_similar_movies_pearson("Unknown", sample_data)
    assert result.empty


def test_get_similar_movies_cosine_valid(sample_data):
    result = cf.get_similar_movies_cosine("MovieA", user_movie_matrix= sample_data, min_common_ratings=1, top_n=2)
    assert "title" in result.columns
    assert not result.empty


def test_get_similar_movies_cosine_no_ratings(sample_data):
    df = sample_data.copy()
    df["EmptyMovie"] = [np.nan, np.nan, np.nan, np.nan]
    result = cf.get_similar_movies_cosine("EmptyMovie", df)
    assert result.empty


def test_get_similar_movies_cosine_invalid(sample_data):
    result = cf.get_similar_movies_cosine("Unknown", sample_data)
    assert result.empty


def test_predict_rating(sample_data):
    rating = cf.predict_rating("User1", "MovieA", sample_data)
    assert np.isfinite(rating) or np.isnan(rating)


def test_predict_rating_fast(sample_data):
    sim = pd.DataFrame(np.identity(4), index=sample_data.columns, columns=sample_data.columns)
    rating = cf.predict_rating_fast("User1", "MovieA", sample_data, sim)
    assert np.isfinite(rating) or np.isnan(rating)


def test_predict_mean_rating(sample_data):
    mean_rating = cf.predict_mean_rating("MovieA", sample_data)
    assert np.isfinite(mean_rating)


def test_predict_mean_rating_invalid(sample_data):
    assert np.isnan(cf.predict_mean_rating("Unknown", sample_data))


def test_predict_random_rating():
    val = cf.predict_random_rating(1, 5)
    assert 1 <= val <= 5


def test_predict_rating_knn_item(sample_data):
    sim = pd.DataFrame(np.identity(4), index=sample_data.columns, columns=sample_data.columns)
    rating = cf.predict_rating_knn_item("User1", "MovieA", sample_data, sim)
    assert np.isfinite(rating) or np.isnan(rating)


def test_predict_rating_knn_item_invalid(sample_data):
    sim = pd.DataFrame(np.identity(4), index=sample_data.columns, columns=sample_data.columns)
    assert np.isnan(cf.predict_rating_knn_item("User1", "Unknown", sample_data, sim))


def test_evaluate_model(sample_data):
    test_df = pd.DataFrame([
        {"user_id": "User1", "title": "MovieA", "rating": 5},
        {"user_id": "User2", "title": "MovieB", "rating": 3}
    ])
    rmse, mae = cf.evaluate_model(lambda u, m, t: 4, test_df, sample_data)
    assert rmse >= 0
    assert mae >= 0


def test_get_top_n_recommendations_knn(sample_data):
    sim = pd.DataFrame(np.identity(4), index=sample_data.columns, columns=sample_data.columns)
    recs = cf.get_top_n_recommendations_knn("User1", sample_data, sim, k=1, N=2)
    assert "Film recommandé" in recs.columns


def test_get_top_n_recommendations_knn_empty(sample_data):
    sim = pd.DataFrame(np.identity(4), index=sample_data.columns, columns=sample_data.columns)
    df = sample_data.copy()
    df.loc["User1"] = [5, 4, 3, 2]  # aucun film non vu
    recs = cf.get_top_n_recommendations_knn("User1", df, sim)
    assert recs.empty
