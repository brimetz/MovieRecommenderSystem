import pandas as pd
import numpy as np
import pytest
import src.collaborative_filtering as cf


@pytest.fixture
def sample_data():
    # Exemple simple : 4 utilisateurs x 4 films
    data = {
        1: [5, 4, np.nan, 1],
        2: [4, np.nan, 2, 1],
        3: [np.nan, 3, 5, 1],
        4: [4, 3, 5, 1],
    }
    df = pd.DataFrame(data, index=[1, 2, 3, 4])
    return df


@pytest.fixture
def sample_movies():
    # Mapping simple titres -> ids (1..4)
    return pd.DataFrame(
        {"movie_id": [1, 2, 3, 4], "title": ["MovieA", "MovieB", "MovieC", "MovieD"]}
    )


def test_get_similar_movies_pearson_valid(sample_data):
    result = cf.get_similar_movies_pearson(
        1, user_movie_matrix=sample_data, min_common_ratings=1, top_n=2
    )
    assert "movie_id" in result.columns
    assert not result.empty


def test_get_similar_movies_pearson_invalid(sample_data):
    result = cf.get_similar_movies_pearson(6, sample_data)
    assert result.empty


def test_get_similar_movies_cosine_valid(sample_data, sample_movies):
    result = cf.get_similar_movies_cosine(1, sample_data, min_common_ratings=1, top_n=2)
    assert "movie_id" in result.columns
    assert not result.empty


def test_get_similar_movies_cosine_no_ratings(sample_data, sample_movies):
    df = sample_data.copy()
    df["EmptyMovie"] = [np.nan, np.nan, np.nan, np.nan]
    sample_movies = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4, 5],
            "title": ["MovieA", "MovieB", "MovieC", "MovieD", "EmptyMovie"],
        }
    )
    result = cf.get_similar_movies_cosine(5, df, sample_movies)

    assert result.empty


def test_get_similar_movies_cosine_invalid(sample_data, sample_movies):
    result = cf.get_similar_movies_cosine(10, sample_data, sample_movies)
    assert result.empty


def test_predict_rating(sample_data):
    rating = cf.predict_rating(1, 1, sample_data)
    assert np.isfinite(rating) or np.isnan(rating)


def test_predict_rating_fast(sample_data):
    sim = pd.DataFrame(
        np.identity(4), index=sample_data.columns, columns=sample_data.columns
    )
    rating = cf.predict_rating_fast(1, 1, sample_data, sim)
    assert np.isfinite(rating) or np.isnan(rating)


def test_predict_mean_rating(sample_data):
    mean_rating = cf.predict_mean_rating(1, sample_data)
    assert np.isfinite(mean_rating)


def test_predict_mean_rating_invalid(sample_data):
    assert np.isnan(cf.predict_mean_rating(99, sample_data))


def test_predict_random_rating():
    val = cf.predict_random_rating(1, 5)
    assert 1 <= val <= 5


def test_predict_rating_knn_item(sample_data):
    sim = pd.DataFrame(
        np.identity(4), index=sample_data.columns, columns=sample_data.columns
    )
    rating = cf.predict_rating_knn_item(1, 1, sample_data, sim)
    assert np.isfinite(rating) or np.isnan(rating)


def test_predict_rating_knn_item_invalid(sample_data):
    sim = pd.DataFrame(
        np.identity(4), index=sample_data.columns, columns=sample_data.columns
    )
    assert np.isnan(cf.predict_rating_knn_item(1, 99, sample_data, sim))


def test_evaluate_model(sample_data):
    test_df = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 1, "rating": 5},
            {"user_id": 2, "movie_id": 2, "rating": 3},
        ]
    )
    rmse, mae = cf.evaluate_model(lambda u, m, t: 4, test_df, sample_data)
    assert rmse >= 0
    assert mae >= 0


def test_get_top_n_recommendations_knn(sample_data, sample_movies):
    sim = pd.DataFrame(
        np.identity(4), index=sample_data.columns, columns=sample_data.columns
    )
    recs = cf.get_top_n_recommendations_knn(1, sample_data, sim, k=1, N=2)
    assert "movie_id" in recs.columns


def test_get_top_n_recommendations_knn_empty(sample_data, sample_movies):
    sim = pd.DataFrame(
        np.identity(4), index=sample_data.columns, columns=sample_data.columns
    )
    df = sample_data.copy()
    df.loc[1] = [5, 4, 3, 2]  # aucun film non vu
    recs = cf.get_top_n_recommendations_knn(1, df, sim, sample_movies)
    assert recs.empty
