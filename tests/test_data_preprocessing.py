import pandas as pd
import numpy as np
import pytest
import src.data_preprocessing as dp


@pytest.fixture
def sample_movies_df():
    data = {
        "movie_id": [1, 2],
        "title": ["Film A", "Film B"],
        "release_date": ["01-Jan-2000", "02-Feb-2001"],
        "video_release_date": [None, None],
        "IMDb_URL": ["urlA", "urlB"],
        "Action": [1, 0],
        "Comedy": [0, 1],
    }
    # Add all genre columns with default 0 except "Action" and "Comedy" above
    for col in dp.GENRE_COLUMNS:
        if col not in data:
            data[col] = [0, 0]
    return pd.DataFrame(data)


@pytest.fixture
def sample_ratings_df():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 1],
            "movie_id": [1, 1, 2],
            "rating": [5, 3, 4],
            "timestamp": [111111, 111112, 111113],
        }
    )


def test_get_genre_matrix(sample_movies_df):
    genre_matrix = dp.get_genre_matrix(sample_movies_df)
    expected_cols = ["movie_id", "title"] + dp.GENRE_COLUMNS
    assert list(genre_matrix.columns) == expected_cols
    assert genre_matrix.shape[0] == sample_movies_df.shape[0]


def test_merge_ratings_with_titles(sample_ratings_df, sample_movies_df):
    merged = dp.merge_ratings_with_titles(sample_ratings_df, sample_movies_df)
    assert "title" in merged.columns
    assert set(merged["title"].unique()) == {"Film A", "Film B"}


def test_get_user_movie_matrix(sample_ratings_df, sample_movies_df):
    merged = dp.merge_ratings_with_titles(sample_ratings_df, sample_movies_df)
    user_movie_matrix = dp.get_user_movie_matrix(merged)
    assert isinstance(user_movie_matrix, pd.DataFrame)
    assert 1 in user_movie_matrix.index
    assert "Film A" in user_movie_matrix.columns


def test_compute_density_and_sparsity():
    df = pd.DataFrame(
        {
            "Film A": [5, None],
            "Film B": [None, 3],
        },
        index=[1, 2],
    )
    density = dp.compute_density(df)
    sparsity = dp.compute_sparsity(df)
    assert 0 <= density <= 1
    assert np.isclose(density + sparsity, 1)


def test_compute_item_similarity_matrix(sample_ratings_df, sample_movies_df):
    merged = dp.merge_ratings_with_titles(sample_ratings_df, sample_movies_df)
    user_movie_matrix = dp.get_user_movie_matrix(merged)
    sim_matrix = dp.compute_item_similarity_matrix(user_movie_matrix)
    assert isinstance(sim_matrix, pd.DataFrame)
    assert sim_matrix.shape[0] == sim_matrix.shape[1]
    assert all(sim_matrix.columns == sim_matrix.index)


@pytest.mark.parametrize(
    "load_movies_mock, load_ratings_mock",
    [
        (pd.DataFrame(), pd.DataFrame()),
    ],
)
def test_load_data_calls(mocker, load_movies_mock, load_ratings_mock):
    m1 = mocker.patch(
        "src.data_preprocessing.load_movies", return_value=load_movies_mock
    )
    m2 = mocker.patch(
        "src.data_preprocessing.load_ratings", return_value=load_ratings_mock
    )
    m3 = mocker.patch(
        "src.data_preprocessing.get_genre_matrix", return_value=pd.DataFrame()
    )
    m4 = mocker.patch(
        "src.data_preprocessing.merge_ratings_with_titles", return_value=pd.DataFrame()
    )
    m5 = mocker.patch(
        "src.data_preprocessing.get_user_movie_matrix", return_value=pd.DataFrame()
    )

    movie_genres, ratings, user_movie_matrix, movies = dp.load_data()

    m1.assert_called_once()
    m2.assert_called_once()
    m3.assert_called_once()
    m4.assert_called_once()
    m5.assert_called_once()
