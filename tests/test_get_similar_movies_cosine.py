import pandas as pd
import pytest
from src.collaborative_filtering import get_similar_movies_cosine


@pytest.fixture
def sample_user_movie_matrix():
    return pd.DataFrame(
        {
            "Film A": [5, 4, None, 2],
            "Film B": [4, 5, 3, None],
            "Film C": [None, 2, 4, 3],
            "Film D": [1, None, 5, 4],
        },
        index=[1, 2, 3, 4],
    )


@pytest.fixture
def sample_movies():
    # Mapping simple titres -> ids (1..4)
    return pd.DataFrame(
        {"movie_id": [1, 2, 3, 4], "title": ["Film A", "Film B", "Film C", "Film D"]}
    )


def test_valid_case(sample_user_movie_matrix, sample_movies):
    result = get_similar_movies_cosine(
        target_title="Film A",
        user_movie_matrix=sample_user_movie_matrix,
        movies=sample_movies,
        min_common_ratings=1,
        top_n=2,
    )
    assert not result.empty
    assert "title" in result.columns
    assert "similarity" in result.columns
    assert "num_common_ratings" in result.columns
    assert "Film A" not in result["title"].values
    assert len(result) <= 2


def test_movie_not_in_matrix(sample_user_movie_matrix, sample_movies):
    result = get_similar_movies_cosine(
        target_title="Film Z",
        user_movie_matrix=sample_user_movie_matrix,
        movies=sample_movies,
    )
    assert result.empty
    assert list(result.columns) == ["title", "similarity", "num_common_ratings"]


def test_no_ratings_for_target(sample_user_movie_matrix, sample_movies):
    matrix = sample_user_movie_matrix.copy()
    matrix["Film A"] = None
    result = get_similar_movies_cosine(
        target_title="Film A", user_movie_matrix=matrix, movies=sample_movies
    )
    assert result.empty


def test_no_common_ratings(sample_user_movie_matrix, sample_movies):
    matrix = sample_user_movie_matrix.copy()
    matrix.loc[:, ["Film B", "Film C", "Film D"]] = None
    result = get_similar_movies_cosine(
        target_title="Film A",
        user_movie_matrix=matrix,
        movies=sample_movies,
        min_common_ratings=2,
    )
    assert result.empty


def test_self_removed_from_results(sample_user_movie_matrix, sample_movies):
    result = get_similar_movies_cosine(
        target_title="Film A",
        user_movie_matrix=sample_user_movie_matrix,
        movies=sample_movies,
        min_common_ratings=1,
    )
    assert "Film A" not in result["title"].values


def test_sorted_by_similarity(sample_user_movie_matrix, sample_movies):
    result = get_similar_movies_cosine(
        target_title="Film A",
        user_movie_matrix=sample_user_movie_matrix,
        movies=sample_movies,
        min_common_ratings=1,
    )
    assert all(
        result["similarity"].iloc[i] >= result["similarity"].iloc[i + 1]
        for i in range(len(result) - 1)
    )
