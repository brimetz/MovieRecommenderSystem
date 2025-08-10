import pandas as pd
import pytest
from src.collaborative_filtering import get_similar_movies_pearson


@pytest.fixture
def sample_user_movie_matrix():
    # Matrice utilisateurs x films avec valeurs NaN
    return pd.DataFrame(
        {
            "Film A": [5, 4, None, 2],
            "Film B": [4, 5, 3, None],
            "Film C": [None, 2, 4, 3],
            "Film D": [1, None, 5, 4],
        },
        index=[1, 2, 3, 4],
    )


def test_valid_case(sample_user_movie_matrix):
    result = get_similar_movies_pearson(
        target_title="Film A",
        user_movie_matrix=sample_user_movie_matrix,
        min_common_ratings=1,
        top_n=2,
    )
    assert not result.empty
    assert "title" in result.columns
    assert "correlation" in result.columns
    assert "num_common_ratings" in result.columns
    # Vérifie qu'on ne recommande pas le film lui-même
    assert "Film A" not in result["title"].values
    # Vérifie que la taille max est respectée
    assert len(result) <= 2


def test_movie_not_in_matrix(sample_user_movie_matrix):
    result = get_similar_movies_pearson(
        target_title="Film Z", user_movie_matrix=sample_user_movie_matrix
    )
    assert result.empty
    assert list(result.columns) == ["title", "correlation", "num_common_ratings"]


def test_no_common_ratings(sample_user_movie_matrix):
    # Met tout à NaN sauf la cible pour casser la corrélation
    matrix = sample_user_movie_matrix.copy()
    matrix.loc[:, ["Film B", "Film C", "Film D"]] = None
    result = get_similar_movies_pearson(
        target_title="Film A", user_movie_matrix=matrix, min_common_ratings=2
    )
    assert result.empty


def test_self_removed_from_results(sample_user_movie_matrix):
    # Vérifie que le film cible est bien supprimé du DataFrame final
    result = get_similar_movies_pearson(
        target_title="Film A",
        user_movie_matrix=sample_user_movie_matrix,
        min_common_ratings=1,
    )
    assert "Film A" not in result["title"].values


def test_sorted_by_correlation(sample_user_movie_matrix):
    result = get_similar_movies_pearson(
        target_title="Film A",
        user_movie_matrix=sample_user_movie_matrix,
        min_common_ratings=1,
    )
    assert all(
        result["correlation"].iloc[i] >= result["correlation"].iloc[i + 1]
        for i in range(len(result) - 1)
    )
