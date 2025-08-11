import pandas as pd
import numpy as np
from src.content_based_reco import (
    load_content_based_data,
    get_nlp_content_based_recommendations,
    combine_text_features,
    merge_movies_overviews,
)


def test_load_content_based_data(monkeypatch):
    # Mock de pd.read_csv pour éviter de lire un vrai fichier
    def mock_read_csv(*args, **kwargs):
        cols = [
            "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        data = [[1, "Movie1"] + [""] * 3 + [0] * 19]
        return pd.DataFrame(data, columns=cols)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    df = load_content_based_data("fake_path")
    assert isinstance(df, pd.DataFrame)
    assert "movie_id" in df.columns
    assert "title" in df.columns
    assert "genres" in df.columns
    # genres devrait être chaîne de caractères même si tous 0
    assert isinstance(df.loc[0, "genres"], str)


def test_get_nlp_content_based_recommendations():
    # Jeu de données minimal
    df_movies = pd.DataFrame({
        "title": ["MovieA", "MovieB", "MovieC"],
        "title_norm": ["moviea", "movieb", "moviec"],
    })
    # Matrice identité 3x3 (similarité parfaite avec soi-même)
    cosine_sim = np.identity(3)

    # Le film existe
    res = get_nlp_content_based_recommendations("MovieA", cosine_sim,
                                                df_movies, top_n=2)
    assert isinstance(res, pd.DataFrame)
    assert "title" in res.columns
    assert "Score de similarité" in res.columns
    # Ne doit pas contenir le film lui-même
    assert "MovieA" not in res["title"].values
    # Résultat limité à top_n
    assert len(res) <= 2

    # Le film n'existe pas -> df vide
    res_empty = get_nlp_content_based_recommendations("FilmInexistant",
                                                      cosine_sim, df_movies)
    assert res_empty.empty


def test_combine_text_features():
    row = {
        "title": "My Movie",
        "unknown": 0,
        "Action": 1,
        "Adventure": 0,
        "overview": "A nice story"
    }
    text = combine_text_features(row)
    assert isinstance(text, str)
    assert "My Movie" in text
    assert "Action" in text
    assert "A nice story" in text

    # Test sans résumé
    row_no_overview = {
        "title": "Movie2",
        "unknown": 0,
        "Action": 0,
        "Adventure": 1,
        "overview": ""
    }
    text2 = combine_text_features(row_no_overview)
    assert "Movie2" in text2
    assert "Adventure" in text2


def test_merge_movies_overviews():
    df_movies = pd.DataFrame({
        "movie_id": [1, 2],
        "title": ["MovieA", "MovieB"],
        "unknown": [0, 0],
        "Action": [1, 0],
        "Adventure": [0, 1],
    })
    df_overviews = pd.DataFrame({
        "title": ["MovieA", "MovieB"],
        "overview": ["Story A", None]
    })

    merged = merge_movies_overviews(df_movies, df_overviews)
    assert "text_features" in merged.columns
    # "overview" doit être string même si None initialement
    assert merged.loc[1, "overview"] == ""
    # text_features contient titre + genre + résumé si dispo
    assert "MovieA" in merged.loc[0, "text_features"]
    assert "Story A" in merged.loc[0, "text_features"]
