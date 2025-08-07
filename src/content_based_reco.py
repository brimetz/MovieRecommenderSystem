import pandas as pd
from pathlib import Path

def load_content_based_data(path="data/u.item"):
    movies_path = path

    # Colonnes selon MovieLens 100k
    movie_columns = [
        "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    print(movies_path)
    df_movies = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        header=None,
        names=movie_columns
    )

    # 🧠 Création d'une colonne "genres" textuelle
    def combine_genres(row):
        return " ".join([genre for genre in genre_cols if row[genre] == 1])
    
    genre_cols = movie_columns[5:]  # genres

    df_movies["genres"] = df_movies.apply(combine_genres, axis=1)

    #df_movies = pd.concat([df_movies[["movie_id", "title"]], df_movies[genre_cols]], axis=1)
    #df_movies.head()
    # Garder uniquement les colonnes utiles
    df_movies = df_movies[["movie_id", "title", "genres"]]
    return df_movies

def get_nlp_content_based_recommendations(movie_title, cosine_sim_matrix, df_movies, top_n=10):
    # Normaliser les titres pour la comparaison
    movie_title_norm = movie_title.strip().lower()
    df_movies['title_norm'] = df_movies['title'].str.strip().str.lower()
    
    # Vérifier que le film existe
    if movie_title_norm not in df_movies['title_norm'].values:
        print(f"Film '{movie_title}' non trouvé.")
        return pd.DataFrame()

    # Obtenir l'index du film demandé
    idx = df_movies[df_movies['title_norm'] == movie_title_norm].index[0]

    # Récupérer les scores de similarité
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Trier les films par score décroissant
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ignorer le premier (le film lui-même), prendre top_n suivants
    sim_scores = sim_scores[1:top_n+1]

    # Récupérer les indices des films similaires
    movie_indices = [i for i, _ in sim_scores]

    # Renvoyer le DataFrame avec les titres et les scores
    results = df_movies.iloc[movie_indices][["title"]].copy()
    results["Score de similarité"] = [round(score, 2) for _, score in sim_scores]

    return results.reset_index(drop=True)

# Colonnes selon MovieLens 100k
genre_cols = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def combine_text_features(row):
    genres = [genre for genre in genre_cols if row.get(genre, 0) == 1]

    # Base : titre + genres
    base = f"{row['title']} {' '.join(genres)}"

    # Ajout du résumé s’il existe
    if pd.notna(row['overview']) and row['overview'].strip() != "":
        return f"{base} {row['overview']}"
    else:
        return base
    
def merge_movies_overviews(df_movies, overviews_df):
    # Fusionner les deux datasets sur le titre
    df = pd.merge(df_movies, overviews_df, on="title", how="left")
    df["overview"] =  df["overview"].fillna("").astype(str)
    print(df.columns)
    df['text_features'] = df.apply(combine_text_features, axis=1)
    df[["text_features"]].head()
    return df