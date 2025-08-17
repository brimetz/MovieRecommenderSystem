import pandas as pd


# Columnns of genres used
GENRE_COLUMNS: list[str] = [
    "unkown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def load_content_based_data(path: str = "data/u.item") -> pd.DataFrame:
    movies_path = path

    # Columns of MovieLens 100k
    movie_columns: list[str] = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    df_movies: pd.DataFrame = pd.read_csv(
        movies_path, sep="|", encoding="latin-1", header=None, names=movie_columns
    )

    # Create a text column "genres"
    def combine_genres(row: pd.Series) -> str:
        return " ".join([genre for genre in genre_cols if row[genre] == 1])

    genre_cols: list[str] = movie_columns[5:]  # genres

    df_movies["genres"] = df_movies.apply(combine_genres, axis=1)

    # Keep only usefull columns
    df_movies = df_movies[["movie_id", "title", "genres"]]
    return df_movies


def get_nlp_content_based_recommendations(
    movie_title: str, cosine_sim_matrix, df_movies: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    # Normalize title for comparison
    movie_title_norm: str = movie_title.strip().lower()
    df_movies["title_norm"] = df_movies["title"].str.strip().str.lower()

    # Verify that movie_title exist
    if movie_title_norm not in df_movies["title_norm"].values:
        print(f"Movie '{movie_title}' not found.")
        return pd.DataFrame()

    # Get index of movie_title
    idx: int = df_movies[df_movies["title_norm"] == movie_title_norm].index[0]

    # Get similarity scores
    sim_scores: list = list(enumerate(cosine_sim_matrix[idx]))

    # Sorting movies
    sim_scores: list = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ignore first because it is the chosen movie
    sim_scores: list = sim_scores[1 : top_n + 1]

    # Get indices of similar movies
    movie_indices: list[int] = [i for i, _ in sim_scores]

    # return Dataframe with title and similarity score
    results: pd.DataFrame = df_movies.iloc[movie_indices][["title"]].copy()
    results["Score de similarité"] = [round(score, 2) for _, score in sim_scores]

    return results.reset_index(drop=True)


def combine_text_features(row: pd.Series) -> str:
    genres: list = [genre for genre in GENRE_COLUMNS if row.get(genre, 0) == 1]

    # Base : title + genres
    base: str = f"{row['title']} {' '.join(genres)}"

    # Add an overview if it exist
    if pd.notna(row["overview"]) and row["overview"].strip() != "":
        return f"{base} {row['overview']}"
    else:
        return base


def merge_movies_overviews(df_movies, overviews_df) -> pd.DataFrame:
    # Megre both DataFrame by the title
    df: pd.DataFrame = pd.merge(df_movies, overviews_df, on="title", how="left")
    df["overview"] = df["overview"].fillna("").astype(str)
    df["text_features"] = df.apply(combine_text_features, axis=1)
    df[["text_features"]].head()
    return df
