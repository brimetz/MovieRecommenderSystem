import streamlit as st
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.reco import recommend_similar_movies, recommend_by_user_ratings
from src.data_preprocessing import (
    load_data,
    GENRE_COLUMNS,
    compute_item_similarity_matrix,
)
from src.collaborative_filtering import (
    get_similar_movies_cosine,
    get_top_n_recommendations_knn,
)
from src.svd_recommender import train_svd_model, get_top_n_recommendations_svd
from src.content_based_reco import (
    get_nlp_content_based_recommendations,
    merge_movies_overviews,
)

from src.sql_utils import (
    get_movies,
    get_ratings,
    get_user_movie_matrix,
    load_content_based_data_sql,
)


st.set_page_config(page_title="Reco de Films", page_icon="🎬")
st.markdown(
    """
# 🎬 Recommendation de Films
un projet Python avec **Streamlit** et **scikit-learn**
Recommande des films par **genres**, **notes d'utilisateurs** ou via **KNN**
"""
)

movie_genres, ratings, user_movie_matrix, movies = load_data()

movies: pd.DataFrame = get_movies()
ratings: pd.DataFrame = get_ratings()
user_movie_matrix: pd.DataFrame = get_user_movie_matrix()

# Calcul matrice de similarité (item-item) —
# can be done after the loading
similarity_matrix: pd.DataFrame = compute_item_similarity_matrix(user_movie_matrix)

# train the svd model
svd_model = train_svd_model(ratings)


# --- Streamlit Interface ---
st.sidebar.markdown("### 🔧 Paramètres de la recommandation")

k: int = st.sidebar.slider(
    "Nombre de voisins (k for KNN)", min_value=1, max_value=50, value=5, step=1
)
N: int = st.sidebar.slider(
    "Nombre de recommandations", min_value=1, max_value=20, value=10, step=1
)


# Charger les fichiers nécessaires
df_nlp_movies: pd.DataFrame = load_content_based_data_sql()
print(df_nlp_movies.columns)
df_nlp_overviews: pd.DataFrame = pd.read_csv("data/movie_overviews_sample.csv")

# Fusionner les deux datasets sur le titre
df_nlp: pd.DataFrame = merge_movies_overviews(df_nlp_movies, df_nlp_overviews)
print(df_nlp.columns)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df_nlp["genres"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# === Interface Streamlit ===
col1, col2 = st.columns(2)
with col1:
    film_title = st.selectbox("🎞️ Film :", sorted(movies["title"].unique()))
    # Convertir directement en movie_id
    film_row = movies[movies["title"] == film_title]
    movie_id = int(film_row["movie_id"].iloc[0]) if not film_row.empty else None
with col2:
    mode = st.radio(
        "Mode :",
        [
            "Par genres",
            "Par utilisateurs (Pearson)",
            "Par utilisateurs (Cosine)",
            "KNN item-item",
            "SVD",
            "Content-Based (NLP)",
        ],
    )

user_id = None
if mode == "KNN item-item" or mode == "SVD":
    umin, umax = int(ratings["user_id"].min()), int(ratings["user_id"].max())
    user_id = st.number_input(
        "Entrez votre ID utilisateur", min_value=umin, max_value=umax, value=umin
    )

if st.button("Recommander des films similaires"):
    if mode == "Content-Based (NLP)":
        reco = get_nlp_content_based_recommendations(movie_id, cosine_sim, df_nlp, N)
    elif mode == "Par genres":
        reco = recommend_similar_movies(movie_id, movie_genres, GENRE_COLUMNS, N)
    elif mode == "Par utilisateurs (Pearson)":
        reco = recommend_by_user_ratings(
            movie_id, user_movie_matrix, ratings, movies, 50, N
        )
    elif mode == "Par utilisateurs (Cosine)":
        reco = get_similar_movies_cosine(movie_id, user_movie_matrix, 5, N)
    elif mode == "KNN item-item":
        if user_id is None:
            st.warning("Veuillez entrer un ID utilisateur valide pour KNN.")
            reco = pd.DataFrame()
        else:
            reco = get_top_n_recommendations_knn(
                user_id, user_movie_matrix, similarity_matrix, k, N
            )
    elif mode == "SVD":
        reco = get_top_n_recommendations_svd(user_id, ratings, svd_model, N)
    else:
        reco = pd.DataFrame()  # Security if mode has an invalid value

    if not reco.empty:
        reco = reco.merge(movies[["movie_id", "title"]], on="movie_id", how="left")
        reco.rename(columns={"title": "Film recommandé"}, inplace=True)

        # Arrondir la similarité/corrélation
        if "similarity" in reco.columns:
            reco["similarity"] = reco["similarity"].round(3)
        if "correlation" in reco.columns:
            reco["correlation"] = reco["correlation"].round(3)
        if "num_common_ratings" in reco.columns:
            reco = reco.rename(columns={"num_common_ratings": "Votes communs"})
        st.write("Films recommandés :")
        st.table(reco)
    else:
        st.warning("Aucune recommandation trouvée.")
