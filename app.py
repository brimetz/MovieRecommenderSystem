import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.reco import recommend_similar_movies, recommend_by_user_ratings

st.set_page_config(page_title="Reco de Films", page_icon="🎬")
st.markdown("""
# 🎬 Recommendation de Films
un projet Python avec **Streamlit** et **scikit-learn**
Recommande des films par **genres** ou par **notes d'utilisateurs**
""")


# === Données ===
# u.item contient les infos films + genres
movies = pd.read_csv(
    "data/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
           "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
           "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
           "Romance", "Sci-Fi", "Thriller", "War", "Western"]
)

# On garde les colonnes utiles
genre_columns = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                 "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                 "Romance", "Sci-Fi", "Thriller", "War", "Western"]

movie_genres = movies[["movie_id", "title"] + genre_columns].copy()

# === Chargement des notes des utilisateurs ===
ratings = pd.read_csv(
    "data/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)
# On fusionne les infos de film
ratings = ratings.merge(movies[["movie_id", "title"]], on="movie_id")
# Matrice : lignes = utilisateurs, colonnes = films, valeurs = notes
user_movie_matrix = ratings.pivot_table(index="user_id", columns="title", values="rating")

# === Interface Streamlit ===
col1, col2 = st.columns(2)
with col1:
    film_title = st.selectbox("🎞️ Film :", sorted(movies["title"].unique()))
with col2:
    mode = st.radio("Mode :", ["Par genres", "Par utilisateurs (collaboratif)"])

if st.button("Recommander des films similaires"):
    if mode == "Par genres":
        reco = recommend_similar_movies(film_title, movie_genres, genre_columns)
    else:
        reco = recommend_by_user_ratings(film_title, user_movie_matrix, ratings)

    if not reco.empty:
        st.write("Films recommandés :")
        # Arrondir la similarité/corrélation
        if "similarity" in reco.columns:
            reco["similarity"] = reco["similarity"].round(3)
        if "correlation" in reco.columns:
            reco["correlation"] = reco["correlation"].round(3)
        if "num_ratings" in reco.columns:
            reco = reco.rename(columns={"num_ratings": "Nombre de votes"})
        st.table(reco)
    else:
        st.warning("Aucune recommandation trouvée.")