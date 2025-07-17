import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.reco import recommend_similar_movies, recommend_by_user_ratings
from src.data_preprocessing import load_data, GENRE_COLUMNS
from src.collaborative_filtering import get_similar_movies_cosine

st.set_page_config(page_title="Reco de Films", page_icon="🎬")
st.markdown("""
# 🎬 Recommendation de Films
un projet Python avec **Streamlit** et **scikit-learn**
Recommande des films par **genres** ou par **notes d'utilisateurs**
""")

movie_genres, ratings, user_movie_matrix, movies = load_data()

# === Interface Streamlit ===
col1, col2 = st.columns(2)
with col1:
    film_title = st.selectbox("🎞️ Film :", sorted(movies["title"].unique()))
with col2:
    mode = st.radio(
        "Mode :", 
        ["Par genres", "Par utilisateurs (Pearson)", "Par utilisateurs (Cosine)"])

if st.button("Recommander des films similaires"):
    if mode == "Par genres":
        reco = recommend_similar_movies(film_title, movie_genres, GENRE_COLUMNS)
    elif mode == "Par utilisateurs (Pearson)":
        reco = recommend_by_user_ratings(film_title, user_movie_matrix, ratings, 50, 10)
    elif mode == "Par utilisateurs (Cosine)":
        reco = get_similar_movies_cosine(film_title, user_movie_matrix)
    else:
        reco = pd.DataFrame() # Security if mode has an invalid value

    if not reco.empty:
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