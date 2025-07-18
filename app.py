import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.reco import recommend_similar_movies, recommend_by_user_ratings
from src.data_preprocessing import load_data, GENRE_COLUMNS
from src.collaborative_filtering import get_similar_movies_cosine, get_top_n_recommendations

st.set_page_config(page_title="Reco de Films", page_icon="🎬")
st.markdown("""
# 🎬 Recommendation de Films
un projet Python avec **Streamlit** et **scikit-learn**
Recommande des films par **genres**, **notes d'utilisateurs** ou via **KNN**
""")

movie_genres, ratings, user_movie_matrix, movies = load_data()

# Calcul matrice de similarité (item-item) — tu peux le faire une fois au chargement
similarity_matrix = user_movie_matrix.corr(method="pearson", min_periods=5)

# === Interface Streamlit ===
col1, col2 = st.columns(2)
with col1:
    film_title = st.selectbox("🎞️ Film :", sorted(movies["title"].unique()))
with col2:
    mode = st.radio(
        "Mode :", 
        ["Par genres", "Par utilisateurs (Pearson)", "Par utilisateurs (Cosine)", "KNN item-item"])

user_id = None
if mode == "KNN item-item":
    user_id = st.number_input("Entrez votre ID utilisateur", min_value=int(ratings["user_id"].min()), max_value=int(ratings["user_id"].max()), value=int(ratings["user_id"].min()))

if st.button("Recommander des films similaires"):
    if mode == "Par genres":
        reco = recommend_similar_movies(film_title, movie_genres, GENRE_COLUMNS)
    elif mode == "Par utilisateurs (Pearson)":
        reco = recommend_by_user_ratings(film_title, user_movie_matrix, ratings, 50, 10)
    elif mode == "Par utilisateurs (Cosine)":
        reco = get_similar_movies_cosine(film_title, user_movie_matrix)
    elif mode == "KNN item-item":
        if user_id is None:
            st.warning("Veuillez entrer un ID utilisateur valide pour KNN.")
            reco = pd.DataFrame()
        else:
            reco = get_top_n_recommendations(user_id, user_movie_matrix, similarity_matrix)
    else:
        reco = pd.DataFrame() # Security if mode has an invalid value

    if not reco.empty:
        if mode == "KNN":
            st.markdown("### 🎯 Top 10 recommandations KNN")
            for i, (_, row) in enumerate(reco.iterrows(), start=1):
                note = row["predicted_rating"]  # ou autre nom exact dans reco
                if pd.notna(note):
                    stars = "⭐" * int(round(note))
                    note_str = f"{note:.2f}"
                else:
                    stars = "❓"
                    note_str = "Indisponible"
                st.markdown(f"**{i}. {row['title']}** — {note_str} {stars}")
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