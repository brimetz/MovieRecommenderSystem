import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.reco import recommend_similar_movies, recommend_by_user_ratings
from src.data_preprocessing import load_data, GENRE_COLUMNS, compute_item_similarity_matrix
from src.collaborative_filtering import get_similar_movies_cosine, get_top_n_recommendations_knn
from src.svd_recommender import train_svd_model, get_top_n_recommendations_svd

import streamlit as st

st.set_page_config(page_title="Reco de Films", page_icon="🎬")
st.markdown("""
# 🎬 Recommendation de Films
un projet Python avec **Streamlit** et **scikit-learn**
Recommande des films par **genres**, **notes d'utilisateurs** ou via **KNN**
""")

movie_genres, ratings, user_movie_matrix, movies = load_data()

# Calcul matrice de similarité (item-item) — tu peux le faire une fois au chargement
similarity_matrix = compute_item_similarity_matrix(user_movie_matrix)

# train the svd model
svd_model = train_svd_model(ratings)

st.sidebar.markdown("### 🔧 Paramètres de la recommandation")

k = st.sidebar.slider("Nombre de voisins (k for KNN)", min_value=1, max_value=50, value=5, step=1)
N = st.sidebar.slider("Nombre de recommandations", min_value=1, max_value=20, value=10, step=1)

# === Interface Streamlit ===
col1, col2 = st.columns(2)
with col1:
    film_title = st.selectbox("🎞️ Film :", sorted(movies["title"].unique()))
with col2:
    mode = st.radio(
        "Mode :", 
        ["Par genres", "Par utilisateurs (Pearson)", "Par utilisateurs (Cosine)", "KNN item-item", "SVD"])

user_id = None
if mode == "KNN item-item" or mode == "SVD":
    user_id = st.number_input("Entrez votre ID utilisateur", min_value=int(ratings["user_id"].min()), max_value=int(ratings["user_id"].max()), value=int(ratings["user_id"].min()))

if st.button("Recommander des films similaires"):
    if mode == "Par genres":
        reco = recommend_similar_movies(film_title, movie_genres, GENRE_COLUMNS, N)
    elif mode == "Par utilisateurs (Pearson)":
        reco = recommend_by_user_ratings(film_title, user_movie_matrix, ratings, 50, N)
    elif mode == "Par utilisateurs (Cosine)":
        reco = get_similar_movies_cosine(film_title, user_movie_matrix, 10, N)
    elif mode == "KNN item-item":
        if user_id is None:
            st.warning("Veuillez entrer un ID utilisateur valide pour KNN.")
            reco = pd.DataFrame()
        else:
            reco = get_top_n_recommendations_knn(user_id, user_movie_matrix, similarity_matrix, k, N)
    elif mode == "SVD":
        reco = get_top_n_recommendations_svd(user_id, ratings, svd_model, N)
    else:
        reco = pd.DataFrame() # Security if mode has an invalid value

    if not reco.empty:
        if mode == "KNN item-item":
            st.markdown("### 🎯 Top 10 recommandations KNN")
            for i, (_, row) in enumerate(reco.iterrows(), start=1):
                note = row["Note prédite (estimation d'appréciation)"]  # ou autre nom exact dans reco
                if pd.notna(note):
                    stars = "⭐" * int(round(note))
                    note_str = f"{note:.2f}"
                else:
                    stars = "❓"
                    note_str = "Indisponible"
                st.markdown(f"**{i}. {row['Film recommandé']}** — {note_str} {stars}")
        elif mode == "SVD":
            reco = reco.merge(movies[["movie_id", "title"]], on="movie_id")
            reco.rename(columns={"title": "Film recommandé"}, inplace=True)

            for i, row in reco.iterrows():
                note = row["predicted_rating"]
                stars = "⭐" * int(round(note))
                st.markdown(f"**{i+1}. {row['Film recommandé']}** — {note}/5 {stars}")
        else:
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