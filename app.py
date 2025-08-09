import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.reco import recommend_similar_movies, recommend_by_user_ratings
from src.data_preprocessing import (
                load_data,
                GENRE_COLUMNS,
                compute_item_similarity_matrix)
from src.collaborative_filtering import (
    get_similar_movies_cosine,
    get_top_n_recommendations_knn)
from src.svd_recommender import (
    train_svd_model,
    get_top_n_recommendations_svd)
from src.content_based_reco import (
    get_nlp_content_based_recommendations,
    merge_movies_overviews,
    load_content_based_data)


st.set_page_config(page_title="Reco de Films", page_icon="🎬")
st.markdown("""
# 🎬 Recommendation de Films
un projet Python avec **Streamlit** et **scikit-learn**
Recommande des films par **genres**, **notes d'utilisateurs** ou via **KNN**
""")

movie_genres, ratings, user_movie_matrix, movies = load_data()


# Calcul matrice de similarité (item-item) —
# can be done after the loading
similarity_matrix = compute_item_similarity_matrix(user_movie_matrix)

# train the svd model
svd_model = train_svd_model(ratings)

st.sidebar.markdown("### 🔧 Paramètres de la recommandation")

k = st.sidebar.slider("Nombre de voisins (k for KNN)", min_value=1,
                      max_value=50, value=5, step=1)
N = st.sidebar.slider("Nombre de recommandations", min_value=1,
                      max_value=20, value=10, step=1)


# Charger les fichiers nécessaires
df_nlp_movies = load_content_based_data()
df_nlp_overviews = pd.read_csv("data/movie_overviews_sample.csv")
print(df_nlp_movies.columns)
print(df_nlp_overviews.columns)
# Fusionner les deux datasets sur le titre
df_nlp = merge_movies_overviews(df_nlp_movies, df_nlp_overviews)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df_nlp["text_features"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# === Interface Streamlit ===
col1, col2 = st.columns(2)
with col1:
    film_title = st.selectbox("🎞️ Film :", sorted(movies["title"].unique()))
with col2:
    mode = st.radio(
        "Mode :",
        ["Par genres", "Par utilisateurs (Pearson)",
         "Par utilisateurs (Cosine)", "KNN item-item",
         "SVD", "Content-Based (NLP)"])

user_id = None
if mode == "KNN item-item" or mode == "SVD":
    umin, umax = int(ratings["user_id"].min()), int(ratings["user_id"].max())
    user_id = st.number_input("Entrez votre ID utilisateur",
                              min_value=umin, max_value=umax, value=umin)

if st.button("Recommander des films similaires"):
    if mode == "Content-Based (NLP)":
        reco = get_nlp_content_based_recommendations(film_title,
                                                     cosine_sim, df_nlp)
    elif mode == "Par genres":
        reco = recommend_similar_movies(film_title,
                                        movie_genres, GENRE_COLUMNS, N)
    elif mode == "Par utilisateurs (Pearson)":
        reco = recommend_by_user_ratings(film_title,
                                         user_movie_matrix, ratings, 50, N)
    elif mode == "Par utilisateurs (Cosine)":
        reco = get_similar_movies_cosine(film_title, user_movie_matrix, 10, N)
    elif mode == "KNN item-item":
        if user_id is None:
            st.warning("Veuillez entrer un ID utilisateur valide pour KNN.")
            reco = pd.DataFrame()
        else:
            reco = get_top_n_recommendations_knn(user_id,
                                                 user_movie_matrix,
                                                 similarity_matrix, k, N)
    elif mode == "SVD":
        reco = get_top_n_recommendations_svd(user_id, ratings, svd_model, N)
    else:
        reco = pd.DataFrame()  # Security if mode has an invalid value

    if not reco.empty:
        if mode == "KNN item-item":
            st.markdown("### 🎯 Top 10 recommandations KNN")
            for i, (_, row) in enumerate(reco.iterrows(), start=1):
                note = row["Note prédite (estimation d'appréciation)"]
                if pd.notna(note):
                    stars = "⭐" * int(round(note))
                    note_str = f"{note:.2f}"
                else:
                    stars = "❓"
                    note_str = "Indisponible"
                movie_reco = row['Film recommandé']
                st.markdown(f"**{i}. {movie_reco}** — {note_str} {stars}")
        elif mode == "SVD":
            reco = reco.merge(movies[["movie_id", "title"]], on="movie_id")
            reco.rename(columns={"title": "Film recommandé"}, inplace=True)

            for i, row in reco.iterrows():
                note = row["predicted_rating"]
                stars = "⭐" * int(round(note))
                movie_reco = row['Film recommandé']
                st.markdown(f"**{i+1}. {movie_reco}** — {note}/5 {stars}")
        elif mode == "Content-Based (NLP)":
            st.subheader("Films similaires recommandés :")
            for i, row in reco.iterrows():
                title, sim = row['title'], row['Score de similarité']
                st.markdown(f"**{i+1}. {title}** — Similarité : {sim}")
        else:
            # Arrondir la similarité/corrélation
            if "similarity" in reco.columns:
                reco["similarity"] = reco["similarity"].round(3)
            if "correlation" in reco.columns:
                reco["correlation"] = reco["correlation"].round(3)
            if "num_common_ratings" in reco.columns:
                reco = reco.rename(columns={"num_common_ratings":
                                            "Votes communs"})
            st.write("Films recommandés :")
            st.table(reco)
    else:
        st.warning("Aucune recommandation trouvée.")
