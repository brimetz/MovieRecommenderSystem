# MovieRecommenderSystem

# 🎬 Movie Recommender
Web application build with **Python** and **Streamlit**.

## 🔍 Functionalities
- Recommendation based on movie **genre** (content-based)
- Recommendation based on **users notes** (collaboratif)
- Interactive interface with Streamlit

## 📁 Datas
Use the MovieLens 100k dataset ('u.data' and 'u.item')

## 🚀 Launch the app
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Author
Baptiste Rimetz

Recommandation Engine project base on the MovieLens dataset
Two approaches: Collaborative filtering (SVD) and content filtering (genres + cosine similarity)
