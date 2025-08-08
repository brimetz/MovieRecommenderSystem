# MovieRecommenderSystem

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.0-orange)
![Build Status](https://img.shields.io/github/actions/workflow/status/<brimetz>/<MovieRecommenderSystem>/pytest.yml?branch=main)
![Coverage](https://img.shields.io/codecov/c/github/<brimetz>/<MovieRecommenderSystem>?branch=main)
![Lint](https://github.com/brimetz/MovieRecommenderSystem/actions/workflows/lint.yml/badge.svg)

# 🎬 Movie Recommender
A **Streamlit** web application built in **Python** for movie recommendation.  
This project explores both **content-based filtering** (genre similarity) and **collaborative filtering** (user ratings) techniques, leveraging popular algorithms such as Pearson correlation and Cosine similarity.  

Designed to be a hands-on learning experience, it demonstrates key concepts in recommendation systems and provides an intuitive interface for users to discover movies tailored to their preferences.

## 🔍 Functionalities
- Recommendation based on movie **genre** (content-based)
- Recommendation based on **users notes** (collaborative) with:
    - **Pearson** correlation
    - **Cosine** similarity
- Interactive interface with Streamlit

## 📁 Datas
Use the MovieLens 100k dataset ('u.data' and 'u.item')
https://grouplens.org/datasets/movielens/100k/

## 🚀 Launch the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
Or access the live app here:
```https://brz-movie-recommender.streamlit.app/```

## Run Tests
Run the automated tests using:
python -m pytest tests/

### Author
Baptiste Rimetz