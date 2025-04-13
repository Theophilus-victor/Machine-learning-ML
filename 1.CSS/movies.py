import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    data = pd.merge(ratings, movies, on="movieId")
    return data

@st.cache_data
def compute_similarity(data):
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
    user_movie_matrix.fillna(0, inplace=True)
    similarity = cosine_similarity(user_movie_matrix.T)
    similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    return similarity_df

data = load_data()
similarity_df = compute_similarity(data)

st.title("🎬 Movie Recommender System")
st.write("Type part of a movie name to get suggestions and recommendations!")

movie_input = st.text_input("🔎 Search for a movie", placeholder="e.g., dark, ring, star wars...")

if movie_input:
    matched_movies = [title for title in similarity_df.columns if movie_input.lower() in title.lower()]

    if not matched_movies:
        st.warning("❌ No matching movies found.")
    else:
        st.subheader("🎯 Matching Movies")
        st.write(f"Found {len(matched_movies)} movies matching **'{movie_input}'**")

        selected_movie = st.selectbox("Select the movie you meant", matched_movies)

        if selected_movie:
            st.success(f"Showing recommendations based on: **{selected_movie}**")

            similar_movies = similarity_df[selected_movie].sort_values(ascending=False)[1:16]

            st.subheader("🍿 You might also enjoy:")
            for i, (title, score) in enumerate(similar_movies.items(), 1):
                st.write(f"{i}. {title} — Similarity: {score:.2f}")
