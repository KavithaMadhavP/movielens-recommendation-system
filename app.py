import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation Engine")
st.write("User-Based Collaborative Filtering with **Cold-Start Handling**")

# -------------------------------
# Load MovieLens Data
# -------------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "data/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"]
    )

    return ratings, movies

ratings_df, movies_df = load_data()

# -------------------------------
# User-Movie Matrix
# -------------------------------
user_movie_matrix = ratings_df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# -------------------------------
# Popular movies (for cold-start)
# -------------------------------
popular_movies = (
    ratings_df.groupby("movie_id")
    .size()
    .sort_values(ascending=False)
    .head(5)
    .index
)

popular_movie_titles = movies_df[
    movies_df["movie_id"].isin(popular_movies)
]

# -------------------------------
# Cold-Start UI
# -------------------------------
st.sidebar.subheader("🆕 Cold-Start (New User)")
is_new_user = st.sidebar.checkbox("I am a new user")

new_user_ratings = {}

if is_new_user:
    st.subheader("⭐ Rate a few popular movies")

    for _, row in popular_movie_titles.iterrows():
        rating = st.slider(
            row["title"],
            0, 5, 0,
            key=int(row["movie_id"])
        )
        if rating > 0:
            new_user_ratings[row["movie_id"]] = rating

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_movies(user_id, matrix, similarity_df, top_n):
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]

    scores = np.zeros(matrix.shape[1])
    similarity_sum = 0

    for sim_user, similarity in similar_users.head(10).items():
        scores += similarity * matrix.loc[sim_user].values
        similarity_sum += similarity

    scores = scores / similarity_sum
    rated = matrix.loc[user_id].values > 0
    scores[rated] = 0

    top_indices = np.argsort(scores)[::-1][:top_n]
    movie_ids = matrix.columns[top_indices]

    return movies_df[movies_df["movie_id"].isin(movie_ids)]

# -------------------------------
# Existing user selection
# -------------------------------
if not is_new_user:
    selected_user = st.selectbox(
        "👤 Select User",
        user_movie_matrix.index
    )

top_n = st.slider("🔢 Number of Recommendations", 1, 10, 5)

# -------------------------------
# Generate Recommendations
# -------------------------------
if st.button("🚀 Recommend"):

    matrix = user_movie_matrix.copy()

    if is_new_user:
        if len(new_user_ratings) < 2:
            st.warning("Please rate at least 2 movies")
            st.stop()

        new_user_id = matrix.index.max() + 1
        matrix.loc[new_user_id] = 0

        for movie_id, rating in new_user_ratings.items():
            matrix.loc[new_user_id, movie_id] = rating

        selected_user = new_user_id

    similarity = cosine_similarity(matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=matrix.index,
        columns=matrix.index
    )

    recommendations = recommend_movies(
        selected_user, matrix, similarity_df, top_n
    )

    st.subheader("✅ Recommended Movies")
    for _, row in recommendations.iterrows():
        st.write(f"🎥 **{row['title']}**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built by Kavitha P. | MCA AI & ML")
