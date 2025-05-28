import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load data
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# TF-IDF vectorizer for content-based
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Build collaborative filtering model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, _ = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

# Helper functions
def get_content_recommendations(title, top_n=5):
    if title not in movies_df['title'].values:
        return pd.DataFrame()
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][['title', 'genres']]

def get_collab_recommendations(user_id, n=5):
    movie_ids = movies_df['movieId'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_preds = predictions[:n]
    top_movie_ids = [pred.iid for pred in top_preds]
    return movies_df[movies_df['movieId'].isin(top_movie_ids)][['title', 'genres']]

def get_similar_users(user_id, top_n=3):
    user_ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if user_id not in user_ratings_matrix.index:
        return []
    cosine_user_sim = cosine_similarity(user_ratings_matrix)
    user_idx = user_ratings_matrix.index.tolist().index(user_id)
    sim_scores = list(enumerate(cosine_user_sim[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [user_ratings_matrix.index[i[0]] for i in sim_scores]

def ai_driven_matchmaking_recommendations(user_id, n=5):
    similar_users = get_similar_users(user_id)
    similar_users_ratings = ratings_df[ratings_df['userId'].isin(similar_users)]
    top_movies = similar_users_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(n)
    return movies_df[movies_df['movieId'].isin(top_movies.index)][['title', 'genres']]

# Streamlit UI
st.title("🎬 AI-Powered Movie Recommender System")

st.sidebar.title("Select Options")

menu = st.sidebar.radio("Choose recommendation type:", 
                        ["Content-Based", "Collaborative Filtering", "AI Matchmaking"])

if menu == "Content-Based":
    movie_title = st.text_input("Enter a Movie Title (e.g., Toy Story (1995))")
    if st.button("Recommend"):
        results = get_content_recommendations(movie_title)
        if results.empty:
            st.warning("Movie not found.")
        else:
            st.subheader("📽️ Similar Movies:")
            st.dataframe(results)

elif menu == "Collaborative Filtering":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    if st.button("Recommend for User"):
        results = get_collab_recommendations(user_id)
        st.subheader("📊 Recommendations for User:")
        st.dataframe(results)

elif menu == "AI Matchmaking":
    user_id = st.number_input("Enter User ID for Matching", min_value=1, step=1, key="match_id")
    if st.button("Get Matchmaking Recommendations"):
        results = ai_driven_matchmaking_recommendations(user_id)
        st.subheader("🤝 Movies Liked by Similar Users:")
        st.dataframe(results)
