# 1. Import Required Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import numpy as np

# 2. Load the Dataset
movies_df = pd.read_csv("movies.csv")  # Assume columns: movieId, title, genres
ratings_df = pd.read_csv("ratings.csv")  # Assume columns: userId, movieId, rating

# 3. Content-Based Filtering (TF-IDF on Genres)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on content similarity
def get_content_recommendations(title, top_n=5):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][['title', 'genres']]

# 4. Collaborative Filtering using SVD (Surprise Library)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

# Predict top movies for a user
def get_collab_recommendations(user_id, n=5):
    movie_ids = movies_df['movieId'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_preds = predictions[:n]
    top_movie_ids = [pred.iid for pred in top_preds]
    return movies_df[movies_df['movieId'].isin(top_movie_ids)][['title', 'genres']]

# 5. AI-Driven Matchmaking (Find users with similar taste)
def get_similar_users(user_id, top_n=3):
    user_ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    cosine_user_sim = cosine_similarity(user_ratings_matrix)
    user_idx = user_ratings_matrix.index.tolist().index(user_id)
    sim_scores = list(enumerate(cosine_user_sim[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    similar_user_ids = [user_ratings_matrix.index[i[0]] for i in sim_scores]
    return similar_user_ids

# 6. Integrated Recommendation: Recommend movies liked by similar users
def ai_driven_matchmaking_recommendations(user_id, n=5):
    similar_users = get_similar_users(user_id)
    similar_users_ratings = ratings_df[ratings_df['userId'].isin(similar_users)]
    top_movies = similar_users_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(n)
    return movies_df[movies_df['movieId'].isin(top_movies.index)][['title', 'genres']]

# Example Usage
print("📽 Content-based Recommendations:")
print(get_content_recommendations("Toy Story (1995)"))

print("\n📊 Collaborative Filtering Recommendations:")
print(get_collab_recommendations(user_id=1))

print("\n🤝 AI-Driven Matchmaking Recommendations:")
print(ai_driven_matchmaking_recommendations(user_id=1))
