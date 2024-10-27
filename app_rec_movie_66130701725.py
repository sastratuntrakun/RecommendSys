import streamlit as st
import pickle
import pandas as pd

# Load the saved model and data
with open('recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Title
st.title("Movie Recommendation System")

# Input user ID
user_id = st.number_input("Enter User ID:", min_value=1, value=1)

# Function to get recommendations
def get_recommendations(user_id):
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_recommendations = sorted_predictions[:10]

    recommendations = []
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        recommendations.append(f"{movie_title} (Estimated Rating: {recommendation.est})")
    return recommendations

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id)
    st.write("Top 10 Movie Recommendations:")
    for recommendation in recommendations:
        st.write(recommendation)
