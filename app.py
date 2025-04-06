import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Preprocessing
movies['overview'] = movies['overview'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Function to filter movie titles based on input
def autocomplete(query, movie_list):
    results = [movie for movie in movie_list if query.lower() in movie.lower()]
    return results

# Streamlit App
st.title("🎬 Movie Recommender")

# Create a list of movie titles
movie_titles = movies['title'].tolist()

# Display text input box with autocomplete behavior
movie_name = st.text_input("Enter a movie name:")

if movie_name:
    # Filter movie titles based on the input
    suggestions = autocomplete(movie_name, movie_titles)

    # If suggestions are available, display a dropdown to select a movie
    if suggestions:
        selected_movie = st.selectbox("Select a movie:", suggestions)
        if selected_movie:
            results = get_recommendations(selected_movie)
            st.write("Recommendations:")
            for r in results:
                st.write(f"- {r}")
    else:
        st.write("No matching movies found.")
