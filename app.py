import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pickle.load(open('movies.pkl', 'rb'))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

similarity = cosine_similarity(vectors)


def recommend(movie):
    if movie in df['title'].values:
        movie_ind = df[df['title'] == movie].index[0]
    else:
        return ["Movie not found"]
    
    distances = similarity[movie_ind]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    return [df.iloc[i[0]].title for i in movies_list]


st.title("Movie Recommendation System 🎬")

movie_list = df['title'].values
selected_movie = st.selectbox('Select a movie', movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    
    for movie in recommendations:
        st.write(movie)