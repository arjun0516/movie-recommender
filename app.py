import streamlit as st
import pickle
import pandas as pd

df = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    if movie in df['title'].values:
        movie_ind = df[df['title'] == movie].index[0]
    else:
        return ["Movie not found"]
    
    distances = similarity[movie_ind]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    return [df.iloc[i[0]].title for i in movies_list]

st.title("Movie Recommendation System 🎬")

movie_list=df['title'].values
selected_movie=st.selectbox('select a movie',movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    
    for movie in recommendations:
        st.write(movie)