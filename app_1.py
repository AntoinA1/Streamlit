
import streamlit as st
import pandas as pd


# Titre de l'application
st.title("Recommender System, Group 6")

# Section pour l'insertion du nom d'utilisateur
username = st.text_input("User Name: ")

# Recherche de films
st.header("Recherche de films")
movie_query = st.text_input("Entrez le nom d'un film")

if movie_query:
    movie_choices = movies_df['title'].tolist()
    search_results = search_movie(movie_query, movie_choices)
    
    st.write("Résultats de la recherche :")
    for movie in search_results:
        st.write(movie)

    # Permettre à l'utilisateur de noter les films trouvés
    ratings = {}
    for movie in search_results:
        rating = st.slider(f"Notez le film {movie}", 0, 5, 3)
        ratings[movie] = rating

    # Afficher les notations pour vérification
    st.write("Vos notations :")
    st.write(ratings)
    
    # Placeholder pour les recommandations
    if st.button("Obtenir des recommandations"):
        st.subheader("Recommandations basées sur vos notations")
        # Ici, vous ajouterez le code pour générer et afficher les recommandations
        st.write("Recommandations à venir...")

# Afficher les notations pour vérification
st.write("Vos notations :")
st.write(ratings)
