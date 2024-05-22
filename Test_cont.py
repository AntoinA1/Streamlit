import pandas as pd
import streamlit as st
from loaders import load_items, load_ratings
from sklearn.metrics.pairwise import cosine_similarity

# Chargement des données
movies_df = load_items()
ratings_df = load_ratings(surprise_format=False)


# One-hot encoding des genres
genres_df = movies_df['genres'].str.get_dummies(sep='|')

print(genres_df.head())

# Calcul de la similarité cosinus entre les films
film_similarity = cosine_similarity(genres_df)

# Demander à l'utilisateur d'entrer l'ID d'un film qu'il a aimé
selected_movie_id = st.text_input("Entrez l'ID du film que vous avez aimé :")

# Vérifier si l'ID du film est valide
if selected_movie_id is not None and selected_movie_id.isdigit() and int(selected_movie_id) in movies_df.index:
    selected_movie_id = int(selected_movie_id)

    # Trouver les films similaires à celui que l'utilisateur a aimé
    similar_movies = pd.Series(film_similarity[selected_movie_id], index=movies_df.index).sort_values(ascending=False)[1:]

    # Afficher les films recommandés à l'utilisateur
    st.header("Films recommandés :")
    for i, (movie_id, similarity_score) in enumerate(similar_movies.head(5).items(), start=1):
        movie_title = movies_df.loc[movie_id, 'title']
        st.write(f"{i}. {movie_title} (Similarité : {similarity_score:.2f})")
else:
    st.error("Veuillez entrer un ID de film valide.")



