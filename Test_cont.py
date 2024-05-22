import pandas as pd
import streamlit as st
from loaders import load_items, load_ratings
from sklearn.metrics.pairwise import cosine_similarity

# Chargement des données
movies_df = load_items()
ratings_df = load_ratings(surprise_format=False)

# One-hot encoding des genres
genres_df = movies_df['genres'].str.get_dummies(sep='|')

# Calcul de la similarité cosinus entre les films
film_similarity = cosine_similarity(genres_df)

# Demander à l'utilisateur d'entrer l'ID d'un film qu'il a aimé
selected_movie_id = st.text_input("Entrez l'ID du film que vous avez aimé :")

# Vérifier si l'ID du film est valide
if selected_movie_id is not None and selected_movie_id.isdigit() and int(selected_movie_id) in movies_df.index:
    selected_movie_id = int(selected_movie_id)
    selected_movie_title = movies_df.loc[selected_movie_id, 'title']

    # Calculer le nombre de genres du film sélectionné
    num_genres_selected_movie = genres_df.loc[selected_movie_id].sum()

    # Trouver les films similaires à celui que l'utilisateur a aimé
    similar_movies = pd.Series(film_similarity[selected_movie_id], index=movies_df.index).sort_values(ascending=False)[1:]

    # Afficher le titre du film demandé
    st.write(f"Voici les films les plus similaires à '{selected_movie_title}' :")

    # Afficher les films recommandés à l'utilisateur
    num_recommendations = 0
    movie_titles = {}
    for i, (movie_id, similarity_score) in enumerate(similar_movies.items(), start=1):
        movie_title = movies_df.loc[movie_id, 'title']
        movie_titles[i] = (movie_id, movie_title)

        # Calculer le nombre de genres du film actuellement examiné
        num_genres_movie = genres_df.loc[movie_id].sum()

        # Normaliser la similarité par la racine du produit des nombres de genres des deux films
        weighted_similarity_score = similarity_score / ((num_genres_selected_movie * num_genres_movie) ** 0.5)

        st.write(f"{i}. {movie_title} (Similarité pondérée : {weighted_similarity_score:.2f})")
        num_recommendations += 1
        if num_recommendations >= 20:
            break
else:
    st.error("Veuillez entrer un ID de film valide.")




