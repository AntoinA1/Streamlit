import streamlit as st
from fuzzywuzzy import process
from loaders import load_items, load_ratings
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise import Dataset, Reader
from collections import defaultdict
import pandas as pd
import os
import glob
from surprise import AlgoBase, PredictionImpossible
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import random as rd
from sklearn.metrics.pairwise import cosine_similarity


# Chargement des données
movies_df = load_items()
ratings_data = load_ratings(surprise_format=True)
trainset = ratings_data.build_full_trainset()

# Créer un dictionnaire qui mappe les IDs des films aux titres correspondants
movie_id_to_title = dict(zip(movies_df.index, movies_df['title']))

# Classe Recommender pour User-Based
class UserBasedRecommender:
    def __init__(self, sim_options={}, k=3, min_k=4):
        self.sim_options = sim_options
        self.k = k
        self.min_k = min_k
        self.algorithm = None

    def fit(self, data):
        train_set, _ = train_test_split(data, test_size=0.25)
        self.algorithm = KNNWithMeans(sim_options=self.sim_options, k=self.k, min_k=self.min_k)
        self.algorithm.fit(train_set)

    def recommend_items(self, user_id, n=5):
        anti_test_set = self.algorithm.trainset.build_anti_testset()
        predictions = self.algorithm.test(anti_test_set)
        user_recommendations = defaultdict(list)
        for uid, iid, _, est, _ in predictions:
            user_recommendations[uid].append((iid, est))
        if user_id in user_recommendations:
            user_recommendations[user_id].sort(key=lambda x: x[1], reverse=True)
            return user_recommendations[user_id][:n]
        else:
            return []

    def get_user_recommendations(self, user_id, n=10):
        return self.recommend_items(user_id, n)
    
class ContentBased:
    def __init__(self):
        # Chargement des données
        self.movies_df = load_items()
        self.ratings_df = load_ratings(surprise_format=False)
        
        # One-hot encoding des genres
        self.genres_df = self.movies_df['genres'].str.get_dummies(sep='|')
        
        # Calcul de la similarité cosinus entre les films
        self.film_similarity = cosine_similarity(self.genres_df)
    
    def recommend_movies(self, movie_query):
        # Recherche des films correspondant à la requête de l'utilisateur
        movie_choices = self.movies_df['title'].tolist()
        search_results = process.extract(movie_query, movie_choices, limit=5)
        selected_movie = st.selectbox("Sélectionnez un film :", [result[0] for result in search_results])

        # Vérifier si un film a été sélectionné
        if selected_movie:
            # Trouver l'ID du film sélectionné
            selected_movie_id = self.movies_df[self.movies_df['title'] == selected_movie].index[0]

            # Calculer le nombre de genres du film sélectionné
            num_genres_selected_movie = self.genres_df.loc[selected_movie_id].sum()

            # Trouver les films similaires à celui que l'utilisateur a aimé
            similar_movies = pd.Series(self.film_similarity[selected_movie_id], index=self.movies_df.index).sort_values(ascending=False)[1:]

            # Afficher le titre du film demandé
            st.write(f"Voici les films les plus similaires à '{selected_movie}' :")

            # Afficher les films recommandés à l'utilisateur
            num_recommendations = 0
            for i, (movie_id, similarity_score) in enumerate(similar_movies.items(), start=1):
                movie_title = self.movies_df.loc[movie_id, 'title']

                # Calculer le nombre de genres du film actuellement examiné
                num_genres_movie = self.genres_df.loc[movie_id].sum()

                # Normaliser la similarité par la racine du produit des nombres de genres des deux films
                weighted_similarity_score = similarity_score / ((num_genres_selected_movie * num_genres_movie) ** 0.5)

                st.write(f"{i}. {movie_title} (Similarité pondérée : {weighted_similarity_score:.2f})")
                num_recommendations += 1
                if num_recommendations >= 20:
                    break
        else:
            st.error("Veuillez sélectionner un film.")
    

# Utiliser le dictionnaire pour obtenir le titre du film par son ID
def get_movie_title_by_id(movie_id, movie_id_to_title):
    return movie_id_to_title.get(movie_id, "Titre non trouvé")


if 'new_ratings' not in st.session_state:
    st.session_state['new_ratings'] = []

# Titre de l'application
st.title("Recommender System, Group 6")

# Section pour choisir le type de système de recommandation
st.header("Choisissez le type de système de recommandation")
rec_type = st.selectbox("Type de recommandation", ["User-Based", "Content-Based"])

# Si User-Based est sélectionné
if rec_type == "User-Based":
    # Section pour l'insertion des films et des notes pour un nouvel utilisateur
    st.header("Nouvel utilisateur - Entrez vos films et notes")
    movie_query = st.text_input("Entrez le nom d'un film que vous avez vu")
    if movie_query:
        movie_choices = movies_df['title'].tolist()
        search_results = process.extract(movie_query, movie_choices, limit=5)
        selected_movie = st.selectbox("Sélectionnez un film :", [result[0] for result in search_results])
        if selected_movie:
            rating = st.slider(f"Notez le film {selected_movie}", 0.5, 5.0, 3.0, step=0.5)
            if st.button("Ajouter ce film et sa note"):
                movie_id = movies_df[movies_df['title'] == selected_movie].index[0]
                st.session_state['new_ratings'].append((movie_id, rating))
                st.success(f"Ajouté: {selected_movie} - {rating}")

    # Afficher les films et les notes ajoutés
    if st.session_state['new_ratings']:
        st.header("Films notés")
        for i, (movie_id, rating) in enumerate(st.session_state['new_ratings']):
            st.write(f"Film : {get_movie_title_by_id(movie_id, movie_id_to_title)}, Note : {rating}")
            if st.button(f"Supprimer la notation {i+1}", key=f"del_{i}"):
                del st.session_state['new_ratings'][i]
                st.success("Notation supprimée avec succès !")
                st.experimental_rerun()

    # Création de l'instance du Recommender en fonction du choix
    recommender = UserBasedRecommender(sim_options={'name': 'cosine', 'user_based': True, 'min_support': 3}, k=3, min_k=4)

    # Entraîner le modèle avec les données initiales
    recommender.fit(ratings_data)

    # Bouton pour générer les recommandations
    if st.button("Obtenir des recommandations"):
        if st.session_state['new_ratings']:
            new_ratings_df = pd.DataFrame(st.session_state['new_ratings'], columns=['movieId', 'rating'])
            new_ratings_df['userId'] = ratings_data.df['userId'].max() + 1

            combined_ratings = pd.concat([ratings_data.df, new_ratings_df], ignore_index=True)
            reader = Reader(rating_scale=(0.5, 5.0))
            combined_data = Dataset.load_from_df(combined_ratings[['userId', 'movieId', 'rating']], reader)

            recommender.fit(combined_data)

            user_id = new_ratings_df['userId'].iloc[0]
            recommendations = recommender.get_user_recommendations(user_id)

            if recommendations:
                st.header("Recommandations personnalisées")
                for movie_id, rating in recommendations:
                    movie_title = get_movie_title_by_id(movie_id, movie_id_to_title)
                    st.write(f"Film recommandé : {movie_title}, Estimation : {rating}")
            else:
                st.warning("Aucune recommandation trouvée pour cet utilisateur.")

# Si Content-Based est sélectionné
elif rec_type == "Content-Based":
    # Section pour entrer l'ID du film que l'utilisateur a aimé
    st.header("Quel film avez-vous aimé? :")
    selected_movie_id = st.text_input("Entrez le nom du film que vous avez aimé :")
    recommender = ContentBased()

    # Afficher les recommandations
    if st.button("Obtenir des recommandations"):
        recommender.recommend_movies(selected_movie_id)
