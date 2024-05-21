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

# Chargement des données
movies_df = load_items()
ratings_data = load_ratings(surprise_format=True)

# Créer un dictionnaire qui mappe les IDs des films aux titres correspondants
movie_id_to_title = dict(zip(movies_df.index, movies_df['title']))

# Classe Recommender
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

# Utiliser le dictionnaire pour obtenir le titre du film par son ID
def get_movie_title_by_id(movie_id, movie_id_to_title):
    return movie_id_to_title.get(movie_id, "Titre non trouvé")

# Initialiser l'état de la session pour stocker les nouvelles notations
if 'new_ratings' not in st.session_state:
    st.session_state['new_ratings'] = []

# Titre de l'application
st.title("Recommender System, Group 6")

# Section pour l'insertion des films et des notes pour un nouvel utilisateur
st.header("Nouveau utilisateur - Entrez vos films et notes")

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

# Création de l'instance du Recommender
recommender = UserBasedRecommender(sim_options={'name': 'cosine', 'user_based': True, 'min_support': 3}, k=3, min_k=4)
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
