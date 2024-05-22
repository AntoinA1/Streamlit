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
    
class ContentBased():
    def __init__(self, trainset, features_method, regressor_method):
        self.trainset = trainset
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)
        self.user_profiles = {}

    def create_content_features(self, features_method):
        df_items = load_items()
        if features_method is None:
            df_features = None
        elif features_method == "title_length":
            df_features = df_items['title'].apply(lambda x: len(x)).to_frame('n_character_title')
        elif features_method == "release_year":
            df_items['release_year'] = df_items['title'].str.extract(r'\((\d{4})\)')
            df_items['release_year'] = df_items['release_year'].astype(float)
            df_features = df_items[['release_year']]
        elif features_method == "genres":
            df_features = df_items['genres'].str.get_dummies(sep='|')
        else:
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features

    def fit(self, trainset):
        
        self.user_profile = {u: None for u in trainset.all_users()}

        for u in self.user_profile:
            user_data = []

            for inner_iid, rating in trainset.ur[u]:
                raw_iid = trainset.to_raw_iid(inner_iid)
                user_data.append({'item_id': raw_iid, 'user_ratings': rating})

            df_user = pd.DataFrame(user_data)
            df_user = df_user.merge(
                self.content_features,
                how='left',
                left_on='item_id',
                right_index=True
            )

            genre_columns = [col for col in df_user.columns if col != 'item_id' and col != 'user_ratings']
            X = df_user[genre_columns].values
            y = df_user['user_ratings'].values

            if self.regressor_method == 'linear_regression':
                regressor = LinearRegression()
            elif self.regressor_method == 'ridge_regression':
                regressor = Ridge(alpha=1)
            elif self.regressor_method == 'lasso_regression':
                regressor = Lasso(alpha=1.0)
            elif self.regressor_method == 'random_forest':
                regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise NotImplementedError(f'Regressor method {self.regressor_method} not yet implemented')

            regressor.fit(X, y)
            self.user_profile[u] = regressor

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        user_regressor = self.user_profile[u]
        raw_item_id = self.trainset.to_raw_iid(i)
        item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values

        score = user_regressor.predict(item_features.reshape(1, -1))[0]
        score = max(0.5, min(score, 5))
        return score

    def get_user_recommendations(self, user_id, n=10):
        anti_test_set = self.trainset.build_anti_testset()
        predictions = self.test(anti_test_set)
        user_recommendations = defaultdict(list)
        for uid, iid, _, est, _ in predictions:
            user_recommendations[uid].append((iid, est))
        if user_id in user_recommendations:
            user_recommendations[user_id].sort(key=lambda x: x[1], reverse=True)
            return user_recommendations[user_id][:n]
        else:
            return []

test = ContentBased.create_content_features("ContentBased", 'genres')

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

# Choisir le type de système de recommandation
st.header("Choisissez le type de système de recommandation")
rec_type = st.selectbox("Type de recommandation", ["User-Based", "Content-Based"])

# Création de l'instance du Recommender en fonction du choix
if rec_type == "User-Based":
    recommender = UserBasedRecommender(sim_options={'name': 'cosine', 'user_based': True, 'min_support': 3}, k=3, min_k=4)
elif rec_type == "Content-Based":
    recommender = ContentBased(trainset=trainset, features_method= "genres", regressor_method= "ridge_regression" )

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
