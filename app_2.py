import streamlit as st
from fuzzywuzzy import process
from loaders import load_items, load_ratings
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNWithMeans
from collections import defaultdict

# Chargement des données
movies_df = load_items()
ratings_data = load_ratings(surprise_format=True)

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

# Titre de l'application
st.title("Recommender System, Group 6")

# Création de l'instance du Recommender
recommender = UserBasedRecommender(sim_options={'name': 'cosine', 'user_based': True, 'min_support': 3}, k=3, min_k=2)
recommender.fit(ratings_data)

# Section pour l'insertion de l'ID utilisateur
user_id = st.text_input("Enter User ID:")

# Obtenir les recommandations personnalisées pour l'utilisateur
if user_id:
    # Obtention des recommandations pour cet utilisateur
    recommendations = recommender.get_user_recommendations(user_id)
    
    # Affichage des recommandations
    if recommendations:
        st.header("Recommandations personnalisées")
        for movie_id, rating in recommendations:
            
            st.write(f"Film recommandé : {movie_id}, Estimation : {rating}")
    else:
        st.warning("Aucune recommandation trouvée pour cet utilisateur.")
