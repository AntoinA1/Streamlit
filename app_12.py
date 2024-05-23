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
import re
from sklearn.metrics.pairwise import cosine_similarity


# Chargement des données
movies_df = load_items()
ratings_data = load_ratings(surprise_format=True)
trainset = ratings_data.build_full_trainset()

# Créer un dictionnaire qui mappe les IDs des films aux titres correspondants
movie_id_to_title = dict(zip(movies_df.index, movies_df['title']))

# Fonction pour la barre latérale
def sidebar(rec_type):
    
    st.sidebar.title('Options')

    st.markdown(
            """
            <style>
            .sidebar .sidebar-content {
                background-color: #2E2E2E; /* Couleur de fond sombre */
                color: #FFFFFF; /* Couleur du texte */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    if rec_type == "User-Based":
        # Options pour le User-Based Recommender
        st.sidebar.subheader("User-Based Options")
        # Choix de la métrique de similarité
        similarity_metric = st.sidebar.selectbox("Similarity Metric", ["Cosine", "Pearson", "Msd"])
        # Réglage du nombre de voisins
        k_neighbors = st.sidebar.slider("Number of Neighbors", min_value=1, max_value=20, value=3, step=1)
        
        return similarity_metric, k_neighbors
    
    elif rec_type == "Content-Based":
        # Sélection des genres
        unique_genres = set('|'.join(movies_df['genres']).split('|'))  # Obtenir les genres uniques
        selected_genres = st.sidebar.multiselect("Wished Genres", list(unique_genres))
        return selected_genres

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
    
    def extract_year_from_title(self, title):
        # Utilisation d'une expression régulière pour extraire l'année du titre du film
        match = re.search(r'\(([0-9]{4})\)$', title)
        if match:
            return int(match.group(1))
        else:
            return None
    
    def recommend_movies(self, selected_movie, selected_genres):
        # Vérifier si un film a été sélectionné
        if selected_movie:
            # Trouver l'ID du film sélectionné (qui est l'index dans le DataFrame)
            selected_movie_id = self.movies_df[self.movies_df['title'] == selected_movie].index[0]

            # Extraire l'année de publication du film sélectionné
            selected_movie_year = self.extract_year_from_title(selected_movie)

            # Calculer le nombre de genres du film sélectionné
            num_genres_selected_movie = self.genres_df.loc[selected_movie_id].sum()

            # Si aucun genre n'est sélectionné, utilisez tous les genres disponibles
            if not selected_genres:
                similar_movies = pd.Series(self.film_similarity[selected_movie_id], index=self.movies_df.index).sort_values(ascending=False)[1:]
            else:
                # Filtrer les recommandations par les genres sélectionnés par l'utilisateur
                similar_movies = pd.Series(self.film_similarity[selected_movie_id], index=self.movies_df.index).sort_values(ascending=False)[1:]
                similar_movies = similar_movies[self.movies_df['genres'].apply(lambda x: any(genre in selected_genres for genre in x.split('|')))]

            # Filtrer les recommandations pour exclure le film entré par l'utilisateur
            similar_movies = similar_movies[similar_movies.index != selected_movie_id]

            # Créer une liste de tuples (ID de film, score de similarité, année de publication) pour les films similaires
            movie_scores = []
            for movie_id, similarity_score in similar_movies.items():
                movie_title = self.movies_df.loc[movie_id, 'title']
                movie_year = self.extract_year_from_title(movie_title)
                movie_scores.append((movie_id, similarity_score, movie_year))

            # Trier les films similaires par score de similarité (décroissant)
            movie_scores.sort(key=lambda x: x[1], reverse=True)

            # Créer un dictionnaire pour regrouper les films ayant le même score de similarité
            similar_score_groups = {}
            for movie_id, similarity_score, movie_year in movie_scores:
                if similarity_score not in similar_score_groups:
                    similar_score_groups[similarity_score] = []
                similar_score_groups[similarity_score].append((movie_id, similarity_score, movie_year))

            # Parcourir les groupes de films ayant le même score de similarité et trier les années de publication
            sorted_movie_scores = []
            for similarity_score, movie_year_group in similar_score_groups.items():
                # Séparer les films avec une année valide et ceux sans année
                valid_year_group = [movie for movie in movie_year_group if movie[2] is not None]
                invalid_year_group = [movie for movie in movie_year_group if movie[2] is None]
                
                # Trier les films avec une année valide par proximité de l'année de publication
                valid_year_group.sort(key=lambda x: abs(x[2] - selected_movie_year))
                
                # Ajouter les films triés et non triés au résultat final
                sorted_movie_scores.extend(valid_year_group)
                sorted_movie_scores.extend(invalid_year_group)

            # Afficher le titre du film demandé
            st.write(f"Similar movies to '{selected_movie}' are :")

            # Afficher les films recommandés à l'utilisateur
            num_recommendations = 0
            for movie_id, similarity_score, movie_year in sorted_movie_scores:
                movie_title = self.movies_df.loc[movie_id, 'title']
                if movie_year is not None:  # Vérifier si l'année de publication peut être extraite
                    st.write(f"{movie_title} (Similarité : {similarity_score:.2f}, Année de publication : {movie_year})")
                else:
                    st.write(f"{movie_title} (Similarité : {similarity_score:.2f})")
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

explication = """
    \nWelcome to our movie recommendation application:

    \n- User-based recommends movies based on the similarity between users. In other words, it suggests movies to a user based on the preferences and ratings of other similar users.
Start by adding movies and their ratings in the "New User - Create a profile" section. 
Once you've added some ratings, you can press the "Get recommendations" button to receive personalized recommendations based on your preferences.
The system will provide you with a list of recommended movies based on the ratings you provided.

    \n- Content-based recommends movies based on the intrinsic characteristics of the movies themselves, such as their genres and release years.
Start by entering the name of a movie you liked in the "Which film did you like?" section.
Then, you can select specific genres (if you wish) in the sidebar to refine your recommendations.
Press the "Get recommendations" button to receive a list of movies similar to the one you liked.

    \n- Feel free to try some options available on the sidebar :)
    """
if st.button("Rec Sys Explanation", key= "Explanation button"):
    st.write(explication)

# Section pour choisir le type de système de recommandation
st.header("Choose one Recommender System")
rec_type = st.radio("", ["User-Based", "Content-Based"])

# Si User-Based est sélectionné
if rec_type == "User-Based":
    # Affichage de la sidebar pour les options
    similarity_metric, k_neighbors = sidebar("User-Based")

    # Initialisation de st.session_state pour stocker les nouvelles évaluations
    if 'new_ratings' not in st.session_state:
        st.session_state['new_ratings'] = []

    # Création de l'instance du Recommender en fonction des options sélectionnées
    recommender = UserBasedRecommender(
        sim_options={'name': similarity_metric, 'user_based': True, 'min_support': 3},
        k=k_neighbors, 
        min_k=2
    )

    # Entraîner le modèle avec les données initiales
    recommender.fit(ratings_data)

    # Section pour l'insertion des films et des notes pour un nouvel utilisateur
    st.header("New User - Create a profile")

    # Entrée du nom du film
    movie_query = st.text_input("Enter film name")

    selected_movie = None

    if movie_query:
        # Proposer une liste de films ressemblant à l'orthographe dès que l'utilisateur commence à saisir le nom du film
        movie_choices = movies_df['title'].tolist()
        search_results = process.extract(movie_query, movie_choices, limit=5)
        selected_movie = st.selectbox("Select a film :", [result[0] for result in search_results])

        # Sélection de la note
        if selected_movie:
            rating = st.slider("Select a rating", 0.5, 5.0, 3.0, step=0.5)

            # Bouton pour ajouter le film et sa note
            if st.button("Add film and rating"):
                movie_id = movies_df[movies_df['title'] == selected_movie].index[0]
                st.session_state['new_ratings'].append((movie_id, rating))
                st.success(f"Added: {selected_movie} - {rating}")

    # Afficher les films et les notes ajoutés
    if st.session_state['new_ratings']:
        st.header("Rated films")
        for i, (movie_id, rating) in enumerate(st.session_state['new_ratings']):
            st.write(f"Film : {get_movie_title_by_id(movie_id, movie_id_to_title)}, Rating : {rating}")
            if st.button(f"Delete rating {i+1}", key=f"del_{i}"):
                del st.session_state['new_ratings'][i]
                st.success("Rating deleted !")
                st.experimental_rerun()

    # Bouton pour générer les recommandations
    if st.button("Get recommendations"):
        if st.session_state['new_ratings']:
            new_ratings_df = pd.DataFrame(st.session_state['new_ratings'], columns=['movieId', 'rating'])
            new_ratings_df['userId'] = ratings_data.df['userId'].max() + 1

            combined_ratings = pd.concat([ratings_data.df, new_ratings_df], ignore_index=True)
            reader = Reader(rating_scale=(0.5, 5.0))
            combined_data = Dataset.load_from_df(combined_ratings[['userId', 'movieId', 'rating']], reader)

            # Réentraîner le modèle avec les nouvelles données
            recommender = UserBasedRecommender(
                sim_options={'name': similarity_metric, 'user_based': True, 'min_support': 3},
                k=k_neighbors, 
                min_k=2
            )
            recommender.fit(combined_data)

            user_id = new_ratings_df['userId'].iloc[0]
            recommendations = recommender.get_user_recommendations(user_id)

            if recommendations:
                st.header("Personalized recommendations")
                for movie_id, rating in recommendations:
                    movie_title = get_movie_title_by_id(movie_id, movie_id_to_title)
                    if movie_title != "Titre non trouvé":  # Vérifier si le titre est trouvé
                        st.write(f"Recommended film : {movie_title}, Estimation : {rating:.2f}")
            else:
                st.warning("No recommendation for this user.")
        else:
            st.warning("Please add some ratings before getting recommendations.")


# Si Content-Based est sélectionné
elif rec_type == "Content-Based":
    # Appeler la fonction sidebar pour obtenir les filtres de genre
    selected_genres = sidebar("Content-Based")
    
    # Section pour entrer le titre du film que l'utilisateur a aimé
    st.header("Which film did you like? :")
    recommender = ContentBased()
    
    # Entrée du titre du film
    movie_query = st.text_input("Enter film:", key="movie_input")
    
    # Afficher les suggestions de films ressemblants à l'orthographe
    if movie_query:
        movie_choices = recommender.movies_df['title'].tolist()
        search_results = process.extract(movie_query, movie_choices, limit=5)
        selected_movie = st.selectbox("Select a film :", [result[0] for result in search_results], key="movie_selectbox")

        # Afficher les recommandations si un film est sélectionné
        if selected_movie:
            # Afficher les recommandations
            if st.button("Get recommendations"):
                recommender.recommend_movies(selected_movie, selected_genres)