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
from surprise.prediction_algorithms.matrix_factorization import SVD as OriginalSVD



# Loading data
movies_df = load_items()
ratings_data = load_ratings(surprise_format=True)
trainset = ratings_data.build_full_trainset()

# Dict. for mapping movieId to movie_title
movie_id_to_title = dict(zip(movies_df.index, movies_df['title']))

# Streamlit Sidebar
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
        st.sidebar.subheader("User-Based Options")
        similarity_metric = st.sidebar.selectbox("Similarity Metric", ["Cosine", "Pearson", "Msd"])
        k_neighbors = st.sidebar.slider("Number of Neighbors", min_value=1, max_value=20, value=3, step=1)
        
        return similarity_metric, k_neighbors
    
    elif rec_type == "Content-Based":
        unique_genres = set('|'.join(movies_df['genres']).split('|'))  # Obtenir les genres uniques
        selected_genres = st.sidebar.multiselect("Wished Genres", list(unique_genres))
        return selected_genres

class UserBasedRecommender:
    def __init__(self, sim_options={}, k=3, min_k=4):
        self.sim_options = sim_options
        self.k = k
        self.min_k = min_k
        self.algorithm = None

    def fit(self, data):
        train_set, _ = train_test_split(data, test_size=0.25)
        # Using Knnwithmeans for fitting the data
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
    
    def estimate(self, user_id, item_id):
        # Predictions with knn algorithm
        prediction = self.algorithm.predict(str(user_id), str(item_id))
        return prediction.est



class ContentBased:
    def __init__(self):
        self.movies_df = load_items()
        self.ratings_df = load_ratings(surprise_format=False)
        # One-hot encoding for genres
        self.genres_df = self.movies_df['genres'].str.get_dummies(sep='|')
        # Cosinus similarity between films
        self.film_similarity = cosine_similarity(self.genres_df)
    
    def extract_year_from_title(self, title):
        # Reg exp for year in title
        match = re.search(r'\(([0-9]{4})\)$', title)
        if match:
            return int(match.group(1))
        else:
            return None
    
    def recommend_movies(self, selected_movie, selected_genres):
        if selected_movie:
            # Find movie ID
            selected_movie_id = self.movies_df[self.movies_df['title'] == selected_movie].index[0]
            selected_movie_year = self.extract_year_from_title(selected_movie)

            # Number of genres if ponderation
            num_genres_selected_movie = self.genres_df.loc[selected_movie_id].sum()

            if not selected_genres:
                similar_movies = pd.Series(self.film_similarity[selected_movie_id], index=self.movies_df.index).sort_values(ascending=False)[1:]
            else:
                # Using genres selected by user
                similar_movies = pd.Series(self.film_similarity[selected_movie_id], index=self.movies_df.index).sort_values(ascending=False)[1:]
                similar_movies = similar_movies[self.movies_df['genres'].apply(lambda x: any(genre in selected_genres for genre in x.split('|')))]

            # Not recommend selected_movie
            similar_movies = similar_movies[similar_movies.index != selected_movie_id]

            movie_scores = []
            for movie_id, similarity_score in similar_movies.items():
                movie_title = self.movies_df.loc[movie_id, 'title']
                movie_year = self.extract_year_from_title(movie_title)
                movie_scores.append((movie_id, similarity_score, movie_year))

            movie_scores.sort(key=lambda x: x[1], reverse=True)

            # For same similarity score, year sorting
            similar_score_groups = {}
            for movie_id, similarity_score, movie_year in movie_scores:
                if similarity_score not in similar_score_groups:
                    similar_score_groups[similarity_score] = []
                similar_score_groups[similarity_score].append((movie_id, similarity_score, movie_year))

            sorted_movie_scores = []
            for similarity_score, movie_year_group in similar_score_groups.items():
                valid_year_group = [movie for movie in movie_year_group if movie[2] is not None]
                invalid_year_group = [movie for movie in movie_year_group if movie[2] is None]
                
                valid_year_group.sort(key=lambda x: abs(x[2] - selected_movie_year))
                
                sorted_movie_scores.extend(valid_year_group)
                sorted_movie_scores.extend(invalid_year_group)

            st.write(f"Similar movies to '{selected_movie}' are :")

            # Showing recommendations
            num_recommendations = 0
            for movie_id, similarity_score, movie_year in sorted_movie_scores:
                movie_title = self.movies_df.loc[movie_id, 'title']
                if movie_year is not None:  
                    st.write(f"{movie_title}")
                else:
                    st.write(f"{movie_title}")
                num_recommendations += 1
                if num_recommendations >= 20:
                    break
        else:
            st.error("Veuillez sélectionner un film.")

class SVDBiasedSGD(AlgoBase):
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.svd_model = None

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.svd_model = OriginalSVD(n_factors=self.n_factors, n_epochs=self.n_epochs, lr_all=self.lr_all, reg_all=self.reg_all)
        self.svd_model.fit(trainset)

    def estimate(self, u, i):
        return self.svd_model.predict(u, i).est

    def recommend_items(self, user_id, n=5):
        anti_test_set = self.trainset.build_anti_testset()
        predictions = self.svd_model.test(anti_test_set)
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
    

def get_movie_title_by_id(movie_id, movie_id_to_title):
    return movie_id_to_title.get(movie_id, "Titre non trouvé")

if 'new_ratings' not in st.session_state:
    st.session_state['new_ratings'] = []

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

    \n- Latent Factor Model is based on factors that are derived from the datas. It recommends movies based on underlying patterns in the user-item interaction matrix, captured through latent factors.
These recommendations are based on similarities in how movies are rated across different latent factors, allowing the system to predict which movies the user might enjoy based on existing ratings.

    \n- Feel free to try some options available on the sidebar :)
    """
if st.button("Rec Sys Explanation", key= "Explanation button"):
    st.write(explication)

st.header("Choose one Recommender System")
rec_type = st.radio("", ["User-Based", "Content-Based", "Latent Factor Model"])

            
if rec_type == "User-Based":
    # Specific sidebar
    similarity_metric, k_neighbors = sidebar("User-Based")

    if 'new_ratings' not in st.session_state:
        st.session_state['new_ratings'] = []

    recommender = UserBasedRecommender(
        sim_options={'name': similarity_metric, 'user_based': True, 'min_support': 3},
        k=k_neighbors, 
        min_k=2
    )

    recommender.fit(ratings_data)

    st.header("New User - Create a profile")

    movie_query = st.text_input("Enter film name")

    selected_movie = None

    if movie_query:
        # Showing films close grammarly speaking
        movie_choices = movies_df['title'].tolist()
        search_results = process.extract(movie_query, movie_choices, limit=5)
        selected_movie = st.selectbox("Select a film :", [result[0] for result in search_results])

        if selected_movie:
            rating = st.slider("Select a rating", 0.5, 5.0, 3.0, step=0.5)

            if st.button("Add film and rating"):
                movie_id = movies_df[movies_df['title'] == selected_movie].index[0]
                st.session_state['new_ratings'].append((movie_id, rating))
                st.success(f"Added: {selected_movie} - {rating}")

    # Showing added films and ratings
    if st.session_state['new_ratings']:
        st.header("Rated films")
        for i, (movie_id, rating) in enumerate(st.session_state['new_ratings']):
            st.write(f"Film : {get_movie_title_by_id(movie_id, movie_id_to_title)}, Rating : {rating}")
            if st.button(f"Delete rating {i+1}", key=f"del_{i}"):
                del st.session_state['new_ratings'][i]
                st.success("Rating deleted !")
                st.experimental_rerun()

    if st.button("Get recommendations"):
        if st.session_state['new_ratings']:
            new_ratings_df = pd.DataFrame(st.session_state['new_ratings'], columns=['movieId', 'rating'])
            new_ratings_df['userId'] = ratings_data.df['userId'].max() + 1

            combined_ratings = pd.concat([ratings_data.df, new_ratings_df], ignore_index=True)
            reader = Reader(rating_scale=(0.5, 5.0))
            combined_data = Dataset.load_from_df(combined_ratings[['userId', 'movieId', 'rating']], reader)

            # Train on new data
            recommender.fit(combined_data)

            user_id = new_ratings_df['userId'].iloc[0]
            recommendations = recommender.get_user_recommendations(user_id)

            if recommendations:
                st.header("Personalized recommendations")
                for movie_id, rating in recommendations:
                    movie_title = get_movie_title_by_id(movie_id, movie_id_to_title)
                    if movie_title != "Titre non trouvé":  
                        estimated_rating = recommender.estimate(user_id, movie_id)
                        st.write(f"Recommended film : {movie_title}")
            else:
                st.warning("No recommendation for this user.")
        else:
            st.warning("Please add some ratings before getting recommendations.")

elif rec_type == "Content-Based":
    
    selected_genres = sidebar("Content-Based")
    
    st.header("Which film did you like? :")
    recommender = ContentBased()
    
    movie_query = st.text_input("Enter film:", key="movie_input")
    
    if movie_query:
        movie_choices = recommender.movies_df['title'].tolist()
        search_results = process.extract(movie_query, movie_choices, limit=5)
        selected_movie = st.selectbox("Select a film :", [result[0] for result in search_results], key="movie_selectbox")

        if selected_movie:
            # Showing recommendations
            if st.button("Get recommendations"):
                recommender.recommend_movies(selected_movie, selected_genres)

elif rec_type == "Latent Factor Model":
    
    st.sidebar.title("Options")

    # Sidebar parameters
    n_factors = st.sidebar.slider("Number of factors", min_value=1, max_value=100, value=20, step=1)
    n_epochs = st.sidebar.slider("Number of epochs", min_value=1, max_value=100, value=20, step=1)

    algo = OriginalSVD(n_factors=n_factors, n_epochs=n_epochs)

    algo.fit(trainset)

    # Getting unique users ID
    unique_users = sorted(set([user_id for user_id, *_ in ratings_data.raw_ratings]))

    max_users_to_display = 700

    users_to_display = unique_users[:max_users_to_display]

    user_id = st.selectbox("Select a user", users_to_display)
  
    # Showing recommendations
    if st.button("Get recommendations"):
        recommendations = algo.test(trainset.build_anti_testset())
        user_recommendations = defaultdict(list)
        for uid, iid, _, est, _ in recommendations:
            user_recommendations[uid].append((iid, est))
        
        if user_id in user_recommendations:
            st.header(f"Top recommendations for User {user_id}:")
            for movie_id, rating in user_recommendations[user_id][:10]:
                movie_title = get_movie_title_by_id(movie_id, movie_id_to_title)
                st.write(f"- {movie_title}")
        else:
            st.warning("No recommendations available for this user.")
