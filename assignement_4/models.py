# standard library imports
from collections import defaultdict
# third parties imports
import numpy as np
import random as rd
import pandas as pd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from loaders import load_items,load_ratings
from surprise.prediction_algorithms.predictions import PredictionImpossible
from constants import Constant as C
from sklearn.linear_model import LinearRegression
from loaders import load_ratings
from loaders import load_items
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import heapq
from constants import Constant
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
 
 
def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
 
    Args:
        predictions(list of tuples): The list of predictions, where each tuple
            contains (uid, iid, r_ui, est, details).
        n(int): The number of recommendations to output for each user.
 
   Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
 
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        top_n[uid].append((iid, est))
 
    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
 
    return top_n
 
 

 
   
 
class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)
 
    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        if features_method is None:
            df_features = None

        # title_length feature
        elif features_method == "title_length": 
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')

        # release year feature
        elif features_method == "release_year":

            # Extract release year from movie title
            df_items['release_year'] = df_items['title'].str.extract(r'\((\d{4})\)')
            df_items['release_year'] = df_items['release_year'].astype(float)

            # Inside the create_content_features method
            df_features = df_items[['release_year']].fillna(1990.0)

        # genres feature
        elif features_method == "genres":
            df_features = df_items['genres'].str.get_dummies(sep='|')
            df_ratings = load_ratings(surprise_format=False)

            # Replace binary values with average ratings for each genre
            for genre in df_features.columns:

                # Calculate the average rating for movies with this genre
                genre_ratings = []
                for index, row in df_items.iterrows():
                    if genre in row['genres']:

                        # Check if the movie has been rated by users
                        if index in df_ratings['movieId'].values:
                            ratings_for_movie = df_ratings[df_ratings['movieId'] == index]['rating'].values
                            genre_ratings.extend(ratings_for_movie)

                # Calculate the average rating for this genre
                if genre_ratings:
                    genre_avg_rating = sum(genre_ratings) / len(genre_ratings)
                else:
                    genre_avg_rating = 0 

                # Replace binary values with average rating for this genre
                df_features[genre] = genre_avg_rating
 
        else: 
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features
   
 
    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
       
        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}
 
        if self.regressor_method == 'random_score':
            pass
        elif self.regressor_method == 'linear_regression':
            for u in self.user_profile:

                # List to stock data of each users 
                user_data = []  
                for inner_iid, rating in trainset.ur[u]:
                    raw_iid = trainset.to_raw_iid(inner_iid)
                    user_data.append({'item_id': raw_iid, 'user_ratings': rating})
              
                # Create the datafram
                df_user = pd.DataFrame(user_data)

                # Fusion df_user with self.content_features
                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
               )
                # Extract features (X) and targets (y)
                X = df_user['release_year'].values.reshape(-1, 1)
                y = df_user['user_ratings'].values

                # Fitting the linear regressor
                regressor = LinearRegression(fit_intercept=True)
                regressor.fit(X, y)

                # Assigning the linear regressor to the user
                self.user_profile[u] = regressor


        elif self.regressor_method == 'ridge_regression':
            for u in self.user_profile:

                # List to stock data of each users 
                user_data = []  
                for inner_iid, rating in trainset.ur[u]:
                    raw_iid = trainset.to_raw_iid(inner_iid)
                    user_data.append({'item_id': raw_iid, 'user_ratings': rating})

                # Create the datafram
                df_user = pd.DataFrame(user_data)

                # Fusion df_user with self.content_features
                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
                )

                # Select column 
                genre_columns = [col for col in df_user.columns if col != 'item_id' and col != 'user_ratings']

                # Extract features (X) and targets (y)
                X = df_user[genre_columns].values
                y = df_user['user_ratings'].values

                # Fitting the ridge regressor
                regressor = Ridge(alpha=1)
                regressor.fit(X, y)

                # Assigning the ridge regressor to the user
                self.user_profile[u] = regressor

        elif self.regressor_method == 'lasso_regression':
            for u in self.user_profile:

                # List to stock data of each users 
                user_data = []  
                for inner_iid, rating in trainset.ur[u]:
                    raw_iid = trainset.to_raw_iid(inner_iid)
                    user_data.append({'item_id': raw_iid, 'user_ratings': rating})

                # Create the datafram
                df_user = pd.DataFrame(user_data)

                # Fusion df_user with self.content_features 
                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
                )
                # Select column 
                genre_columns = [col for col in df_user.columns if col != 'item_id' and col != 'user_ratings']

                # Extract features (X) and targets (y)
                X = df_user['release_year'].values.reshape(-1, 1)
                y = df_user['user_ratings'].values

                # Fitting the lasso regressor
                regressor = Lasso(alpha=0.1)
                regressor.fit(X, y)

                # Assigning the lasso regressor to the user
                self.user_profile[u] = regressor

        elif self.regressor_method == 'random_forest':
            for u in self.user_profile:

                # List to stock data of each users 
                user_data = [] 

                for inner_iid, rating in trainset.ur[u]:
                    raw_iid = trainset.to_raw_iid(inner_iid)
                    user_data.append({'item_id': raw_iid, 'user_ratings': rating})

                # Create the dataframe 
                df_user = pd.DataFrame(user_data)

                # Fusion df_user with self.content_features 
                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
                )

                # Select column 
                genre_columns = [col for col in df_user.columns if col != 'item_id' and col != 'user_ratings']

                # Extract features (X) and targets (y)
                X = df_user[genre_columns].values.reshape(-1, 1)
                y = df_user['user_ratings'].values

                # Fitting the random forest regressor
                regressor = RandomForestRegressor(n_estimators=5, random_state=42)
                regressor.fit(X, y)

                # Assigning the random forest regressor to the user
                self.user_profile[u] = regressor

        elif self.regressor_method == 'gradient_boosting':
            for u in self.user_profile:

                # List to stock data of each users 
                user_data = []  

                for inner_iid, rating in trainset.ur[u]:
                    raw_iid = trainset.to_raw_iid(inner_iid)
                    user_data.append({'item_id': raw_iid, 'user_ratings': rating})

                # Create the dataframe 
                df_user = pd.DataFrame(user_data)

                # Fusion df_user with self.content_features 
                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
                )

                # Complete the Nan value by 0
                df_user.fillna(0, inplace=True)

                # Select colomn 
                genre_columns = [col for col in df_user.columns if col != 'item_id' and col != 'user_ratings']
                
                # Extract features (X) and targets (y)
                X = df_user[genre_columns].values
                y = df_user['user_ratings'].values

                # Fitting the gradient boosting regressor
                regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                regressor.fit(X, y)

                # Assigning the gradient boosting regressor to the user
                self.user_profile[u] = regressor


        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]
        else:
            pass

        
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
 
        # estimate for the random score
        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5,5)

        # estimate for the random sample
        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])

        # estimate for linear_regression, ridge_regression, lasso_regression
        elif self.regressor_method == 'linear_regression' or self.regressor_method == 'ridge_regression' or self.regressor_method == 'lasso_regression':
            user_regressor = self.user_profile[u]

            # Convert the id of element into id brut 
            raw_item_id = self.trainset.to_raw_iid(i)

            # Recover features of element 
            item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values

            # Make prediction with regressor
            score = user_regressor.predict(item_features.reshape(1, -1))[0] 

       # estimate for random_forest
        elif self.regressor_method == 'random_forest':
            user_regressor = self.user_profile[u]

            # Convert the id of element into id brut 
            raw_item_id = self.trainset.to_raw_iid(i)

            # Recover features of element 
            item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values

            # Make prediction with regressor 
            score = user_regressor.predict(item_features.reshape(1, -1))[0]  #

        # estimate for gradient_boosting
        elif self.regressor_method == 'gradient_boosting':
            user_regressor = self.user_profile[u]

            # Convert the id of element into id brut 
            raw_item_id = self.trainset.to_raw_iid(i)

            # Recover features of element 
            item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values

            # Make prediction with regressor
            score = user_regressor.predict(item_features.reshape(1, -1))[0] 
        
        else:
            score=None
        
 
        return score
 