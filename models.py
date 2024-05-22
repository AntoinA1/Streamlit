# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
import pandas as pd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from loaders import load_items, load_ratings
from surprise import PredictionImpossible
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from constants import Constant as C



def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    rd.seed(0)

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First algorithm
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2


# Second algorithm
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):
        return self.the_mean


# Fourth Model
class ModelBaseline4(SVD):
    def __init__(self):
        SVD.__init__(self, n_factors=100)

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
        elif features_method == "title_length":
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')
        elif features_method == "release_year":  
            pattern = r'\((\d{4})\)'
            df_items['release_year'] = df_items[C.LABEL_COL].str.extract(pattern)
            df_features = df_items['release_year'].astype(float).to_frame('release_year')
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
        
        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]
        
        elif self.regressor_method == 'linear_regression':
            for u in trainset.all_users():
                user_ratings = [rating for (_, rating) in trainset.ur[u]]
                item_ids = [trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[u]]
                df_user = pd.DataFrame({'item_id': item_ids, 'user_ratings': user_ratings})
                df_user['item_id'] = df_user['item_id'].map(trainset.to_raw_iid)
                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
                ).dropna()
                lr = LinearRegression(fit_intercept=False)
                X = df_user[['n_character_title', 'release_year']].values
                y = df_user['user_ratings'].values
                lr.fit(X, y)
                self.user_profile[u] = lr
        
        elif self.regressor_method == 'ridge_regression':
            for u in trainset.all_users():
                user_ratings = [rating for (_, rating) in trainset.ur[u]]
                item_ids = [trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[u]]
                df_user = pd.DataFrame({'item_id': item_ids, 'user_ratings': user_ratings})
                df_user['item_id'] = df_user['item_id'].map(trainset.to_raw_iid)
                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
                ).dropna()
                ridge = Ridge(alpha=1.0, fit_intercept=False)
                X = df_user[['n_character_title', 'release_year']].values
                y = df_user['user_ratings'].values
                ridge.fit(X, y)
                self.user_profile[u] = ridge
        
        else:
            pass
    
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5, 5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])

        elif self.regressor_method in ['linear_regression', 'ridge_regression']:
            raw_item_id = self.trainset.to_raw_iid(i)
            item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values.reshape(1, -1)
            regressor = self.user_profile[u]
            if regressor is None: 
                raise PredictionImpossible('User profile is not trained.')
            score = regressor.predict(item_features.reshape(1, -1))[0]  # Récupérer le scalaire à partir du tableau numpy
            # Clipper le score pour s'assurer qu'il est compris entre 0.5 et 5
            score = max(0.5, min(score, 5))

        else:
            score = None

        return score
