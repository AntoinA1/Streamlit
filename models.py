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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor





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
    def __init__(self, features_method, regressor_method, alpha =1.0):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.features_method = features_method
        self.content_features, self.scaler = self.create_content_features(features_method)
        self.alpha = alpha

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        if features_method is None:
            df_features = None
            scaler = None
        elif features_method == "title_length":
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')
            scaler = StandardScaler()
            df_features[['n_character_title']] = scaler.fit_transform(df_features[['n_character_title']])
        elif features_method == "release_year":
            pattern = r'\((\d{4})\)'
            df_items['release_year'] = df_items[C.LABEL_COL].str.extract(pattern)
            df_features = df_items['release_year'].astype(float).to_frame('release_year')
            scaler = StandardScaler()
            df_features[['release_year']] = scaler.fit_transform(df_features[['release_year']])
        elif features_method == "genres":
            df_items['genres'] = df_items['genres'].fillna('')
            tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
            tfidf_matrix = tfidf_vectorizer.fit_transform(df_items['genres'])
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=tfidf_vectorizer.get_feature_names_out())
            scaler = StandardScaler()
            df_features = pd.DataFrame(scaler.fit_transform(tfidf_df), columns=tfidf_df.columns, index=tfidf_df.index)
        elif features_method == "all":
            df_items['title_length'] = df_items[C.LABEL_COL].apply(lambda x: len(x))
            pattern = r'\((\d{4})\)'
            df_items['release_year'] = df_items[C.LABEL_COL].str.extract(pattern)
            df_items['genres'] = df_items['genres'].fillna('')
            tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
            tfidf_matrix = tfidf_vectorizer.fit_transform(df_items['genres'])
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=tfidf_vectorizer.get_feature_names_out())
            scaler = StandardScaler()
            df_features = pd.concat([df_items['title_length'], df_items['release_year'].astype(float), tfidf_df], axis=1)
            df_features = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns, index=df_features.index)
        else:
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        
        return df_features, scaler
    
    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
        
        self.user_profile = {u: None for u in trainset.all_users()}

        if self.regressor_method == 'random_score':
            pass
        
        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]

        elif self.regressor_method in ['linear_regression', 'ridge_regression', 'random_forest']:
            for u in trainset.all_users():
                user_ratings = [rating for (_, rating) in trainset.ur[u]]
                item_ids = [trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[u]]

                # Initialize df_user to None
                df_user = None

                if self.regressor_method in ['linear_regression', 'ridge_regression']:
                    df_user = pd.DataFrame({'item_id': item_ids, 'user_ratings': user_ratings})
                    df_user['item_id'] = df_user['item_id'].map(trainset.to_raw_iid)
                    df_user = df_user.merge(
                        self.content_features,
                        how='left',
                        left_on='item_id',
                        right_index=True
                    ).dropna()

                    if df_user.empty:
                        print(f"User {u} has no valid data after merging with content features.")
                        continue

                    if self.regressor_method == 'ridge_regression':
                        model = Ridge(alpha=self.alpha, fit_intercept=False)
                    else:
                        model = LinearRegression(fit_intercept=False)
                    
                    X = df_user.drop(columns=['item_id', 'user_ratings']).values  # Extract features
                    y = df_user['user_ratings'].values
                    
                    model.fit(X, y)
                    
                    self.user_profile[u] = model

                elif self.regressor_method == 'random_forest':
                    for u in trainset.all_users():
                        user_ratings = [rating for (_, rating) in trainset.ur[u]]
                        item_ids_filtered = []  # Initialize list to store filtered item ids
                        for (iid, _) in trainset.ur[u]:
                            if iid in self.content_features.index:  # Filter out unknown item ids
                                item_ids_filtered.append(trainset.to_raw_iid(iid))  # Add to filtered item ids

                        if item_ids_filtered and len(item_ids_filtered) == len(user_ratings):  # Check if both lists have the same length
                            df_user = pd.DataFrame({'item_id': item_ids_filtered, 'user_ratings': user_ratings})
                            df_user['item_id'] = df_user['item_id'].map(trainset.to_raw_iid)
                            df_user = df_user.merge(
                                self.content_features,
                                how='left',
                                left_on='item_id',
                                right_index=True
                            ).dropna()

                            if not df_user.empty:  # Check if the DataFrame is not empty
                                model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust parameters as needed
                                X = df_user.drop(columns=['item_id', 'user_ratings']).values  # Extract features
                                y = df_user['user_ratings'].values
                                
                                model.fit(X, y)
                                
                                self.user_profile[u] = model
                        else:
                            print(f"User {u} has no valid data after filtering or mismatch in lengths.")


        else:
            pass


    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5, 5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])

        elif self.regressor_method in ['linear_regression', 'ridge_regression']:
            raw_item_id = self.trainset.to_raw_iid(i)
            if raw_item_id not in self.content_features.index:
                raise PredictionImpossible('Item is unknown.')

            item_features = self.content_features.loc[raw_item_id].values.reshape(1, -1)
            regressor = self.user_profile[u]
            if regressor is None: 
                raise PredictionImpossible('User profile is not trained.')

            score = regressor.predict(item_features)[0]
            score = float(score)
            score = max(0.5, min(score, 5))

        else:
            raise PredictionImpossible('Unknown regressor method.')

        return score
