# local imports
from models import *
from surprise import accuracy
 
class EvalConfig:
  
    models = [("contentBased-random_forest-year",ContentBased,{"features_method":"release_year", "regressor_method":'random_forest'}),
              ("contentBased-lasso_regression-year",ContentBased,{"features_method":"release_year", "regressor_method":'lasso_regression'}),
              ("contentBased-ridge_regression-year",ContentBased,{"features_method":"release_year", "regressor_method":'ridge_regression'}),
              ("contentBased-linear_regression-year",ContentBased,{"features_method":"release_year", "regressor_method":'linear_regression'}),
              ("contentBased-gradient_boosting-year",ContentBased,{"features_method":"release_year", "regressor_method":'gradient_boosting'})]
    split_metrics = ["mae", "rmse"]
    loo_metrics = ["hit_rate"]  
    full_metrics = ["novelty"] 
    random_state = 1
    # Split parameters
    test_size = 0.25  
    # Loo parameters
    top_n_value = 40 
 
 