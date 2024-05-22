# local imports
from models import *
from surprise import accuracy

class EvalConfig:
   
    models = [("ContentBased", ContentBased, {'features_method': 'release_year' , "regressor_method": 'ridge_regressor'} )]
    split_metrics = ["mae", "rmse"]
    loo_metrics = ["hit_rate"]  # Add "hit rate"
    full_metrics = ["novelty"]  # Add "novelty" 
    random_state = 1
    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --
 
    # Loo parameters
    top_n_value = 40  # -- configure the number of recommendations (> 1) --
