# local imports
from models import *


class EvalConfig:
    
    models = [
        ("baseline_1", ModelBaseline1, {}),  # model_name, model class, model parameters (dict)
    ]
    split_metrics = ["mae","rmse"]
    loo_metrics = []
    full_metrics = []
    random_state= 42
    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
