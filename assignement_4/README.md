![Ajout d'image d'en-tête](https://www.freecodecamp.org/news/content/images/size/w2000/2023/11/movie-recommendation.png)

# Group 06 - Movie Recommender System, Coding 1

This file is the fourth step for the project in the recommender system's course. The purpose is to understand and to be able to create create recommandations based on the content with some features of the content but also some method of regression. The goal is to find the best system. In this project we have tested as features the length of title, the year of release and the genres of movies. And as models of regressor, the Ridge regressor, the lasso regressor, the linear regressor, the random forest regressor and the gradient voosting regressor. And we find that the most performant one is the ridge regressor with thr year of release.
For this we also used the part of evaluator to find the better system. This differents code make an output of a report of the evalutation that is in the data file in data/tiny/evaluations.

## Installation 

Make sure you install those libraries : 
- Numpy
- Pandas
- Seaborn
- Surprise
- scikit-learn

## Data 

Make sure you download the MovieLens dataset from the course's drive.

## Credits 

For the definition get_top_N, the lagorythm has been inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py.