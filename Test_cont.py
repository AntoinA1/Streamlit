import pandas as pd
import streamlit as st
from loaders import load_items, load_ratings
from sklearn.linear_model import LinearRegression

# Chargement des données
movies_df = load_items()
ratings_df = load_ratings(surprise_format=False)  # Sans surprise format pour manipulation

print(movies_df.head())

# One-hot encoding des genres
genres_df = movies_df['genres'].str.get_dummies(sep='|')

# Fusionner les notes avec les genres
ratings_with_genres = ratings_df.merge(genres_df, left_on='movieId', right_index=True)

# Initialiser la matrice de features pour les utilisateurs
user_ids = ratings_df['userId'].unique()
df_features = pd.DataFrame(0, index=user_ids, columns=genres_df.columns)

# Remplir la matrice avec les notes moyennes des genres par utilisateur
for user_id in user_ids:
    user_ratings = ratings_with_genres[ratings_with_genres['userId'] == user_id]
    for genre in genres_df.columns:
        genre_ratings = user_ratings[user_ratings[genre] == 1]['rating']
        if not genre_ratings.empty:
            df_features.at[user_id, genre] = genre_ratings.mean()

# Afficher la matrice one-hot encoding
st.title("Matrice One-Hot Encoding des Genres")
st.write(df_features)

