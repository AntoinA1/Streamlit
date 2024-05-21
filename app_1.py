import streamlit as st
from fuzzywuzzy import process
from loaders import load_items

# Définition de la fonction pour rechercher des films avec tolérance aux fautes d'orthographe
def search_movie(query, choices, limit=5):
    results = process.extract(query, choices, limit=limit)
    return [result[0] for result in results]

movies_df = load_items()

# Titre de l'application
st.title("Recommender System, Group 6")

# Section pour l'insertion du nom d'utilisateur
username = st.text_input("User Name: ")

# Initialisation de la structure de données pour stocker les notations
ratings = {}

# Champ de saisie pour rechercher un film
st.header("Rechercher un film")
movie_query = st.text_input("Entrez le nom d'un film")

# Recherche de films basée sur la saisie de l'utilisateur
if movie_query:
    movie_choices = movies_df['title'].tolist()
    search_results = search_movie(movie_query, movie_choices)
    
    # Liste déroulante pour sélectionner un film parmi les suggestions
    selected_movie = st.selectbox("Sélectionnez un film :", search_results)
    
    # Permettre à l'utilisateur de noter le film sélectionné et stocker les notations dans la structure de données
    if selected_movie:
        rating = st.slider(f"Notez le film {selected_movie}", 0, 5, 3)
        ratings[selected_movie] = rating

# Afficher les notations actuelles pour tous les films
st.header("Vos notations actuelles")
for movie, rating in ratings.items():
    st.write(f"{movie} : {rating}")

# Bouton pour enregistrer les notations
if st.button("Enregistrer mes notations"):
    # Enregistrer les notations dans une base de données ou un fichier, par exemple
    # Vous pouvez ajouter le code pour cette étape ici
    st.success("Notations enregistrées avec succès !")
