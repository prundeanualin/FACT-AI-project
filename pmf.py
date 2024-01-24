# Prepare analysis
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

from movie_lens_data import MovieLensData

# Read the data
path = "/kaggle/input/movielens-100k-dataset/ml-100k"
movie_lens_data = MovieLensData(
    users_path = os.path.join(path, "u.user"),
    ratings_path = os.path.join(path, "u.data"),
    movies_path = os.path.join(path, "u.item"),
    genre_path = os.path.join(path, "u.genre")
    )

evaluation_data = movie_lens_data.read_ratings_data()
movie_data = movie_lens_data.read_movies_data()
popularity_rankings = movie_lens_data.get_popularity_ranks()
ratings = movie_lens_data.get_ratings()