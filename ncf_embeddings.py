import numpy as np
import pandas as pd
from utils_ml import load_ncf_model, ncf_get_movie_embeddings, ncf_get_user_embeddings


base_path = "data/ml-1m/"
model_dir = "ncf_models/"
epoch = 20
threshold = 3
model_path = f"{model_dir}thresh_{threshold}_epoch_{epoch}"

model = load_ncf_model(model_path, base_path)

# Load the validation set
train = pd.read_csv(base_path + "train.csv")

# Get user embeddings
users = [_ for _ in range(1, 6041)]
user_emb = ncf_get_user_embeddings(model, users)

# Get movie embeddings
unique_movies = train['MovieID'].unique()
movie_emb = ncf_get_movie_embeddings(model, unique_movies)

# Convert user_emb to a DataFrame
user_emb_df = pd.DataFrame(user_emb, columns=[f'user_emb_{i}' for i in range(user_emb.shape[1])])

# UserID in user_emb_df should start from 1
user_emb_df['UserID'] = np.arange(1, len(user_emb_df) + 1)

# Reset the index of the DataFrame
user_emb_df.reset_index(drop=True, inplace=True)

# Reorder columns to have UserID as the first column
cols = user_emb_df.columns.tolist()
cols = cols[-1:] + cols[:-1]  # Move the last column (UserID) to the first position
user_emb_df = user_emb_df[cols]

# Create a DataFrame from movie_emb
movie_emb_df = pd.DataFrame(movie_emb, columns=[f'movie_emb_{i}' for i in range(movie_emb.shape[1])])

# Create a DataFrame from unique_movies
unique_movies_df = pd.DataFrame(unique_movies, columns=['MovieID'])

# Now associate each MovieID with its embedding
movie_emb_mapped = unique_movies_df.join(movie_emb_df)

print(user_emb_df.head())
print(movie_emb_mapped.head())

# Paths for saving CSV files
user_emb_csv_path = f'{base_path}ncf_user_embs_32_thresh_{threshold}.csv'
movie_emb_csv_path = f'{base_path}ncf_movie_embs_32_thresh_{threshold}.csv'

# Save to CSV
user_emb_df.to_csv(user_emb_csv_path, index=False)
movie_emb_mapped.to_csv(movie_emb_csv_path, index=False)
