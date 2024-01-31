import numpy as np
import pandas as pd

from user_models import PMF


base_path = 'data/ml-1m/'
train_path = base_path + "train.csv"
checkpoint_dir = 'pmf_models'
latent_embedding_size = 32
explicit = False  # Set to True for explicit feedback (1 to 5), False for implicit (0 or 1)
threshold = 3  # Threshold model to load used
epoch = 50  # Epoch to load the model on, pick the one with highest validation performance,
# we pick 50 for threshold 3 implicit, 100 for threshold 1 implicit because they had the best valid performance

feedback_type = 'explicit' if explicit else 'implicit'

train = pd.read_csv(train_path)
min_rating, max_rating = train['Rating'].min(), train['Rating'].max()
n_users, n_movies = len(train['UserID'].unique()), len(train['MovieID'].unique())

pmf_model = PMF(n_users, n_movies, latent_vectors=latent_embedding_size, lam_u=0.05, lam_v=0.05, explicit=explicit,
                min_rating=min_rating, max_rating=max_rating)
pmf_model.load_model(f'{checkpoint_dir}/pmf_model_{feedback_type}_emb_{latent_embedding_size}_thresh_{threshold}_epoch_{epoch}.pth')

# Get user embeddings
users = [_ for _ in range(1, 6041)]
user_emb = pmf_model.user_features.detach().numpy()

# Get movie embeddings
unique_movies = sorted(train['MovieID'].unique())
print(unique_movies)

movie_emb = pmf_model.movie_features.detach().numpy()

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
user_emb_csv_path = base_path + f'pmf_user_embs_{latent_embedding_size}_thresh_{threshold}.csv'
movie_emb_csv_path = base_path + f'pmf_movie_embs_{latent_embedding_size}_thresh_{threshold}.csv'

# Save to CSV
user_emb_df.to_csv(user_emb_csv_path, index=False)
movie_emb_mapped.to_csv(movie_emb_csv_path, index=False)
