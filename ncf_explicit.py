import math
import os
from torch.functional import F
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from user_models import NCF


# Function to transform data into DataLoader format for PyTorch
def transform(user_ids, item_ids, score, movie_id_to_index_mapping, batch_size=256):
    # Convert MovieIDs to indices
    item_indices = [movie_id_to_index_mapping[movie_id] for movie_id in item_ids]
    # Adjust user indices to be zero-indexed
    user_indices = user_ids - 1

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(torch.LongTensor(user_indices),
                            torch.LongTensor(item_indices),
                            torch.FloatTensor(score))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Set device to GPU or CPU
if torch.cuda.is_available():
    print("Using cuda")
    device = torch.device("cuda")
else:
    print("Using cpu")
    device = torch.device("cpu")

base_path = 'data/ml-1m/'
train_path = base_path + "train.csv"
checkpoint_dir = 'ncf_models_explicit'
latent_embedding_size = 32
layers = [2*latent_embedding_size, latent_embedding_size, int(latent_embedding_size / 2)]

explicit = True  # Set to True for explicit feedback (1 to 5), False for implicit (0 or 1)
epochs = 200
save_every = 10  # Saves the model during training every save_every epochs
validate_every = 1
batch_size = 1024
print_batches = True
print_n_batches_per_epoch = 5
save_model = True
only_load = False   # Set to False for enabling training loop, True for only loading model
lr = 0.001

print_padding = '-' * 5
feedback_type = 'explicit' if explicit else 'implicit'

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

ratings = pd.read_csv(train_path)
ratings = ratings.drop(['Gender', 'Age', 'Occupation'], axis=1)
if not explicit:
    ratings['Rating'] = (ratings['Rating'] > 1).astype(int)

# Scaling ratings to between 0 and 1, this helps our model by constraining predictions
min_rating, max_rating = ratings['Rating'].min(), ratings['Rating'].max()
ratings['Rating'] = (ratings['Rating'] - min_rating) / (max_rating - min_rating)
n_users, n_movies = len(ratings['UserID'].unique()), len(ratings['MovieID'].unique())

movie_ids = sorted(ratings['MovieID'].unique())
movie_id_to_index_mapping = {}
for i, id in enumerate(movie_ids):
    movie_id_to_index_mapping[id] = i

# Load validation data
val_path = base_path + "validation.csv"
val_ratings = pd.read_csv(val_path)
val_ratings = val_ratings.drop(['Gender', 'Age', 'Occupation'], axis=1)
if not explicit:
    val_ratings['Rating'] = (val_ratings['Rating'] > 1).astype(int)
val_ratings['Rating'] = (val_ratings['Rating'] - min_rating) / (max_rating - min_rating)

train_loader, valid_loader = [transform(
    data["UserID"],
    data["MovieID"],
    data["Rating"],
    movie_id_to_index_mapping,
    batch_size=batch_size
) for data in [ratings, val_ratings]]

ncf_model = NCF(n_users, n_movies, latent_dim=latent_embedding_size, layers=layers,
                min_rating=min_rating, max_rating=max_rating)
optimizer = torch.optim.Adam(ncf_model.parameters(), lr=lr, weight_decay=0.0000001)
n_batches = len(train_loader)

mse, rmse = ncf_model.evaluate(valid_loader=valid_loader, denormalize_predictions=True,
                               denormalize_labels=True)
print(f"{print_padding} Validation MSE: {mse:.6f}, "
      f"Validation RMSE: {rmse:.6f} {print_padding}")

if not only_load:
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            user_indices, movie_indices, ratings = batch
            user_indices, movie_indices, ratings = user_indices.to(device), movie_indices.to(device), ratings.to(device)

            optimizer.zero_grad()
            predictions = ncf_model(user_indices, movie_indices)
            loss = F.mse_loss(predictions.squeeze(), ratings)
            loss.backward()
            optimizer.step()

            if print_batches and batch_id % math.floor(n_batches / print_n_batches_per_epoch) == 0:
                print(f"Epoch {epoch}, Batch {batch_id}, Loss: {loss.item():.6f}")

            total_loss += loss

        print(f"{print_padding} Epoch {epoch}, Average Loss: {total_loss / (batch_id + 1):.6f} {print_padding}")

        if epoch % validate_every == 0:
            mse, rmse = ncf_model.evaluate(valid_loader=valid_loader, denormalize_predictions=True,
                                           denormalize_labels=True)
            print(f"{print_padding} Validation MSE: {mse:.6f}, "
                  f"Validation RMSE: {rmse:.6f} {print_padding}")

        if save_model and epoch % save_every == 0:
            checkpoint_path = f'{checkpoint_dir}/ncf_model_{feedback_type}_emb_{latent_embedding_size}_epoch_{epoch}.pth'
            ncf_model.save_model(checkpoint_path)
            print(f"{print_padding} Model saved at epoch {epoch} to {checkpoint_path} {print_padding}")
        print()

# Load the final model for prediction
epoch = 10  # Specify which epoch model to load
ncf_model.load_model(f'{checkpoint_dir}/ncf_model_{feedback_type}_emb_{latent_embedding_size}_epoch_{epoch}.pth')
mse, rmse = ncf_model.evaluate(valid_loader=valid_loader, denormalize_predictions=True,
                               denormalize_labels=True)
print(f"{print_padding} Validation MSE: {mse:.6f}, "
      f"Validation RMSE: {rmse:.6f} {print_padding}")
