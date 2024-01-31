import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from user_models import PMF
import os


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


base_path = 'data/ml-1m/'
train_path = base_path + "train.csv"
checkpoint_dir = 'pmf_models'
explicit = True  # Set to True for explicit feedback (1 to 5), False for implicit (0 or 1)
if explicit:
    latent_embedding_size = 16
else:
    latent_embedding_size = 32
epochs = 1000
save_model = True  # Saves the model during training every save_every epochs
save_every = 100
eval_every = 100
only_load = False  # Set to False for enabling training loop, True for only loading model / saving embeddings
batch_size = 1024
threshold = 1  # Rating threshold for binarizing data, default=1, only for implicit feedback

feedback_type = 'explicit' if explicit else 'implicit'
if explicit:
    checkpoint_path_start = f'{checkpoint_dir}/pmf_model_{feedback_type}_emb_{latent_embedding_size}_epoch_' + "{}.pth"
else:
    checkpoint_path_start = f'{checkpoint_dir}/pmf_model_{feedback_type}_emb_{latent_embedding_size}_thresh_{threshold}_epoch_' + "{}.pth"

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load and preprocess rating data
ratings = pd.read_csv(train_path)
ratings = ratings.drop(['Gender', 'Age', 'Occupation'], axis=1)
if not explicit:
    ratings['Rating'] = (ratings['Rating'] > threshold).astype(int)
min_rating, max_rating = ratings['Rating'].min(), ratings['Rating'].max()
n_users, n_movies = len(ratings['UserID'].unique()), len(ratings['MovieID'].unique())

if not only_load:
    rating_matrix_df = ratings.pivot(index='UserID', columns='MovieID', values='Rating')

    # Scaling ratings to between 0 and 1, this helps our model by constraining predictions
    rating_matrix_df = (rating_matrix_df - min_rating) / (max_rating - min_rating)

    # Replacing missing ratings with -1 so we can filter them out later
    rating_matrix_df[rating_matrix_df.isnull()] = -1
    rating_matrix = torch.FloatTensor(rating_matrix_df.values)

pmf_model = PMF(n_users, n_movies, latent_vectors=latent_embedding_size, lam_u=0.05, lam_v=0.05, explicit=explicit,
                min_rating=min_rating, max_rating=max_rating)
optimizer = torch.optim.Adam([pmf_model.user_features, pmf_model.movie_features], lr=0.01)

# Load validation data
val_path = base_path + "validation.csv"
val_ratings = pd.read_csv(val_path)
val_ratings = val_ratings.drop(['Gender', 'Age', 'Occupation'], axis=1)
if not explicit:
    val_ratings['Rating'] = (val_ratings['Rating'] > threshold).astype(int)

movie_ids_sorted = sorted(ratings['MovieID'].unique())
movie_id_to_index_mapping = {}
for i, id in enumerate(movie_ids_sorted):
    movie_id_to_index_mapping[id] = i

valid_loader = transform(
    val_ratings["UserID"],
    val_ratings["MovieID"],
    val_ratings["Rating"],
    movie_id_to_index_mapping,
    batch_size=batch_size
)

if not only_load:
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = pmf_model(rating_matrix)
        loss.backward()
        optimizer.step()
        if epoch % eval_every == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.3f}")
            if explicit:
                rmse, mae = pmf_model.evaluate(valid_loader=valid_loader, denormalize_predictions=True,
                                               denormalize_labels=False)
                print("Validation RMSE: ", rmse, "Validation MAE: ", mae)
            else:
                bce, auc_roc = pmf_model.evaluate(valid_loader=valid_loader, denormalize_predictions=False,
                                                  denormalize_labels=False)
                print("Validation BCE: ", bce, "Validation AUC-ROC: ", auc_roc)

        if save_model and epoch % save_every == 0:
            checkpoint_path = checkpoint_path_start.format(epoch)
            pmf_model.save_model(checkpoint_path)
            print(f"Model saved at epoch {epoch} to {checkpoint_path}")

models = [200, 400, 600, 800, 1000]

for epoch in models:
    # Load the final model for prediction (or choose a specific checkpoint)
    pmf_model.load_model(checkpoint_path_start.format(epoch))

    if explicit:
        rmse, mae = pmf_model.evaluate(valid_loader=valid_loader, denormalize_predictions=True,
                                       denormalize_labels=False)
        print("Validation RMSE: ", rmse, "Validation MAE: ", mae)
    else:
        bce, auc_roc = pmf_model.evaluate(valid_loader=valid_loader, denormalize_predictions=False,
                                          denormalize_labels=False)
        print("Validation BCE: ", bce, "Validation AUC-ROC: ", auc_roc)

# # Making predictions for a specific user
# user_idx = 7
# predicted_ratings, actual_ratings = pmf_model.predict(user_idx, rating_matrix)
# print("Predictions: \n", predicted_ratings.detach().numpy())
# unique_elements, counts = np.unique(predicted_ratings.detach().numpy(), return_counts=True)
#
# # Printing the counts of each unique element
# print("Counts: ", counts)
# print("Truth: \n", actual_ratings.detach().numpy())
#
# movie_ids_sorted = sorted(ratings['MovieID'].unique())
# movie_id_to_index_mapping = {}
# for i, id in enumerate(movie_ids_sorted):
#     movie_id_to_index_mapping[id] = i
#
# movie_ids = ratings[ratings['UserID'] == user_idx + 1]['MovieID'].to_numpy()
# movie_indices = sorted([movie_id_to_index_mapping[movie_id] for movie_id in movie_ids])
# user_embeddings, movie_embeddings = pmf_model.get_embeddings([user_idx for _ in range(len(movie_indices))], movie_indices)
# preds = pmf_model.predict_batch_with_embeddings(user_embeddings, movie_embeddings)
# print(preds)
#
# res = []
# for i in range(len(movie_ids)):
#     pred = pmf_model.predict_single_with_embeddings(user_embeddings[i], movie_embeddings[i])
#     res.append(pred.item())
#
# unique_elements_res, counts_res = np.unique(res, return_counts=True)
#
# # Printing the counts of each unique element in 'res'
# print("Counts: ", counts_res)
# print(res)

# # Making predictions for a batch of users
# user_indices = [0, 1]
# predicted_ratings, actual_ratings = pmf_model.predict_batch(user_indices, rating_matrix)
#
# for i in range(len(user_indices)):
#     print(len(predicted_ratings[i]))
#     print("Predictions: \n", predicted_ratings[i])
#     print("Truth: \n", actual_ratings[i])
#
# # Making predictions for a batch of users
# user_indices = [1, 2]
# predicted_ratings, actual_ratings = pmf_model.predict_batch(user_indices, val_rating_matrix)
#
# for i in range(len(user_indices)):
#     print(len(predicted_ratings[i]))
#     print("Predictions: \n", predicted_ratings[i])
#     print("Truth: \n", actual_ratings[i])
#     print(val_ratings[val_ratings['UserID'] == user_indices[i] + 1])
#
# user_indices = [1, 2]
# movie_indices = [1245, 647]
# user_embs, movie_embs = pmf_model.get_embeddings(user_indices, movie_indices)
# print(pmf_model.predict_batch_with_embeddings(user_embs, movie_embs))
#
