import argparse
import os
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
from utils_ml import load_ncf_model


# Arguments
args = argparse.ArgumentParser()
# Rating threshold for binarizing data, default=1, only for implicit feedback
args.add_argument("-BINARIZATION_THRESHOLD", default=1, type=int)
args = args.parse_args()

n_epochs = 20
model_type = 'mlp'
use_test_data = False  # Change this to True when done with hyperparameter-tuning, use False for validation data
only_load = False  # Set to False for training, True for loading the model only
overwrite_test_file_full = True  # Change to False if you don't want to re-create the full test file, change to True
# if using changed dataset
binary = False  # Set this to True if you want to convert ratings in a binary fashion using recommenders package i.e.
# set to 1 if rating greater than 1 set to 0 otherwise
manual_thresholding = not binary  # If not using recommenders package for binarizing we will binary based on
# threshold manually
threshold = args.BINARIZATION_THRESHOLD
model_dir = f"ncf_models/thresh_{threshold}_epoch_{n_epochs}"

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

base_path = "data/ml-1m/"
test_valid_file = base_path + "test.csv" if use_test_data else base_path + "validation.csv"
train_file = base_path + "train.csv"

if manual_thresholding:
    train = pd.read_csv(train_file)
    train['Rating'] = (train['Rating'] > threshold).astype(int)

    valid = pd.read_csv(test_valid_file)
    valid['Rating'] = (valid['Rating'] > threshold).astype(int)

    train_file = f"{base_path}train_implicit_ncf_thresh_{threshold}.csv"
    train.to_csv(train_file, index=False)

    test_valid_file = f"{base_path}valid_implicit_ncf_thresh_{threshold}.csv"
    valid.to_csv(test_valid_file, index=False)


nonsplit_data = pd.read_csv(base_path + "merged_dataset.csv")

data = NCFDataset(train_file=train_file,
                  test_file=test_valid_file,
                  col_user='UserID',
                  col_item='MovieID',
                  col_rating='Rating',
                  overwrite_test_file_full=overwrite_test_file_full,  # Set this to true if you want to recreate the
                  # test_full.csv / validation_full.csv file
                  binary=binary
                  )

if not only_load:
    model = NCF(
        n_users=nonsplit_data['UserID'].nunique(),
        n_items=nonsplit_data['MovieID'].nunique(),
        model_type=model_type,
        n_factors=8,
        layer_sizes=[64, 32, 16, 8],
        n_epochs=n_epochs,
        batch_size=256,
        learning_rate=0.001,
        verbose=1
    )

    model.fit(data)
    model.save(model_dir)

    user2id = model.user2id
    item2id = model.item2id
    id2user = model.id2user
    id2item = model.id2item

    # Save the dictionaries to a file
    with open(model_dir + '/user2id.pkl', 'wb') as f:
        pickle.dump(model.user2id, f)
    with open(model_dir + '/item2id.pkl', 'wb') as f:
        pickle.dump(model.item2id, f)
    with open(model_dir + '/id2user.pkl', 'wb') as f:
        pickle.dump(model.id2user, f)
    with open(model_dir + '/id2item.pkl', 'wb') as f:
        pickle.dump(model.id2item, f)
else:
    model = load_ncf_model(model_dir, base_path)

# Load the validation set
validation_data = pd.read_csv(test_valid_file)

# Get user and item inputs for predictions
user_input = validation_data['UserID'].values
item_input = validation_data['MovieID'].values

# Predict ratings
predicted_ratings = model.predict(user_input, item_input, is_list=True)
# print(predicted_ratings)

# Actual ratings
actual_ratings = validation_data['Rating'].values
if binary:
    actual_ratings = (actual_ratings > 1).astype(int)
# print(actual_ratings)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print("RMSE on Validation Set:", rmse)

# users = [_ for _ in range(1, 6041)]
#
# # Get user embeddings
# user_emb = get_user_embeddings(model, users)
#
# unique_movies = nonsplit_data['MovieID'].unique()
#
# movie_emb = get_movie_embeddings(model, unique_movies)


