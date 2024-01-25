from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import pickle


# For now we skip the sensitive attributes because ncf (imported from recommenders) doesn't by default allow
# for handling/including these. Also, the paper doesn't specify whether they include those sensitive ones in
# the training of the user model. So for now we skip them.

# We have already split the data into valid, test, train


def load_ncf_model(model_dir, base_path, model_type='mlp', n_factors=8, layer_sizes=[64, 32, 16, 8], n_epochs=20,
                   batch_size=256, learning_rate=0.001, verbose=1):
    nonsplit_data = pd.read_csv(base_path + "merged_dataset.csv")

    model = NCF(
        n_users=nonsplit_data['UserID'].nunique(),
        n_items=nonsplit_data['MovieID'].nunique(),
        model_type=model_type,
        n_factors=n_factors,
        layer_sizes=layer_sizes,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose
    )

    if model_type == 'mlp':
        model.load(mlp_dir=model_dir)

    # Load the dictionaries from the file
    with open(model_dir + '/user2id.pkl', 'rb') as f:
        model.user2id = pickle.load(f)
    with open(model_dir + '/item2id.pkl', 'rb') as f:
        model.item2id = pickle.load(f)
    with open(model_dir + '/id2user.pkl', 'rb') as f:
        model.id2user = pickle.load(f)
    with open(model_dir + '/id2item.pkl', 'rb') as f:
        model.id2item = pickle.load(f)
    return model


n_epochs = 20
model_type = 'mlp'
use_test_data = False  # Change this to True when done with hyperparameter-tuning, use False for validation data
model_dir = "ncf_models"
load_and_eval = True
overwrite_test_file_full = False  # Change to False if you don't want to re-create the full test file, change to True
# if using changed dataset
binary = True  # Set this to True if you want to convert ratings in a binary fashion i.e. set to 1 if
# rating greater than 1 set to 0 otherwise

base_path = "data/ml-1m/"
test_valid_file = base_path + "test.csv" if use_test_data else base_path + "validation.csv"
train_file = base_path + "train.csv"

# base_path = "neural_collaborative_filtering-master/Data/"
# train_file = base_path + "ml-1m.train.rating"
# test_valid_file = base_path + "ml-1m.test.rating"

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

if not load_and_eval:
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
validation_data = pd.read_csv(base_path + "validation.csv")

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


def get_user_embeddings(model, user_ids):
    user_input = np.array([model.user2id[x] for x in user_ids])
    user_input_tensor = tf.convert_to_tensor(user_input, dtype=tf.int32)

    # Get user embeddings
    user_emb = tf.nn.embedding_lookup(params=model.embedding_mlp_P, ids=user_input_tensor)

    # Compute and retrieve the values of user embeddings
    with model.sess.as_default():
        user_emb_evaluated = user_emb.eval()

    return user_emb_evaluated


def get_movie_embeddings(model, movie_ids):
    item_input = np.array([model.item2id[x] for x in movie_ids])
    item_input_tensor = tf.convert_to_tensor(item_input, dtype=tf.int32)

    # Get user embeddings
    movie_embs = tf.nn.embedding_lookup(params=model.embedding_mlp_Q, ids=item_input_tensor)

    # Compute and retrieve the values of movie embeddings
    with model.sess.as_default():
        movie_embs = movie_embs.eval()

    return movie_embs


# users = [_ for _ in range(1, 6041)]
#
# # Get user embeddings
# user_emb = get_user_embeddings(model, users)
#
# unique_movies = nonsplit_data['MovieID'].unique()
#
# movie_emb = get_movie_embeddings(model, unique_movies)


