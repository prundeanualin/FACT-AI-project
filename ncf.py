from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# For now we skip the sensitive attributes because ncf (imported from recommenders) doesn't by default allow
# for handling/including these. Also, the paper doesn't specify whether they include those sensitive ones in
# the training of the user model. So for now we skip them.

# We have already split the data into valid, test, train

use_test_data = False  # Change this to True when done with hyperparameter-tuning, use False for validation data
model_dir = "ncf_models"
load_and_eval = True
overwrite_test_file_full = False  # Change to False if you don't want to re-create the full test file, change to True
                                  # if using changed dataset

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
                  binary=False  # Set this to True if you want to convert ratings in a binary fashion i.e. set to 1 if
                                # rating greater than 1 set to 0 otherwise
                  )

if not load_and_eval:
    model = NCF(
        n_users=nonsplit_data['UserID'].nunique(),
        n_items=nonsplit_data['MovieID'].nunique(),
        model_type="NeuMF",
        n_factors=8,
        layer_sizes=[64, 32, 16, 8],
        n_epochs=20,
        batch_size=256,
        learning_rate=0.001,
        verbose=1
    )

    model.fit(data)
    model.save(model_dir)
else:
    model = NCF(
        n_users=nonsplit_data['UserID'].nunique(),
        n_items=nonsplit_data['MovieID'].nunique(),
        model_type="NeuMF",
        n_factors=8,
        layer_sizes=[64, 32, 16, 8],
        n_epochs=20,
        batch_size=256,
        learning_rate=0.001,
        verbose=1
    )
    model.load(neumf_dir=model_dir)

# Load the validation set
validation_data = pd.read_csv(base_path + "validation.csv")

# Get user and item inputs for predictions
user_input = validation_data['UserID'].values
item_input = validation_data['MovieID'].values

# Predict ratings
predicted_ratings = model.predict(user_input, item_input, is_list=True)
print(predicted_ratings)

# Actual ratings
actual_ratings = validation_data['Rating'].values
print(actual_ratings)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print("RMSE on Validation Set:", rmse)
