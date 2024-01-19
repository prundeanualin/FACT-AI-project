from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import pickle
from ncf import load_ncf_model, get_user_embeddings


n_epochs = 20
model_type = 'mlp'
model_dir = "ncf_models"
base_path = "data/ml-1m/"
binary = True  # Set to True if the ratings were converted to binary during training (yes)

model = load_ncf_model(model_dir, base_path)

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
if binary:
    actual_ratings = (actual_ratings > 1).astype(int)
print(actual_ratings)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print("RMSE on Validation Set:", rmse)

users = [1, 2, 3]

# Get user embeddings
user_emb = get_user_embeddings(model, users)

print(user_emb)

