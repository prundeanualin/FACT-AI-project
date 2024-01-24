import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Sample code to load data
import pandas as pd

use_test_data = False  # Change this to True when done with hyperparameter-tuning, use False for validation data

base_path = "data/ml-1m/"
test_valid_file = base_path + "test.csv" if use_test_data else base_path + "validation.csv"
train_data = base_path + "train.csv"

# Assuming your data is in a CSV file
file_path = 'your_data.csv'

non_split_data = pd.read_csv(base_path + "merged_dataset.csv")

# Assume you have a user-item matrix with rows as UserIDs, columns as MovieIDs, and values as Ratings
# You may need to convert your data into this format
user_item_matrix = non_split_data.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

# Convert the user-item matrix to PyTorch tensor
user_item_tensor = torch.tensor(user_item_matrix.values, dtype=torch.float32)

# Load the validation set
val_data = pd.read_csv(base_path + "validation.csv")

# Get user and item inputs for predictions
user_input = val_data['UserID'].values
item_input = val_data['MovieID'].values

# Define the LightGCN model
class LightGCN(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim):
        super(LightGCN, self).__init__()
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        # LightGCN does not use any additional layers, just element-wise product
        interaction = torch.mul(user_emb, item_emb)

        return interaction

# Initialize the model
num_users, num_items = user_item_tensor.shape
embedding_dim = 64  # You can adjust this based on your preference
model = LightGCN(num_users, num_items, embedding_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for user, item in tqdm(train_data.nonzero()):
        rating = train_data[user, item]

        user = torch.tensor(user)
        item = torch.tensor(item)

        # Forward pass
        prediction = model(user, item)

        # Compute the loss
        loss = criterion(prediction, rating)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(*val_data.nonzero())
        val_loss = criterion(val_predictions, val_data[val_data.nonzero()])

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Test the model
# model.eval()
# with torch.no_grad():
#     test_predictions = model(*test_data.nonzero())
#     test_loss = criterion(test_predictions, test_data[test_data.nonzero()])
#
# print(f'Test Loss: {test_loss.item():.4f}')