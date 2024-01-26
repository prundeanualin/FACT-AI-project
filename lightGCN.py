import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.deeprec_utils import HParams
from tqdm import tqdm

# Sample code to load data
import pandas as pd

use_test_data = False  # Change this to True when done with hyperparameter-tuning, use False for validation data

base_path = "data/ml-1m/"
test_file = base_path + "test.csv"
validation_file = base_path + "validation.csv"
train_file = base_path + "train.csv"
base_model_path = 'data/ml-1m/lightgcn/'

print("Loading the datasets...")
renaming_columns = {
    "UserID": "userID",
    "MovieID": "itemID",
    "Rating": "rating"
}
train_df = pd.read_csv(train_file)
print("Length of the training set: ", len(train_df))
train_df.rename(columns=renaming_columns, inplace=True)
validation_df = pd.read_csv(validation_file)
validation_df.rename(columns=renaming_columns, inplace=True)
test_df = pd.read_csv(test_file)
test_df.rename(columns=renaming_columns, inplace=True)

seed = 42

data_model = ImplicitCF(train_df, validation_df, seed=seed)

# self.epochs = hparams.epochs
# self.lr = hparams.learning_rate
# self.emb_dim = hparams.embed_size
# self.batch_size = hparams.batch_size
# self.n_layers = hparams.n_layers
# self.decay = hparams.decay
# self.eval_epoch = hparams.eval_epoch
# self.top_k = hparams.top_k
# self.save_model = hparams.save_model
# self.save_epoch = hparams.save_epoch
# self.metrics = hparams.metrics
# self.model_dir = hparams.MODEL_DIR

epochs = 1000
save_epoch = 5
eval_epoch = 2
learning_rate = 0.001
embed_size = 64
batch_size = 2048
n_layers = 3
decay = 1e-4
top_k = 20
save_model = True
metrics = ['precision'] # metric_options = ["map", "ndcg", "precision", "recall"]
MODEL_DIR = base_model_path

hparams = {
    'epochs': epochs,
    'learning_rate': learning_rate,
    'embed_size': embed_size,
    'batch_size': batch_size,
    'n_layers': n_layers,
    'decay': decay,
    'eval_epoch': eval_epoch,
    'top_k': top_k,
    'save_model': save_model,
    'save_epoch': save_epoch,
    'metrics': metrics,
    'MODEL_DIR': MODEL_DIR
}

user_embedding_file = base_model_path + 'user_embedding.csv'
item_embedding_file = base_model_path + 'item_embedding.csv'

model = LightGCN(hparams=HParams(hparams), data=data_model, seed=seed)
print("Training the model...")
model.fit()
model.infer_embedding(user_file=user_embedding_file, item_file=item_embedding_file)
