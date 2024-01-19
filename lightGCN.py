import pandas as pd
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.models.deeprec.deeprec_utils import HParams

use_test_data = False  # Change this to True when done with hyperparameter-tuning, use False for validation data

base_path = "data/ml-1m/"
test_valid_file = base_path + "test.csv" if use_test_data else base_path + "validation.csv"
train_file = base_path + "train.csv"
model_dir = 'lightGCN_models'

train = pd.read_csv(train_file)
test = pd.read_csv(test_valid_file)

# Convert 'Rating' column to binary
train['Rating'] = (train['Rating'] > 1).astype(int)
test['Rating'] = (test['Rating'] > 1).astype(int)

data_object = ImplicitCF(train, test=test, adj_dir=None, col_user='UserID', col_item='MovieID', col_rating='Rating',
                         col_prediction='prediction', seed=None)

hparams_dict = {
    'learning_rate': 0.001,
    'embed_size': 8,        # Embedding size, e.g. 64 or 128.
    'batch_size': 256,
    'n_layers': 3,           # Number of GCN layers; often 2 or 3.
    'decay': 0.0001,         # Regularization term to prevent overfitting.
    'eval_epoch': 5,         # Frequency of evaluation during training.
    'top_k': 10,             # Number of top recommendations to consider.
    'save_model': True,      # Whether to save the model after training.
    'save_epoch': 5,         # Frequency of saving the model.
    'metrics': ['ndcg', 'precision', 'recall', 'map'],  # Evaluation metrics.
    'MODEL_DIR': model_dir,  # Directory to save the model.
    'epochs': 50             # Total number of training epochs.
}

hparams_object = HParams(hparams_dict=hparams_dict)

model = LightGCN(data=data_object, hparams=hparams_object)

model.fit()
