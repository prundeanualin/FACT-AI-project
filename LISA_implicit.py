import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
from model.fairlisa_models import Filter, Discriminator
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
from utils import seed_experiments

# Arguments
args = argparse.ArgumentParser()
args.add_argument("-DATA_BASE_PATH", default="data/ml-1m/", type=str)
args.add_argument(
    "-SENSITIVE_FEATURES",
    default=["Gender", "Age", "Occupation"],
    type=list,
)
args.add_argument("-LAMBDA_1", default=1, type=float)
args.add_argument("-LAMBDA_2", default=20, type=float)
args.add_argument("-LAMBDA_3", default=10, type=float)
args.add_argument("-CUDA", default=2, type=int)
args.add_argument("-SEED", default=4869, type=int)
args.add_argument("-BATCH_SIZE", default=8192, type=int)
args.add_argument("-N_EPOCHS", default=10, type=int)
args.add_argument("-EPOCHS_DISCRIMINATOR", default=10, type=int)
args.add_argument("-EPOCHS_ATTACKER", default=50, type=int)
args.add_argument("-FILTER_LAYER_SIZES", default="16,16", type=lambda s: [int(layer_size) for layer_size in s.split(',')])
args.add_argument("-MODEL", default="NCF", type=str)
args.add_argument("-LR", default=0.001, type=float)
args.add_argument("-LR_DISC", default=0.01, type=float)
args.add_argument("-DISCR_LATENT", default=16, type=int)
args.add_argument("-USE_NOFEATURE", default=False, type=bool)
args.add_argument("-MISSING_RATIO", default=0.2, type=float)
args.add_argument("-DEVICE", default='cuda', type=str)
args.add_argument("-USE_TEST_DATA", default=False, type=bool)
# Specify the file name for saving results
args.add_argument("-RESULTS_FILENAME", default="ml_explicit_missing_ratios_results.txt", type=str)
args.add_argument("-SAVE_RES_TO_TXT", default=False, type=str)
args.add_argument("-ORIGIN", default=False, type=bool)
args = args.parse_args()

seed_experiments(args.SEED)
model_name = args.MODEL
print(args)
print(f"Model: {model_name}")

# Set device to GPU or CPU
if args.DEVICE == 'cuda' and torch.cuda.is_available():
    print("Using cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA)
    device = torch.device("cuda")
else:
    print("Using cpu")
    device = torch.device("cpu")

# Loading data
print("Loading data...")
train = pd.read_csv(args.DATA_BASE_PATH + "train.csv")
valid = pd.read_csv(args.DATA_BASE_PATH + "validation.csv")
test = pd.read_csv(args.DATA_BASE_PATH + "test.csv")
attacker_train = pd.read_csv(args.DATA_BASE_PATH + "attacker_train.csv")
attacker_test = pd.read_csv(args.DATA_BASE_PATH + "attacker_test.csv")

user_embeddings = pd.read_csv(args.DATA_BASE_PATH + f"{args.MODEL.lower()}_user_embs.csv")
movie_embeddings = pd.read_csv(args.DATA_BASE_PATH + f"{args.MODEL.lower()}_movie_embs.csv")

embedding_dim = len(user_embeddings.columns) - 1

# Convert ratings to binary here
train['Rating'] = (train['Rating'] > 1).astype(int)
valid['Rating'] = (valid['Rating'] > 1).astype(int)
test['Rating'] = (test['Rating'] > 1).astype(int)

# Merge the user_embeddings with train, test, and valid DataFrames on 'UserID'
print("Merging tables...")
train = train.merge(user_embeddings, on='UserID', how='left')
test = test.merge(user_embeddings, on='UserID', how='left')
valid = valid.merge(user_embeddings, on='UserID', how='left')
attacker_train = attacker_train.merge(user_embeddings, on='UserID', how='left')
attacker_test = attacker_test.merge(user_embeddings, on='UserID', how='left')

# Merge the movie_embeddings with train, test, and valid DataFrames on 'MovieID'
train = train.merge(movie_embeddings, on='MovieID', how='left')
test = test.merge(movie_embeddings, on='MovieID', how='left')
valid = valid.merge(movie_embeddings, on='MovieID', how='left')


def split_dataset(data, missing_ratio, seed):
    """
    Split the dataset based on a specified MISSING_RATIO with a seed for reproducibility.

    Args:
    data (DataFrame): The original dataset.
    missing_ratio (float): The ratio of data to be placed in train_data_nofeature.
    seed (int): The seed for the random number generator.

    Returns:
    tuple: Two DataFrames, train_data and train_data_nofeature.
    """

    # Calculate the number of records needed for the desired ratio
    target_record_count = int(len(data) * (1 - missing_ratio))

    # Get unique user IDs
    unique_user_ids = data['UserID'].unique()

    # Shuffle the user IDs
    np.random.shuffle(unique_user_ids)

    # Initialize variables to keep track of selected users and record count
    selected_users = []
    selected_record_count = 0

    # Select users until the target record count is reached
    for user_id in unique_user_ids:
        user_record_count = len(data[data['UserID'] == user_id])
        if selected_record_count + user_record_count > target_record_count:
            break
        selected_users.append(user_id)
        selected_record_count += user_record_count

    # Split the dataset
    train_data = data[data['UserID'].isin(selected_users)]
    train_data_nofeature = data[~data['UserID'].isin(selected_users)]

    return train_data, train_data_nofeature


def attacker_transform(user, gender, age, occupation, user_emb_data):
    dataset = TensorDataset(
        torch.tensor(np.array(user), dtype=torch.int64),
        torch.tensor(np.array(gender), dtype=torch.long),
        torch.tensor(np.array(age), dtype=torch.long),
        torch.tensor(np.array(occupation), dtype=torch.long),
        torch.tensor(np.array(user_emb_data), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)


def transform(user, item, score, gender, age, occupation, user_emb_data, movie_emb_data, nofeature=False):
    if not nofeature:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
            torch.tensor(np.array(gender), dtype=torch.long),
            torch.tensor(np.array(age), dtype=torch.long),
            torch.tensor(np.array(occupation), dtype=torch.long),
            torch.tensor(np.array(user_emb_data), dtype=torch.float32),
            torch.tensor(np.array(movie_emb_data), dtype=torch.float32),
        )
    else:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
            torch.tensor(np.array(user_emb_data), dtype=torch.float32),
            torch.tensor(np.array(movie_emb_data), dtype=torch.float32),
        )
    return DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)


def calculate_auc_score_for_feature(feature_true, feature_pred):
    n_classes = len(np.unique(feature_true))

    # Binarize the true labels for OvR calculation
    true_binarized = label_binarize(feature_true, classes=np.arange(n_classes))

    # Predictions should be an array with shape [n_samples, n_classes]
    pred = np.array(feature_pred)

    if n_classes == 2:
        # Binary classification
        return roc_auc_score(true_binarized, pred[:, 1])
    else:
        # Multi-class classification
        return roc_auc_score(true_binarized, pred, multi_class='ovr')


print("Splitting train data based on missing ratio...")
train, train_nofeature = split_dataset(train, args.MISSING_RATIO, args.SEED)

# Calculate and print percentages
total_records = len(train) + len(train_nofeature)
total_users = len(train['UserID'].unique()) + len(train_nofeature['UserID'].unique())

print("train_data: {:.2f}% of total records, {:.2f}% of total users".format(
    len(train) / total_records * 100, len(train['UserID'].unique()) / total_users * 100))
print("train_data_nofeature: {:.2f}% of total records, {:.2f}% of total users".format(
    len(train_nofeature) / total_records * 100, len(train_nofeature['UserID'].unique()) / total_users * 100))

# Define number of classes to predict for sensitive features (e.g. 2 for gender)
num_classes = {feature: len(train[feature].unique()) for feature in args.SENSITIVE_FEATURES}

# Get DataLoaders
print("Getting DataLoaders...")
train, train_nofeature, valid, test = [
    transform(
        data["UserID"],
        data["MovieID"],
        data["Rating"],
        data.filter(like='Gender'),
        data.filter(like='Age'),
        data.filter(like='Occupation'),
        data.filter(like='user_emb'),
        data.filter(like='movie_emb'),
        nofeature=no_feature
    ) for data, no_feature in [(train, False), (train_nofeature, True), (valid, True), (test, True)]]
attacker_train, attacker_test = [
    attacker_transform(
        data["UserID"],
        data.filter(like='Gender'),
        data.filter(like='Age'),
        data.filter(like='Occupation'),
        data.filter(like='user_emb'),
    ) for data in [attacker_train, attacker_test]
]

# Initialize Filter
filter_model = Filter(embedding_dim, dense_layer_dim=args.FILTER_LAYER_SIZES[0], device=device)

# Initialize discriminators
discriminators = {}
for feature in args.SENSITIVE_FEATURES:
    discriminator_model = Discriminator(embed_dim=embedding_dim, num_classes=num_classes[feature],
                                        latent_size=args.DISCR_LATENT, device=device).to(device)
    discriminators[feature] = discriminator_model

# Optimizer for the Filter
trainer = torch.optim.Adam(filter_model.parameters(), args.LR)

# Initialize the discriminator optimizers (one for each sensitive feature)
discriminator_trainers = {
    feature: torch.optim.Adam(discriminators[feature].parameters(), args.LR_DISC)
    for feature in args.SENSITIVE_FEATURES
}

# Combined labeled and unlabeled data to loop through
train_total = [train_nofeature, train]

if not args.ORIGIN:
    for epoch in range(args.N_EPOCHS):
        # Loop through both unlabeled and labeled data
        for whether_feature, train_iter in enumerate(train_total):
            data_type = "Unlabeled" if not whether_feature else "Labeled"

            for batch_data in tqdm(train_iter, desc=f"Epoch {epoch} - Training ({data_type})"):
                # TRAINING FILTER

                # Unpack all elements from the batch data
                if whether_feature:
                    user_id, item_id, rating, gender, age, occupation, user_emb, movie_emb = batch_data
                else:
                    user_id, item_id, rating, user_emb, movie_emb = batch_data

                # Move the tensors to the device
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                rating = rating.to(device)
                user_emb = user_emb.to(device)
                movie_emb = movie_emb.to(device)
                if whether_feature:
                    gender = gender.to(device)
                    age = age.to(device)
                    occupation = occupation.to(device)

                    # Put in dict. so we can loop through the sensitive attributes
                    sensitive_attrs = {"Gender": gender, "Age": age, "Occupation": occupation}

                    # Reshape labels to be 1D if they are not
                    for feature in args.SENSITIVE_FEATURES:
                        if sensitive_attrs[feature].dim() != 1:
                            sensitive_attrs[feature] = sensitive_attrs[feature].view(-1)  # Reshape to 1D

                user_emb_filtered = filter_model(user_emb)

                trainer.zero_grad()

                L_L, L_N = 0, 0
                for i, feature in enumerate(args.SENSITIVE_FEATURES):
                    # IF LABELED DATA
                    if whether_feature == 1:
                        # Pass filtered embeddings to discriminator and get the CrossEntropy loss when comparing
                        # the sensitive attribute prediction (e.g. male) to the true label (e.g. female) by calling
                        # forward() on discr. for the given sens. attr.
                        L_L -= discriminators[feature](user_emb_filtered, sensitive_attrs[feature])
                    else:
                        # IF UNLABELED DATA

                        # Pass filtered embeddings to discriminator and get the loss L_N for the unknown data by
                        # calling hforward() on discr. for the given sens. attr.
                        L_N -= discriminators[feature].hforward(user_emb_filtered)

                # Use the filtered embeddings to do prediction (1: interest vs 0: no interest),
                # then compute BCE loss between predictions and true ratings 0 or 1. This way we train the filter to
                # generate filtered embeddings that are not only fair but are also still good at the user modeling task

                # For now we use cos similarity between user and movie embeddings + sigmoid to do predictions which is
                # naive because the embeddings were trained to be fed into an mlp (for ncf at least) + sigmoid for
                # prediction so actually we want to have this same mlp be loaded here for predictions preferably.
                cos_sim = F.cosine_similarity(user_emb_filtered, movie_emb, dim=1)
                predictions = torch.sigmoid(cos_sim)

                L_task = F.binary_cross_entropy(predictions, rating)

                L_F = args.LAMBDA_1 * L_task + args.LAMBDA_2 * L_L + args.LAMBDA_3 * L_N
                L_F.backward()
                trainer.step()  # optimze filter_model

                # TRAIN DISCRIMINATORS

                # Re-filter embeddings because the filter was just trained one step
                user_emb_filtered = filter_model(user_emb)

                if whether_feature == 1:
                    for _ in range(args.EPOCHS_DISCRIMINATOR):
                        for i, feature in enumerate(args.SENSITIVE_FEATURES):
                            discriminator_trainers[feature].zero_grad()
                            L_D = discriminators[feature](user_emb_filtered.detach(), sensitive_attrs[feature])
                            L_D.backward()
                            discriminator_trainers[feature].step()

# Evaluation loop
filter_model.eval()
y_pred = []
y_true = []

for batch_data in tqdm(test if args.USE_TEST_DATA else valid, desc=f"Testing (Filtered) User Model"):
    # Unpack all elements from the batch data
    user_id, item_id, rating, user_emb, movie_emb = batch_data

    # Move the tensors to the device
    user_id = user_id.to(device)
    item_id = item_id.to(device)
    rating = rating.to(device)
    user_emb = user_emb.to(device)
    movie_emb = movie_emb.to(device)

    user_emb_filtered = filter_model(user_emb).detach()

    cos_sim = F.cosine_similarity(user_emb_filtered, movie_emb, dim=1)
    predictions = torch.sigmoid(cos_sim)

    y_pred.extend(predictions.tolist())
    y_true.extend(rating.tolist())

y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

mse = mean_squared_error(y_true, y_pred)

# Calculate RMSE
rmse = np.sqrt(mse)

print(f"Evaluation Results User Model (Filtered):")
print(f"Accuracy: {accuracy_score(y_true, y_pred_binary):.4f}")
print(f"ROC AUC: {roc_auc_score(y_true, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Added these:
print(f"Precision: {precision_score(y_true, y_pred_binary):.4f}")
print(f"Recall: {recall_score(y_true, y_pred_binary):.4f}")
print(f"F1 Score: {f1_score(y_true, y_pred_binary):.4f}")
print()

filter_model.train()

# Initialize attackers (newly initialized discriminator models) to extract sensitive attributes from filtered
# user representations
if args.MODEL == "NCF":
    attackers = {}
    attacker_trainers = {}
    for feature in args.SENSITIVE_FEATURES:
        attackers[feature] = Discriminator(embed_dim=embedding_dim, num_classes=num_classes[feature],
                                           latent_size=args.DISCR_LATENT, device=device).to(device)
        attackers[feature].train()
        attacker_trainers[feature] = torch.optim.Adam(attackers[feature].parameters(), args.LR_DISC)

# TRAINING ATTACKERS
best_result = {}
best_epoch = {}
for feature in args.SENSITIVE_FEATURES:
    best_result[feature] = 0
    best_epoch[feature] = 0
for _ in range(args.EPOCHS_ATTACKER):
    for batch_data in attacker_train:
        user_id, gender, age, occupation, user_emb = batch_data

        # Move the tensors to the device
        user_id = user_id.to(device)
        user_emb = user_emb.to(device)
        gender = gender.to(device)
        age = age.to(device)
        occupation = occupation.to(device)

        # Put in dict. so we can loop through the sensitive attributes
        sensitive_attrs = {"Gender": gender, "Age": age, "Occupation": occupation}

        # Reshape labels to be 1D if they are not
        for feature in args.SENSITIVE_FEATURES:
            if sensitive_attrs[feature].dim() != 1:
                sensitive_attrs[feature] = sensitive_attrs[feature].view(-1)  # Reshape to 1D

        user_emb_filtered = filter_model(user_emb)

        for i, feature in enumerate(args.SENSITIVE_FEATURES):
            attacker_trainers[feature].zero_grad()
            loss = attackers[feature](user_emb_filtered.detach(), sensitive_attrs[feature])
            loss.backward()
            attacker_trainers[feature].step()

    # EVALUATE ATTACKERS
    feature_pred = {}
    feature_true = {}
    for feature in args.SENSITIVE_FEATURES:
        feature_pred[feature] = []
        feature_true[feature] = []
        attackers[feature].eval()
    for batch_data in attacker_test:
        user_id, gender, age, occupation, user_emb = batch_data

        # Move the tensors to the device
        user_id = user_id.to(device)
        user_emb = user_emb.to(device)
        gender = gender.to(device)
        age = age.to(device)
        occupation = occupation.to(device)

        # Put in dict. so we can loop through the sensitive attributes
        sensitive_attrs = {"Gender": gender, "Age": age, "Occupation": occupation}

        # Reshape labels to be 1D if they are not
        for feature in args.SENSITIVE_FEATURES:
            if sensitive_attrs[feature].dim() != 1:
                sensitive_attrs[feature] = sensitive_attrs[feature].view(-1)  # Reshape to 1D

        user_emb_filtered = filter_model(user_emb)

        for i, feature in enumerate(args.SENSITIVE_FEATURES):
            pred = attackers[feature].predict(user_emb_filtered).cpu()
            feature_pred[feature].extend(pred.tolist())
            feature_true[feature].extend(sensitive_attrs[feature].tolist())
    for feature in args.SENSITIVE_FEATURES:
        auc_score = calculate_auc_score_for_feature(feature_true[feature], feature_pred[feature])
        # print(f"AUC for {feature}: {auc_score:.4f}")
    for feature in args.SENSITIVE_FEATURES:
        attackers[feature].train()

# FINAL ATTACKER SCORES AFTER TRAINING THEM
print(f"Evaluation Results Attackers:")

for feature in args.SENSITIVE_FEATURES:
    auc_score = calculate_auc_score_for_feature(feature_true[feature], feature_pred[feature])
    print(f"AUC for {feature}: {auc_score:.4f}")

print("finish")
