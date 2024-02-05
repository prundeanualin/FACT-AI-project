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
    mean_absolute_error,
    mean_squared_error,
)
from user_models import NCF, PMF
from model.utils import seed_experiments
from utils_ml import calculate_auc_score_for_feature, split_dataset, write_args_to_file, setup_file_path

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
args.add_argument("-MODEL", default="PMF", type=str)
args.add_argument("-LR", default=0.001, type=float)
args.add_argument("-LR_DISC", default=0.01, type=float)
args.add_argument("-DISCR_LATENT", default=16, type=int)
args.add_argument("-USE_NOFEATURE", default=False, type=bool)
args.add_argument("-MISSING_RATIO", default=0.2, type=float)
args.add_argument("-DEVICE", default='cuda', type=str)
args.add_argument("-NCF_LAYERS", default="32,16,8", type=lambda s: [int(layer_size) for layer_size in s.split(',')])
args.add_argument("-EMB_DIM", default=16, type=int)
args.add_argument("-NCF_MODEL_PATH", default="ncf_models_explicit/ncf_model_explicit_emb_16_epoch_30.pth", type=str)
args.add_argument("-PMF_MODEL_PATH", default="pmf_models/pmf_model_explicit_emb_16_epoch_80.pth", type=str)
args.add_argument("-USE_TEST_DATA", default=False, type=bool)
# Specify the file name for saving results
args.add_argument("-RESULTS_FILENAME", default="ml_explicit_missing_ratios_results.txt", type=str)
args.add_argument("-SAVE_RES_TO_TXT", default=False, type=str)
args.add_argument("-ORIGIN", default=False, type=bool)
args.add_argument("-RESULTS_DIR", default="ml_experiments_results", type=str)
args = args.parse_args()

# Set seeds
seed_experiments(args.SEED)
model_name = args.MODEL
print(args)
print(f"Model: {model_name}")

# Create dirs and .txt
if args.SAVE_RES_TO_TXT:
    file_path = setup_file_path(base_dir=args.RESULTS_DIR, results_filename=args.RESULTS_FILENAME)
    current_time = write_args_to_file(args, file_path=file_path)

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

# Define number of classes to predict for sensitive features (e.g. 2 for gender)
num_classes = {feature: len(train[feature].unique()) for feature in args.SENSITIVE_FEATURES}

n_users, n_movies = len(train['UserID'].unique()), len(train['MovieID'].unique())
min_rating, max_rating = train['Rating'].min(), train['Rating'].max()

movie_ids = sorted(train['MovieID'].unique())
movie_id_to_index_mapping = {}
for i, id in enumerate(movie_ids):
    movie_id_to_index_mapping[id] = i


def attacker_transform(user, gender, age, occupation):
    # Map the movie IDs to indices
    user_indices = user - 1

    dataset = TensorDataset(
        torch.tensor(np.array(user_indices), dtype=torch.int64),
        torch.tensor(np.array(gender), dtype=torch.long),
        torch.tensor(np.array(age), dtype=torch.long),
        torch.tensor(np.array(occupation), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)


def transform(user, item, score, gender, age, occupation, movie_id_to_index_mapping, nofeature=False):
    # Map the movie IDs to indices
    item_indices = [movie_id_to_index_mapping[movie_id] for movie_id in item]
    user_indices = user - 1

    if not nofeature:
        dataset = TensorDataset(
            torch.tensor(np.array(user_indices), dtype=torch.int64),
            torch.tensor(np.array(item_indices), dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
            torch.tensor(np.array(gender), dtype=torch.long),
            torch.tensor(np.array(age), dtype=torch.long),
            torch.tensor(np.array(occupation), dtype=torch.long),
        )
    else:
        dataset = TensorDataset(
            torch.tensor(np.array(user_indices), dtype=torch.int64),
            torch.tensor(np.array(item_indices), dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
        )
    return DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)


print("Splitting train data based on missing ratio...")
train, train_nofeature = split_dataset(train, args.MISSING_RATIO, args.SEED)

# Calculate and print percentages
total_records = len(train) + len(train_nofeature)
total_users = len(train['UserID'].unique()) + len(train_nofeature['UserID'].unique())

print("train_data: {:.2f}% of total records, {:.2f}% of total users".format(
    len(train) / total_records * 100, len(train['UserID'].unique()) / total_users * 100))
print("train_data_nofeature: {:.2f}% of total records, {:.2f}% of total users".format(
    len(train_nofeature) / total_records * 100, len(train_nofeature['UserID'].unique()) / total_users * 100))

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
        movie_id_to_index_mapping,
        nofeature=no_feature
    ) for data, no_feature in [(train, False), (train_nofeature, True), (valid, True), (test, True)]]
attacker_train, attacker_test = [
    attacker_transform(
        data["UserID"],
        data.filter(like='Gender'),
        data.filter(like='Age'),
        data.filter(like='Occupation'),
    ) for data in [attacker_train, attacker_test]
]

if args.MODEL.lower() == 'pmf':
    user_model = PMF(n_users, n_movies, latent_vectors=args.EMB_DIM, lam_u=0.05, lam_v=0.05, explicit=True,
                     min_rating=min_rating, max_rating=max_rating)
    user_model.load_model(args.PMF_MODEL_PATH)
elif args.MODEL.lower() == 'ncf':
    user_model = NCF(n_users, n_movies, latent_dim=args.EMB_DIM, layers=args.NCF_LAYERS,
                     min_rating=min_rating, max_rating=max_rating)
    user_model.load_model(args.NCF_MODEL_PATH)
elif args.MODEL.lower() == 'lightgcn':
    pass


# Initialize Filter
filter_model = Filter(args.EMB_DIM, dense_layer_dim=args.FILTER_LAYER_SIZES[0], device=device)


# Initialize discriminators
discriminators = {}
for feature in args.SENSITIVE_FEATURES:
    discriminator_model = Discriminator(embed_dim=args.EMB_DIM, num_classes=num_classes[feature],
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
                    user_indices, movie_indices, rating, gender, age, occupation = batch_data
                else:
                    user_indices, movie_indices, rating = batch_data

                user_indices = user_indices.to(device)
                movie_indices = movie_indices.to(device)

                user_emb, movie_emb = user_model.get_embeddings(user_indices, movie_indices)

                # Move the tensors to the device
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

                predictions = user_model.predict_batch_with_embeddings(user_emb_filtered, movie_emb,
                                                                       scale_back_to_ratings=True)
                L_task = F.mse_loss(predictions.squeeze(), rating)

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
    user_indices, movie_indices, rating = batch_data
    user_indices = user_indices.to(device)
    movie_indices = movie_indices.to(device)

    user_emb, movie_emb = user_model.get_embeddings(user_indices, movie_indices)

    # Move the tensors to the device
    rating = rating.to(device)
    user_emb = user_emb.to(device)
    movie_emb = movie_emb.to(device)

    if not args.ORIGIN:
        user_emb_filtered = filter_model(user_emb).detach()
    else:
        user_emb_filtered = user_emb

    predictions = user_model.predict_batch_with_embeddings(user_emb_filtered, movie_emb, scale_back_to_ratings=True)

    y_pred.extend(predictions.tolist())
    y_true.extend(rating.tolist())

mse = mean_squared_error(y_true, y_pred)

# Calculate RMSE
rmse = np.sqrt(mse)

print(f"Evaluation Results User Model (Filtered):")
print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

filter_model.train()

# Initialize attackers (newly initialized discriminator models) to extract sensitive attributes from filtered
# user representations
attackers = {}
attacker_trainers = {}
for feature in args.SENSITIVE_FEATURES:
    attackers[feature] = Discriminator(embed_dim=args.EMB_DIM, num_classes=num_classes[feature],
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
        user_indices, gender, age, occupation = batch_data
        user_indices = user_indices.to(device)

        user_emb, _ = user_model.get_embeddings(user_indices, None)

        # Move the tensors to the device
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

        if not args.ORIGIN:
            user_emb_filtered = filter_model(user_emb).detach()
        else:
            user_emb_filtered = user_emb

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
        user_indices, gender, age, occupation = batch_data

        user_emb, _ = user_model.get_embeddings(user_indices, None)

        # Move the tensors to the device
        user_indices = user_indices.to(device)
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

        if not args.ORIGIN:
            user_emb_filtered = filter_model(user_emb).detach()
        else:
            user_emb_filtered = user_emb

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

if args.SAVE_RES_TO_TXT:
    # Append the final evaluation results to the file
    with open(file_path, "a") as file:
        file.write(f"Final Epoch Evaluation Results User Model (Filtered) - Run Timestamp: {current_time}\n")
        file.write(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}\n")
        file.write(f"Mean Squared Error: {mse:.4f}\n")
        file.write(f"Root Mean Squared Error: {rmse:.4f}\n\n")

        file.write(f"Final Epoch Evaluation Results Attackers - Run Timestamp: {current_time}\n")
        for feature in args.SENSITIVE_FEATURES:
            auc_score = calculate_auc_score_for_feature(feature_true[feature], feature_pred[feature])
            file.write(f"AUC for {feature}: {auc_score:.4f}\n")
        file.write("\n------------------------------------------------\n\n")
