import pickle
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os
import time

from model.CD import NCDM, IRT, MIRT
from model.fairlisa_models import Filter
from model.Discriminators import Discriminator
from preprocess_dataset_pisa import preprocess_dataset, split_df
from model.dataloader import transform, attacker_transform
from model.trainers.cd_trainer import train_model, evaluate_model
from utils import * 


import wandb

args = argparse.ArgumentParser()
args.add_argument("-DATA", default="pisa2015", type=str)
args.add_argument("-TRAIN_USER_MODEL", default=False, type=bool)
args.add_argument("-FILTER_MODE", default="separate", type=str)
args.add_argument(
    "-SENSITIVE_FEATURES",
    default=["OECD", "GENDER", "EDU", "ECONOMIC"],
    type=list,
)  # ["OECD", "GENDER", "EDU", "ECONOMIC"]
args.add_argument("-LAMBDA_1", default=1.0, type=float)
args.add_argument("-LAMBDA_2", default=2.0, type=float)
args.add_argument("-LAMBDA_3", default=1.0, type=float)
args.add_argument("-CUDA", default=2, type=int)
args.add_argument("-SEED", default=420, type=int) # seeds used are [4869, 420, 23]
args.add_argument("-BATCH_SIZE", default=8192, type=int)
args.add_argument("-BATCH_SIZE_ATTACKER", default=512, type=int)
args.add_argument("-EPOCH", default=10, type=int)
args.add_argument("-EPOCH_DISCRIMINATOR", default=10, type=int)
args.add_argument("-EPOCH_ATTACKER", default=10, type=int)
args.add_argument("-USER_NUM", default=462916, type=int)
args.add_argument("-ITEM_NUM", default=593, type=int)
args.add_argument("-KNOWLEDGE_NUM", default=128, type=int)
args.add_argument("-LATENT_NUM", default=16, type=int)
args.add_argument("-MODEL", default="NCDM", type=str)
args.add_argument("-LR", default=0.001, type=float)
args.add_argument("-LR_DISC", default=0.01, type=float)
args.add_argument("-USE_NOFEATURE", default=True, type=bool)
args.add_argument("-RATIO_NO_FEATURE", default=0.2, type=float)
args.add_argument("-PREPROCESS_DATA", default=False, type=bool)
args.add_argument("-DEVICE", default='cuda', type=str)
args.add_argument("--WANDB_ACTIVE", default=True, type=bool, action=argparse.BooleanOptionalAction)
args = args.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA)
seed_experiments(args.SEED)

model_name = f"{args.DATA}_{args.MODEL}_Lambda1-{args.LAMBDA_1}_Lambda2-{args.LAMBDA_2}_Lambda3-{args.LAMBDA_3}_{args.RATIO_NO_FEATURE}_{args.SEED}"
start_time = time.time()

if args.WANDB_ACTIVE:
    print("Wandb enabled")
    wandb.init(
        project="FACT-fairlisa",
        name=model_name,
        config={
            "dataset": args.DATA,
            "model": args.MODEL,
            "filter_mode": args.FILTER_MODE,
            "lambda_1": args.LAMBDA_1,
            "lambda_2": args.LAMBDA_2,
            "lambda_3": args.LAMBDA_3,
            "ratio_data_without_sensitive_features": args.RATIO_NO_FEATURE,
            "seed": args.SEED,
            "knowledge_dimension": args.KNOWLEDGE_NUM,
            "latent_dimension": args.LATENT_NUM,
            "batch_size": args.BATCH_SIZE,
            "epochs_cd_model": args.EPOCH,
            "lr_cd_model": args.LR,
            "epochs_discriminator": args.EPOCH_DISCRIMINATOR,
            "epochs_attacker": args.EPOCH_ATTACKER,
            "lr_discriminator": args.LR_DISC
        }
    )
else:
    print("Wandb disabled")
    wandb.init(mode="disabled")

print(args)
print(model_name)

print("load data")
if args.DEVICE == 'cuda' and torch.cuda.is_available():
    print("--Using cuda")
    print("--Number of GPU's available: ", torch.cuda.device_count())
    print("--Cuda device name: ", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("--Using cpu")
    device = torch.device("cpu")

pkl = open("./data/" + args.DATA + "/item2knowledge.pkl", "rb")
item2knowledge = pickle.load(pkl)
pkl.close()

if args.PREPROCESS_DATA:
    preprocess_dataset(args.SEED)

train_data_initial = pd.read_csv("./data/" + args.DATA + "/pisa.train.csv")
valid_data_initial = pd.read_csv("./data/" + args.DATA + "/pisa.validation.csv")
test_data_initial = pd.read_csv("./data/" + args.DATA + "/pisa.test.csv")

# Split the dataset based on a specified MISSING_RATIO with a seed for reproducibility.
train_w_sensitive_features, train_without_sensitive_features = split_df(train_data_initial, args.SEED, [args.RATIO_NO_FEATURE])

attacker_train_data = pd.read_csv("./data/" + args.DATA + "/pisa.attacker.train.csv")
attacker_test_data = pd.read_csv("./data/" + args.DATA + "/pisa.attacker.test.csv")

attacker_train, attacker_test = [
    attacker_transform(data["user_id"], data[args.SENSITIVE_FEATURES], batch_size=args.BATCH_SIZE_ATTACKER)
    for data in [attacker_train_data, attacker_test_data]
]

train_w_sensitive_features, train_without_sensitive_features, valid, test = transform(
    args.MODEL, 
    args.BATCH_SIZE, 
    [train_w_sensitive_features, train_without_sensitive_features, valid_data_initial, test_data_initial],
    sensitive_features=args.SENSITIVE_FEATURES, 
    item2knowledge=item2knowledge)

train_total = [train_w_sensitive_features, train_without_sensitive_features]
train_total_description = ["(data w/o sensitive)", "(data w sensitive)"]

saved_model_base_path = 'model/saved_user_models/'

embedding_dim = {
    "IRT": 1,
    "MIRT": args.LATENT_NUM,
    "NCDM": args.KNOWLEDGE_NUM
}

cdm = eval(args.MODEL)(args, device)
user_model_args = {
    'model': args.MODEL,
    'batch_size': args.BATCH_SIZE,
    'lr': args.LR,
    'epochs': args.EPOCH
}
if args.TRAIN_USER_MODEL:
    print(">> Training the user model...")
    train = transform(
        args.MODEL,
        args.BATCH_SIZE,
        [train_data_initial],
        sensitive_features=args.SENSITIVE_FEATURES,
        item2knowledge=item2knowledge)
    # cdm.train(train_data_model, test_data_model, epoch=args.EPOCH, device=device)
    # cdm.save(saved_model_base_path + f'{args.MODEL.lower()}.pt')
    cdm.to(device)
    train_model(cdm, user_model_args, train, valid, test, device, args.SENSITIVE_FEATURES, saved_model_base_path, item2knowledge)


print("Loading the trained user model...")
cdm.load_model(saved_model_base_path + f'{args.MODEL.lower()}.pt')
print("Evaluating the trained user model...")
acc, roc_auc, mae, mse = evaluate_model(cdm, user_model_args, test, device)

training_metrics = {
    "model/acc": acc,
    "model/auc": roc_auc,
    "model/mae": mae,
    "model/mse": mse
}

# Initialize Filter
filter_model = Filter(embedding_dim[args.MODEL], dense_layer_dim=args.LATENT_NUM, device=device).to(device)
# Optimizer for the Filter
filter_trainer = torch.optim.Adam(filter_model.parameters(), args.LR)

discriminators = {}
for feature in args.SENSITIVE_FEATURES:
    discriminators[feature] = Discriminator(args, embedding_dim[args.MODEL], device).to(device)
discriminator_trainers = {
    feature: torch.optim.Adam(discriminators[feature].parameters(), args.LR_DISC)
    for feature in args.SENSITIVE_FEATURES
}

print("\n>> Training the filter & discriminator...")

for epoch in range(args.EPOCH):
    cdm.eval()
    # Loop through both unlabeled and labeled data
    for is_data_with_sensitive_features, train_iter in enumerate(train_total):
        for batch_data in tqdm(train_iter, "Epoch %s " % epoch + train_total_description[is_data_with_sensitive_features]):
            
            # TRAINING FILTER
            filter_model.train()
            # Fix discriminator
            for feature in args.SENSITIVE_FEATURES:
                discriminators[feature].eval()

            if not model_has_knowledge_dimension(args.MODEL):
                user_id, item_id, response, feature_data = batch_data
            else:
                user_id, item_id, knowledge, response, feature_data = batch_data
                knowledge = knowledge.to(device)

            # Get the user embeddings from the user model
            user_embeddings = cdm.get_user_embeddings(user_id)

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            response = response.to(device)
            feature_data = feature_data.to(device)
            user_embeddings = user_embeddings.to(device)

            # Put feature_data in dict. so we can loop through the sensitive attributes
            sensitive_features_data = {feature: feature_data[:, i] for i, feature in enumerate(args.SENSITIVE_FEATURES)}

            # Apply the filter on the user embeddings
            user_emb_filtered = filter_model(user_embeddings)

            filter_trainer.zero_grad()

            L_L, L_N = 0, 0

            for i, feature in enumerate(args.SENSITIVE_FEATURES):
                # IF LABELED DATA
                if is_data_with_sensitive_features == 1:
                    # Pass filtered embeddings to discriminator and get the CrossEntropy loss when comparing
                    # the sensitive attribute prediction (e.g. male) to the true label (e.g. female) by calling
                    # forward() on discr. for the given sens. attr.
                    L_L -= discriminators[feature](user_emb_filtered, sensitive_features_data[feature])
                else:
                    # IF UNLABELED DATA

                    # Pass filtered embeddings to discriminator and get the loss L_N for the unknown data by
                    # calling hforward() on discr. for the given sens. attr.
                    L_N -= discriminators[feature].hforward(user_emb_filtered)

            # Use the filtered embeddings to do prediction (1: interest vs 0: no interest),
            # then compute BCE loss between predictions and true ratings 0 or 1. This way we train the filter to
            # generate filtered embeddings that are not only fair but are also still good at the user modeling task
            if not model_has_knowledge_dimension(args.MODEL):
                out = cdm(user_id, item_id, response, user_emb_filtered)
            else:
                out = cdm(user_id, item_id, knowledge, response, user_emb_filtered)

            L_task = out["loss"]

            L_F = args.LAMBDA_1 * L_task + args.LAMBDA_2 * L_L + args.LAMBDA_3 * L_N
            L_F.backward()
            # optimize filter_model
            filter_trainer.step()

            # TRAIN DISCRIMINATORS
            for feature in args.SENSITIVE_FEATURES:
                discriminators[feature].train()
            # Fix filter
            filter_model.eval()
            # Re-filter embeddings because the filter was just trained one step
            user_emb_filtered = filter_model(user_embeddings)

            if is_data_with_sensitive_features == 1:
                for _ in range(args.EPOCH_DISCRIMINATOR):
                    for i, feature in enumerate(args.SENSITIVE_FEATURES):
                        discriminator_trainers[feature].zero_grad()
                        L_D = discriminators[feature](user_emb_filtered.detach(), sensitive_features_data[feature])
                        L_D.backward()
                        discriminator_trainers[feature].step()

    print(f"-- Filtered model evaluation at epoch {epoch + 1}/{args.EPOCH}")
    acc, roc_auc, mae, mse = evaluate_model(cdm, user_model_args, test, device, filter_model)

    training_metrics = {
        "filtered_model/acc": acc,
        "filtered_model/auc": roc_auc,
        "filtered_model/mae": mae,
        "filtered_model/mse": mse
    }
    wandb.log(training_metrics)

print("\n>> Training the attacker...")

# Initialize attackers (newly initialized discriminator models) to extract sensitive attributes from filtered
# user representations
attackers = {}
attacker_trainers = {}
for feature in args.SENSITIVE_FEATURES:
    attackers[feature] = Discriminator(args, embedding_dim[args.MODEL], device).to(device)
    attacker_trainers[feature] = torch.optim.Adam(attackers[feature].parameters(), args.LR_DISC)

# Retrieve the filtered user embeddings, after the filter has been fully trained
# in an adversarial style along with the discriminator
filter_model.eval()
u_embeddings = filter_model(cdm.user_embedding_layer.weight).detach()

best_result = {}
best_epoch = {}
for feature in args.SENSITIVE_FEATURES:
    best_result[feature] = 0
    best_epoch[feature] = 0
for epoch in range(args.EPOCH_ATTACKER):
    for feature in args.SENSITIVE_FEATURES:
        attackers[feature].train()
    for batch_data in tqdm(attacker_train, "Epoch %s" % (epoch + 1)):
        user_id, feature_label = batch_data
        user_id = user_id.to(device)
        for i, feature in enumerate(args.SENSITIVE_FEATURES):
            label = feature_label[:, i].to(device)
            u_embedding = u_embeddings[user_id, :]
            attacker_trainers[feature].zero_grad()
            loss = attackers[feature](u_embedding, label)
            loss.backward()
            attacker_trainers[feature].step()

    feature_pred = {}
    feature_true = {}
    for feature in args.SENSITIVE_FEATURES:
        feature_pred[feature] = []
        feature_true[feature] = []
        attackers[feature].eval()
    for batch_data in tqdm(attacker_test, "Test"):
        user_id, feature_label = batch_data
        user_id = user_id.to(device)
        for i, feature in enumerate(args.SENSITIVE_FEATURES):
            u_embedding = u_embeddings[user_id, :]
            pred = attackers[feature].predict(u_embedding).cpu()
            feature_pred[feature].extend(pred.tolist())
            feature_true[feature].extend(feature_label[:, i].tolist())
    attacker_metrics = {}
    print(f"-- Attacker evaluation at epoch {epoch + 1}/{args.EPOCH_ATTACKER}")
    for feature in args.SENSITIVE_FEATURES:
        print(feature)
        auc_feature = roc_auc_score(feature_true[feature], feature_pred[feature])
        if auc_feature > best_result[feature]:
            print("-- New highscore!")
            best_result[feature] = auc_feature
            best_epoch[feature] = epoch
        print("auc:", auc_feature)
        attacker_metrics[f"attacker/{feature}"] = auc_feature
    wandb.log(attacker_metrics)

print("\n>> Attacker final evaluation after training")
for feature in args.SENSITIVE_FEATURES:
    print(feature)
    wandb.run.summary[f"attacker/{feature}"] = best_result[feature]
    print(f"best auc: {best_result[feature]} at epoch: {best_epoch[feature]}")

print("\n>> Finish")
end_time = time.time()
print("Total duration: ", end_time - start_time)
