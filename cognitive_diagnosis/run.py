import pickle
import argparse

import pandas as pd
import time

from model.CD import IRT, MIRT, NCDM
from cognitive_diagnosis.plotter import results_file
from model.fairlisa_models import Filter
from model.Discriminators import Discriminator
from preprocess_dataset_pisa import split_df
from cognitive_diagnosis.dataloader import transform, attacker_transform
from cognitive_diagnosis.cd_trainer import *
from model.utils import *


def set_device(args):
    if args.DEVICE == 'cuda' and torch.cuda.is_available():
        print("--Using cuda")
        print("--Number of GPU's available: ", torch.cuda.device_count())
        print("--Cuda device name: ", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("--Using cpu")
        device = torch.device("cpu")
    return device


def train_user_model(args, device):

    pkl = open("../data/" + args.DATA + "/item2knowledge.pkl", "rb")
    item2knowledge = pickle.load(pkl)
    pkl.close()

    cdm = eval(args.MODEL)(args, device)
    user_model_args = {
        'model': args.MODEL,
        'batch_size': args.BATCH_SIZE,
        'lr': args.LR,
        'epochs': args.EPOCH
    }

    train_data_initial = pd.read_csv("../data/" + args.DATA + "/pisa.train.csv")
    valid_data_initial = pd.read_csv("../data/" + args.DATA + "/pisa.validation.csv")
    test_data_initial = pd.read_csv("../data/" + args.DATA + "/pisa.test.csv")

    print(">> Training the user model...")
    [train, valid, test] = transform(
        args.MODEL,
        args.BATCH_SIZE,
        [train_data_initial, valid_data_initial, test_data_initial],
        sensitive_features=args.SENSITIVE_FEATURES,
        item2knowledge=item2knowledge)
    cdm.to(device)
    train_model(cdm, user_model_args, train, valid, test, device, saved_model_base_path)

def run(args, device):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA)
    seed_experiments(args.SEED)

    model_name = f"{args.DATA}_{args.MODEL}_Lambda1({args.LAMBDA_1})_Lambda2({args.LAMBDA_2})_Lambda3({args.LAMBDA_3})_{args.RATIO_NO_FEATURE}_{args.SEED}"
    start_time = time.time()

    print(args)
    print(model_name)

    print("load data")

    pkl = open("../data/" + args.DATA + "/item2knowledge.pkl", "rb")
    item2knowledge = pickle.load(pkl)
    pkl.close()

    train_data_initial = pd.read_csv("../data/" + args.DATA + "/pisa.train.csv")
    valid_data_initial = pd.read_csv("../data/" + args.DATA + "/pisa.validation.csv")
    test_data_initial = pd.read_csv("../data/" + args.DATA + "/pisa.test.csv")

    # Split the dataset based on a specified MISSING_RATIO with a seed for reproducibility.
    train_w_sensitive_features, train_without_sensitive_features = split_df(train_data_initial, args.SEED, [args.RATIO_NO_FEATURE])

    # train_w_sensitive_features, train_without_sensitive_features = np.split(train_data_initial.sample(frac=1, random_state=args.SEED),
    #                                    [int(args.RATIO_NO_FEATURE * len(train_data_initial))])

    attacker_train_data = pd.read_csv("../data/" + args.DATA + "/pisa.attacker.train.csv")
    attacker_test_data = pd.read_csv("../data/" + args.DATA + "/pisa.attacker.test.csv")

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

    # If both lambda2 and lambda3 are 0, then we work with the Origin baseline
    # Origin has no fairness processing
    IS_ORIGIN = (args.LAMBDA_2 == 0) and (args.LAMBDA_3 == 0)
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

    print("Loading the trained user model...")
    cdm.load_model(saved_model_base_path + f'{args.MODEL}.pt')
    print("Evaluating the trained user model...")
    acc, roc_auc, mae, mse = evaluate_model(cdm, user_model_args, test, device)
    training_metrics = {
        "model/acc": acc,
        "model/auc": roc_auc,
        "model/mae": mae,
        "model/mse": mse
    }

    discriminators = {}
    for feature in args.SENSITIVE_FEATURES:
        discriminators[feature] = Discriminator(args, embedding_dim[args.MODEL], device).to(device)
    discriminator_trainers = {
        feature: torch.optim.Adam(discriminators[feature].parameters(), args.LR_DISC)
        for feature in args.SENSITIVE_FEATURES
    }

    if IS_ORIGIN:
        print("\n>> Origin baseline - skipping filter and discriminator training")
        training_metrics = {
            "filtered_model/acc": 0,
            "filtered_model/auc": 0,
            "filtered_model/mae": 0,
            "filtered_model/mse": 0
        }
    else:

        # Initialize Filter
        filter_model = Filter(embedding_dim[args.MODEL], dense_layer_dim=args.LATENT_NUM, device=device).to(device)
        # Optimizer for the Filter
        filter_trainer = torch.optim.Adam(filter_model.parameters(), args.LR)

        print("\n>> Training the filter & discriminator...")

        cdm.eval()

        for epoch in range(args.EPOCH):
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

    print("\n>> Training the attacker...")

    # Initialize attackers (newly initialized discriminator models) to extract sensitive attributes from filtered
    # user representations
    attackers = {}
    attacker_trainers = {}
    for feature in args.SENSITIVE_FEATURES:
        attackers[feature] = Discriminator(args, embedding_dim[args.MODEL], device).to(device)
        attacker_trainers[feature] = torch.optim.Adam(attackers[feature].parameters(), args.LR_DISC)

    if IS_ORIGIN:
        u_embeddings = cdm.user_embedding_layer.weight
    else:
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
                best_result[feature] = format_4dec(auc_feature)
                best_epoch[feature] = epoch
            print("auc:", auc_feature)
            attacker_metrics[f"attacker/{feature}"] = auc_feature

    print("\n>> Attacker final evaluation after training")
    for feature in args.SENSITIVE_FEATURES:
        print(feature)
        print(f"best auc: {best_result[feature]} at epoch: {best_epoch[feature]}")

    print("\n>> Finish")
    end_time = time.time()
    print("Total duration: ", end_time - start_time)

    method = 'FairLISA'
    if args.LAMBDA_3 == 0:
        if args.LAMBDA_2 == 0:
            method = 'ComFair'
        elif args.LAMBDA_2 == 2.0:
            method = 'Origin'

    print("Printing results to table...")
    df = pd.read_csv(results_file, sep=',')
    new_row = {
        'MODEL': args.MODEL,
        'METHOD': method,
        "LAMBDA_2": args.LAMBDA_2,
        "LAMBDA_3": args.LAMBDA_3,
        "MISSING_RATIO": args.RATIO_NO_FEATURE,
        "SEED": args.SEED,
        'ACC': training_metrics["filtered_model/acc"],
        'AUC': training_metrics["filtered_model/auc"],
        'MAE': training_metrics["filtered_model/mae"],
        'MSE': training_metrics["filtered_model/mse"],
        'Region': best_result['OECD'],
        'Gender': best_result['GENDER'],
        'Family Education': best_result['EDU'],
        'Family Economic': best_result['ECONOMIC'],
    }
    df2 = df.append(new_row, ignore_index=True)
    df2.to_csv(results_file, index=False)
