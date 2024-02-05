import argparse

from cognitive_diagnosis.plotter import *
from cognitive_diagnosis.preprocess_dataset_pisa import preprocess_dataset
from run import run, set_device, train_user_model
from cd_trainer import methods
import pandas as pd


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
args.add_argument("-BATCH_SIZE_ATTACKER", default=256, type=int)
args.add_argument("-EPOCH", default=10, type=int)
args.add_argument("-EPOCH_DISCRIMINATOR", default=10, type=int)
args.add_argument("-EPOCH_ATTACKER", default=10, type=int)
args.add_argument("-USER_NUM", default=462916, type=int)
args.add_argument("-ITEM_NUM", default=593, type=int)
args.add_argument("-KNOWLEDGE_NUM", default=128, type=int)
args.add_argument("-LATENT_NUM", default=16, type=int)
args.add_argument("-MODEL", default="MIRT", type=str)
args.add_argument("-LR", default=0.001, type=float)
args.add_argument("-LR_DISC", default=0.01, type=float)
args.add_argument("-USE_NOFEATURE", default=True, type=bool)
args.add_argument("-RATIO_NO_FEATURE", default=0.2, type=float)
args.add_argument("-PREPROCESS_DATA", default=False, type=bool)
args.add_argument("-DEVICE", default='cuda', type=str)
args = args.parse_args()

if args.PREPROCESS_DATA:
    preprocess_dataset(args.SEED)

device = set_device(args)

if args.TRAIN_USER_MODEL:
    train_user_model(args, device)


missing_ratios = [0.2, 0.4, 0.6, 0.8, 0.95]
# Resetting the values in the results table
df_empty = pd.read_csv(results_file, sep=',').head(0)
df_empty.to_csv(results_file, sep=',')

for missing_ratio in missing_ratios:
    for method_params in methods.values():
        lambda_2 = method_params[0]
        lambda_3 = method_params[1]
        vars(args)['LAMBDA_2'] = lambda_2
        vars(args)['LAMBDA_3'] = lambda_3
        vars(args)['RATIO_NO_FEATURE'] = missing_ratio
        run(args)

df_results = pd.read_csv(results_file, sep=',').head(0)
plot_missing_ratios(df_results, args.MODEL, missing_ratios)

