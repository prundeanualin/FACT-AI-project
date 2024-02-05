import argparse
import subprocess

# Arguments
args = argparse.ArgumentParser()
# Rating threshold for binarizing data, default=1, only for implicit feedback
args.add_argument("-BINARIZATION_THRESHOLD", default=1, type=int)
args = args.parse_args()

models = ["PMF", "NCF"]
missing_ratios = [0.2, 0.4, 0.6, 0.8, 0.95]
fairness_methods = {
    "FAIRLISA": {"LAMBDA_1": 1, "LAMBDA_2": 20, "LAMBDA_3": 10, },
    "COMFAIR": {"LAMBDA_1": 1, "LAMBDA_2": 20, "LAMBDA_3": 0, },
    "ORIGIN": {"LAMBDA_1": 1, "LAMBDA_2": 20, "LAMBDA_3": 10, "ORIGIN": True},
}

"""
Experiment 1: ml_implicit_missing_ratios_results.txt, 
-BINARIZE_THRESHOLD was 1 (ratings were convert to label 0 if 1 to label 1 if 2 3 4 or 5 so embeddings were trained 
on a highly imbalanced dataset (like 95.85% 1's, 5.15% 0's) 
"""
default_params = "-USE_TEST_DATA True -BATCH_SIZE 8192 -N_EPOCHS 10 -EPOCHS_DISCRIMINATOR 10 -EPOCHS_ATTACKER 50 -FILTER_LAYER_SIZES \"16,16\" -LR 0.001 -LR_DISC 0.01 -DISCR_LATENT 16 -SEED 4869 -SAVE_RES_TO_TXT True "


if args.BINARIZATION_THRESHOLD:
    """
    Uncomment below 'default_params +=' line to run exp. 2
    
    Experiment 2: ml_implicit_missing_ratios_results_thresh_3.txt, 
    -BINARIZE_THRESHOLD was 3 (ratings were convert to label 0 if 1 2 or 3 to label 1 if 4 or 5 so embeddings were trained 
    on a highly balanced dataset (like 59.43% 1's, 40.57% 0's). This might make more sense because the embeddings might
    be better trained instead of just predicting the majority class. The embeddings might thus carry more information.
    """
    # Use embeddings trained on binarized data with threshold 3
    default_params += "-RESULTS_FILENAME ml_implicit_missing_ratios_results_thresh_3.txt -BINARIZE_THRESHOLD 3"

for model in models:
    for missing_ratio in missing_ratios:
        for method, params in fairness_methods.items():
            print(params)
            command = f"python LISA_implicit.py -MODEL {model} -MISSING_RATIO {missing_ratio} "
            command += " ".join([f"-{param} {value}" for param, value in params.items()])
            command += " " + default_params
            print(f"Running: {command}")
            print(method)
            subprocess.run(command, shell=True)
