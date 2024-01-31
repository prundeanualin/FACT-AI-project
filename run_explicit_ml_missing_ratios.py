import subprocess

models = ["NCF"]
missing_ratios = [0.6, 0.8, 0.95]
fairness_methods = {
    "FAIRLISA": {"LAMBDA_1": 1, "LAMBDA_2": 20, "LAMBDA_3": 10},
    "COMFAIR": {"LAMBDA_1": 1, "LAMBDA_2": 20, "LAMBDA_3": 0},
    "ORIGIN": {"LAMBDA_1": 1, "LAMBDA_2": 20, "LAMBDA_3": 10, "ORIGIN": True},
}

default_params = "-USE_TEST_DATA True -BATCH_SIZE 8192 -N_EPOCHS 10 -EPOCHS_DISCRIMINATOR 10 -EPOCHS_ATTACKER 50 " \
                 "-FILTER_LAYER_SIZES \"16,16\" -LR 0.001 -LR_DISC 0.01 -DISCR_LATENT 16 -NCF_LAYERS \"32,16,8\" " \
                 "-EMB_DIM 16 -SEED 4869 -SAVE_RES_TO_TXT True "

"""
Experiment 2: ml_explicit_missing_ratios_results_overfitted.txt. Here we use overfitted user models that perform better
on the training data but worse on the validation data because they might have extracted more sensitive information. 
"""
# Use more (overfitted) trained user model (trained for more epochs):
default_params += "-RESULTS_FILENAME ml_explicit_missing_ratios_results_overfitted.txt " \
                  "-PMF_MODEL_PATH pmf_models/pmf_model_explicit_emb_16_epoch_1000.pth " \
                  "-NCF_MODEL_PATH ncf_models_explicit/ncf_model_explicit_emb_16_epoch_160.pth"

for model in models:
    for missing_ratio in missing_ratios:
        for method, params in fairness_methods.items():
            command = f"python LISA_explicit.py -MODEL {model} -MISSING_RATIO {missing_ratio} "
            command += " ".join([f"-{param} {value}" for param, value in params.items()])
            command += " " + default_params
            print(f"Running: {command}")
            print(method)
            subprocess.run(command, shell=True)
