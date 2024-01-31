import subprocess

models = ["PMF", "NCF"]
missing_ratios = [0.2, 0.4, 0.6, 0.8, 0.95]
fairness_methods = {
    "FAIRLISA": {"LAMBDA_1": 1, "LAMBDA_2": 20, "LAMBDA_3": 10},
    "COMFAIR": {"LAMBDA_1": 1, "LAMBDA_2": 20, "LAMBDA_3": 0},
    "ORIGIN": {"LAMBDA_1": 1, "LAMBDA_2": 0, "LAMBDA_3": 0},
}

default_params = "-USE_TEST_DATA True -BATCH_SIZE 8192 -N_EPOCHS 10 -EPOCHS_DISCRIMINATOR 10 -EPOCHS_ATTACKER 50 -FILTER_LAYER_SIZES \"16,16\" -LR 0.001 -LR_DISC 0.01 -DISCR_LATENT 16 -NCF_LAYERS \"32,16,8\" -EMB_DIM 16 -SEED 4869 -SAVE_RES_TO_TXT True"

for model in models:
    for missing_ratio in missing_ratios:
        for method, params in fairness_methods.items():
            command = f"python LISA_explicit_implicit.py -MODEL {model} -MISSING_RATIO {missing_ratio} "
            command += " ".join([f"-{param} {value}" for param, value in params.items()])
            command += " " + default_params
            print(f"Running: {command}")
            subprocess.run(command, shell=True)
