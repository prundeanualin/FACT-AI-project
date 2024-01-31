import subprocess

models = ["PMF", "NCF"]
missing_ratios = [0.4]
lambdas = [0, 5, 10, 15]  # 0, 5, 10, 15
other_lambdas = [0, 15]  # 0, 15

default_params = "-USE_TEST_DATA True -BATCH_SIZE 8192 -N_EPOCHS 10 -EPOCHS_DISCRIMINATOR 10 -EPOCHS_ATTACKER 50 " \
                 "-FILTER_LAYER_SIZES \"16,16\" -LR 0.001 -LR_DISC 0.01 -DISCR_LATENT 16 -NCF_LAYERS \"32,16,8\" " \
                 "-EMB_DIM 16 -SEED 4869 -SAVE_RES_TO_TXT True " \
                 "-RESULTS_FILENAME ml_explicit_lambdas_results.txt"

for model in models:
    for missing_ratio in missing_ratios:
        for other_lambda in other_lambdas:
            for lambd in lambdas:
                command = f"python LISA_explicit.py -MODEL {model} -MISSING_RATIO {missing_ratio} -LAMBDA_2 {lambd} " \
                          f"-LAMBDA_3 {other_lambda}"
                command += " " + default_params
                print(f"Running: {command}")
                subprocess.run(command, shell=True)

        for other_lambda in other_lambdas:
            for lambd in lambdas:
                command = f"python LISA_explicit.py -MODEL {model} -MISSING_RATIO {missing_ratio} -LAMBDA_2 {other_lambda} " \
                          f"-LAMBDA_3 {lambd}"
                command += " " + default_params
                print(f"Running: {command}")
                subprocess.run(command, shell=True)

