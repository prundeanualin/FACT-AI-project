import os
import pandas as pd
import re
import matplotlib.pyplot as plt

# Directory structure setup

base_dir = "ml_experiments_results"
file_name = 'ml_explicit_lambdas_results.txt'
sub_dir = file_name.split('.')[0]  # Use the file name as directory name without the file extension
file_path = os.path.join(base_dir, sub_dir, file_name)

# Initialize variables to store the parsed data
data = []

# Regular expressions for parsing the text
run_info_regex = re.compile(r"Run Timestamp: (.+)")
namespace_regex = re.compile(r"Namespace\((.+)\)")
metrics_regex = re.compile(
    r"(Mean Absolute Error|Mean Squared Error|Root Mean Squared Error|AUC for Gender|AUC for Age|AUC for Occupation): ([0-9.]+)")

# Read the file and parse the data
with open(file_path, 'r') as file:
    lines = file.readlines()
    current_run = {}

    for line in lines:
        timestamp_match = run_info_regex.match(line)
        if timestamp_match:
            if current_run:  # If there's a run already parsed, add it to the data list
                data.append(current_run)
            current_run = {'timestamp': timestamp_match.group(1)}
            continue

        namespace_match = namespace_regex.match(line)
        if namespace_match:
            params = namespace_match.group(1).split(", ")
            for param in params:
                if '=' in param:  # Check if the parameter contains '='
                    key, value = param.split('=')
                    try:
                        current_run[key] = eval(value)  # For non-list values
                    except SyntaxError as e:
                        pass
            continue

        metrics_match = metrics_regex.match(line)
        if metrics_match:
            current_run[metrics_match.group(1)] = float(metrics_match.group(2))

# Add the last parsed run to the data list
if current_run:
    data.append(current_run)

# Convert the parsed data to a DataFrame
df = pd.DataFrame(data)
df = df.drop(columns=['DATA_BASE_PATH', 'RESULTS_DIR', 'CUDA', 'SEED', 'BATCH_SIZE', 'N_EPOCHS', 'EPOCHS_DISCRIMINATOR',
                      'EPOCHS_ATTACKER', 'LR', 'LR_DISC', 'DISCR_LATENT', 'USE_NOFEATURE', 'DEVICE', 'EMB_DIM',
                      'NCF_MODEL_PATH', 'PMF_MODEL_PATH', 'USE_TEST_DATA', 'RESULTS_FILENAME', 'SAVE_RES_TO_TXT',
                      'RESULTS_DIR'])

# Dropping duplicate rows based on LAMBDA_2, LAMBDA_3, and MODEL
df_filtered = df.drop_duplicates(subset=['LAMBDA_2', 'LAMBDA_3', 'MODEL']).reset_index()

# Display the filtered DataFrame
print(df_filtered)

# Unique model types
unique_models = df_filtered['MODEL'].unique()

# Containers for the filtered data
lambda_2_zero = {}
lambda_3_zero = {}
lambda_2_fifteen = {}
lambda_3_fifteen = {}

for model in unique_models:
    # Filter where lambda 2 is 0 and lambda 3 is any other value
    lambda_2_zero[model] = df_filtered[(df_filtered['MODEL'] == model) & (df_filtered['LAMBDA_2'] == 0)]

    # Filter where lambda 3 is 0 and lambda 2 is any other value
    lambda_3_zero[model] = df_filtered[(df_filtered['MODEL'] == model) & (df_filtered['LAMBDA_3'] == 0)]

    # Filter where lambda 2 is 15 and lambda 3 is any other value
    lambda_2_fifteen[model] = df_filtered[(df_filtered['MODEL'] == model) & (df_filtered['LAMBDA_2'] == 15)]

    # Filter where lambda 3 is 15 and lambda 2 is any other value
    lambda_3_fifteen[model] = df_filtered[(df_filtered['MODEL'] == model) & (df_filtered['LAMBDA_3'] == 15)]

# Sorting each dataset
for model in unique_models:
    lambda_2_zero[model] = lambda_2_zero[model].sort_values(by='LAMBDA_3')
    lambda_3_zero[model] = lambda_3_zero[model].sort_values(by='LAMBDA_2')
    lambda_2_fifteen[model] = lambda_2_fifteen[model].sort_values(by='LAMBDA_3')
    lambda_3_fifteen[model] = lambda_3_fifteen[model].sort_values(by='LAMBDA_2')

# Example: Print the filtered data for a specific model
model_example = unique_models[0]  # Replace with the model name you want to check
print(f"Model: {model_example}")
print("Lambda 2 Zero:")
print(lambda_2_zero[model_example])
print("Lambda 3 Zero:")
print(lambda_3_zero[model_example])
print("Lambda 2 Fifteen:")
print(lambda_2_fifteen[model_example])
print("Lambda 3 Fifteen:")
print(lambda_3_fifteen[model_example])

sensitive_attributes = ["Gender", "Age", "Occupation"]
sensitive_attribute_cols = [f"AUC for {sens_feat}" for sens_feat in sensitive_attributes]
save_dir = os.path.join(base_dir, sub_dir)

# Main loop to create plots for each model
for model in unique_models:
    # First main plot with other lambda fixed at 0
    fixed_at = 0
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Other Lambda Fixed at {fixed_at}')

    for i, sens_attr_col in enumerate(sensitive_attribute_cols):
        ax = plt.subplot(1, 3, i+1)
        ax.plot(lambda_3_zero[model]['LAMBDA_2'], lambda_3_zero[model][sens_attr_col], 'ro--', label=r'$\lambda_2$ $(U_L)$')
        ax.plot(lambda_2_zero[model]['LAMBDA_3'], lambda_2_zero[model][sens_attr_col], 'bo-.', label=r'$\lambda_3$ $(U_N)$')
        # ax.set_title(sensitive_attributes[i])
        ax.set_xlabel(f'Recommender System: {model} ({sensitive_attributes[i]})')
        ax.set_ylabel('AUC')
        ax.set_xticks([0, 5, 10, 15])
        ax.grid(True)
        if i == 0:
            legend = ax.legend()
            legend.get_frame().set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'lambdas_plot_explicit_{model}_fixed_{fixed_at}.png'))  # Save plot
    plt.show()

    # Second main plot with other lambda fixed at 15
    fixed_at = 15
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Other Lambda Fixed at 15')

    for i, sens_attr_col in enumerate(sensitive_attribute_cols):
        ax = plt.subplot(1, 3, i+1)
        ax.plot(lambda_3_fifteen[model]['LAMBDA_2'], lambda_3_fifteen[model][sens_attr_col], 'ro--', label=r'$\lambda_2$ $(U_L)$')
        ax.plot(lambda_2_fifteen[model]['LAMBDA_3'], lambda_2_fifteen[model][sens_attr_col], 'bo-.', label=r'$\lambda_3$ $(U_N)$')
        # ax.set_title(sensitive_attributes[i])
        ax.set_xlabel(f'Recommender System: {model} ({sensitive_attributes[i]})')
        ax.set_ylabel('AUC')
        ax.set_xticks([0, 5, 10, 15])
        ax.grid(True)
        if i == 0:
            legend = ax.legend()
            legend.get_frame().set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'lambdas_plot_explicit_{model}_fixed_{fixed_at}.png'))  # Save plot
    plt.show()
