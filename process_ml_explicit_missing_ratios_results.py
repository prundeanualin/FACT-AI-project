import os
import pandas as pd
import re
from utils_ml import save_table_latex, plot_auc_performance


# Directory structure setup
base_dir = "ml_experiments_results"
file_name = 'ml_explicit_missing_ratios_results.txt'
model_specific_name = "(Overfitted)" if file_name.split('.')[0].split("_")[-1] == "overfitted" else ""
sub_dir = file_name.split('.')[0]  # Use the file name as directory name without the file extension
file_path = os.path.join(base_dir, sub_dir, file_name)

missing_ratios = [0.2, 0.4, 0.6, 0.8, 0.95]
replace_origin = file_name == "ml_explicit_missing_ratios_results.txt"

# This is only for one run that used old code "ml_explicit_missing_ratios_results.txt"
# Set replace_origin to False if you're re-running the experiment (Exp. 1) by uncommenting the line below:
# replace_origin = False
origin_replacement_data = {
    'NCF': {
        'Mean Absolute Error': 0.6909,
        'Mean Squared Error': 0.9451,
        'Root Mean Squared Error': 0.9722,
        'AUC for Gender': 0.5068,
        'AUC for Age': 0.5003,
        'AUC for Occupation': 0.5103
    },
    'PMF': {
        'Mean Absolute Error': 0.6868,
        'Mean Squared Error': 0.9747,
        'Root Mean Squared Error': 0.9873,
        'AUC for Gender': 0.7203,
        'AUC for Age': 0.5818,
        'AUC for Occupation': 0.5253
    }
}

# Initialize variables to store the parsed data
data = []

# Regular expressions for parsing the text
run_info_regex = re.compile(r"Run Timestamp: (.+)")
namespace_regex = re.compile(r"Namespace\((.+)\)")
metrics_regex = re.compile(r"(Mean Absolute Error|Mean Squared Error|Root Mean Squared Error|AUC for Gender|AUC for Age|AUC for Occupation): ([0-9.]+)")

# Regular expression for parsing the method identifier
method_identifier_regex = re.compile(r"(FairLISA|Origin|ComFair)")

# Read the file and parse the data
with open(file_path, 'r') as file:
    lines = file.readlines()
    current_run = {}
    current_method = None  # Variable to track the current method

    for line in lines:
        # Check for method identifier
        method_match = method_identifier_regex.match(line)
        if method_match:
            current_method = method_match.group(1)
            continue

        # Check for run timestamp
        timestamp_match = run_info_regex.match(line)
        if timestamp_match:
            if current_run:  # If there's a run already parsed, add it to the data list
                data.append(current_run)
                current_run = {}
            current_run['timestamp'] = timestamp_match.group(1)
            current_run['Method'] = current_method  # Assign the current method
            continue

        # Check for namespace with parameters
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

        # Check for metric results
        metrics_match = metrics_regex.match(line)
        if metrics_match:
            current_run[metrics_match.group(1)] = float(metrics_match.group(2))

# Add the last parsed run to the data list
if current_run:
    data.append(current_run)

# Convert the parsed data to a DataFrame
df = pd.DataFrame(data)


# Function to create a table for a specific missing ratio
def create_table_for_missing_ratio(df, missing_ratio, replace_origin=False):
    final_columns = ['Model', 'Method', 'AUC-G', 'AUC-A', 'AUC-O', 'RMSE']
    final_data = []
    df_filtered = df[df['MISSING_RATIO'] == missing_ratio]
    for model in ['PMF', 'NCF']:
        for method in ['Origin', 'ComFair', 'FairLISA']:
            if replace_origin and method == 'Origin':
                # Use replacement data for Origin
                replacement = origin_replacement_data[model]
                final_data.append([
                    model,
                    method,
                    replacement['AUC for Gender'],
                    replacement['AUC for Age'],
                    replacement['AUC for Occupation'],
                    replacement['Root Mean Squared Error']
                ])
            else:
                model_data = df_filtered[(df_filtered['MODEL'] == model) &
                                         (df_filtered['Method'] == method)]
                if not model_data.empty:
                    final_data.append([
                        model,
                        method,
                        model_data.iloc[0]['AUC for Gender'],
                        model_data.iloc[0]['AUC for Age'],
                        model_data.iloc[0]['AUC for Occupation'],
                        model_data.iloc[0]['Root Mean Squared Error']
                    ])
    return pd.DataFrame(final_data, columns=final_columns)


# Create tables for missing ratios
for missing_ratio in missing_ratios:
    table = create_table_for_missing_ratio(df, missing_ratio)
    print(table)

    # Save the tables for missing ratios
    save_table_latex(table, missing_ratio, base_dir, sub_dir)

for model in ['PMF', 'NCF']:
    plot_auc_performance(df, model, missing_ratios, base_dir, sub_dir, replace_origin, origin_replacement_data,
                         model_specific_name)
