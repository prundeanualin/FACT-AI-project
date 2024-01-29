import pandas as pd
import matplotlib.pyplot as plt
import re

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


def parse_list(string):
    """ Parses a string representation of a list into an actual list. """
    return string.strip("[]").replace("'", "").split(", ")


file_path = 'ml_explicit_missing_ratios_results.txt'

# Initialize variables to store the parsed data
data = []

# Regular expressions for parsing the text
run_info_regex = re.compile(r"Run Timestamp: (.+)")
namespace_regex = re.compile(r"Namespace\((.+)\)")
metrics_regex = re.compile(r"(Mean Absolute Error|Mean Squared Error|Root Mean Squared Error|AUC for Gender|AUC for Age|AUC for Occupation): ([0-9.]+)")

# Read the file and parse the data
with open(file_path, 'r') as file:
    lines = file.readlines()
    current_run = {}

    for line in lines:
        # Check for run timestamp
        timestamp_match = run_info_regex.match(line)
        if timestamp_match:
            if current_run:  # If there's a run already parsed, add it to the data list
                data.append(current_run)
                current_run = {}
            current_run['timestamp'] = timestamp_match.group(1)
            continue

        # Check for namespace with parameters
        namespace_match = namespace_regex.match(line)
        if namespace_match:
            params = namespace_match.group(1).split(", ")
            for param in params:
                if '=' in param:  # Check if the parameter contains '='
                    key, value = param.split('=')
                    if value.startswith('[') and value.endswith(']'):
                        current_run[key] = parse_list(value)
                    else:
                        try:
                            current_run[key] = eval(value)  # For non-list values
                        except:
                            current_run[key] = value  # If eval fails, keep the raw string
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

# Define the mapping for methods based on lambda values
method_mapping = {
    (0, 0): 'Origin',
    (20, 0): 'ComFair',
    (20, 10): 'FairLISA'
}

# Function to create a table for a specific missing ratio
def create_table_for_missing_ratio(df, missing_ratio):
    final_columns = ['Model', 'Method', 'AUC-G', 'AUC-A', 'AUC-O', 'RMSE']
    final_data = []
    df_filtered = df[df['MISSING_RATIO'] == missing_ratio]
    for model in ['PMF', 'NCF']:
        for lambdas, method in method_mapping.items():
            lambda_2, lambda_3 = lambdas
            if method == 'Origin':
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
                                         (df_filtered['LAMBDA_2'] == lambda_2) &
                                         (df_filtered['LAMBDA_3'] == lambda_3)]
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

# Create tables for missing ratios 0.2 and 0.4
table_02 = create_table_for_missing_ratio(df, 0.2)
table_04 = create_table_for_missing_ratio(df, 0.4)

# Plotting
missing_ratios = [0.2, 0.4, 0.6, 0.8, 0.95]
features = ['AUC for Gender', 'AUC for Age', 'AUC for Occupation']
colors = {'Origin': 'blue', 'ComFair': 'green', 'FairLISA': 'red'}
# Reverse mapping from method name to lambda values
reverse_method_mapping = {v: k for k, v in method_mapping.items()}

# Plotting
for model in ['PMF', 'NCF']:
    for feature in features:
        plt.figure(figsize=(10, 6))
        for method, color in colors.items():
            y_values = []
            lambda_2, lambda_3 = reverse_method_mapping[method]
            for ratio in missing_ratios:
                if method == 'Origin':
                    # Use replacement data for Origin
                    y_values.append(origin_replacement_data[model][feature])
                else:
                    ratio_data = df[(df['MODEL'] == model) &
                                    (df['MISSING_RATIO'] == ratio) &
                                    (df['LAMBDA_2'] == lambda_2) &
                                    (df['LAMBDA_3'] == lambda_3)]
                    if not ratio_data.empty:
                        y_values.append(ratio_data.iloc[0][feature])
                    else:
                        y_values.append(None)
            plt.plot(missing_ratios, y_values, label=method, color=color, marker='o')

        plt.title(f'{model} - {feature}')
        plt.xlabel('Missing Ratio (%)')
        plt.ylabel('AUC')
        plt.xticks(missing_ratios, [f'{int(ratio*100)}%' for ratio in missing_ratios])
        plt.legend()
        plt.show()
