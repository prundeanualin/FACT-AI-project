import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt


sensitive_columns = ['Region', 'Gender', 'Family Education', 'Family Economic']
id_columns = ['MODEL', 'METHOD', 'LAMBDA_2', 'LAMBDA_3', 'MISSING_RATIO']
base_dir = '/Users/alinprundeanu/UvA/Year 1/Block 3/FACT/FACT-AI-project/results_cd/'
results_file = base_dir + 'results_as_table.csv'

def model_has_knowledge_dimension(model_name):
    return model_name == "NCDM"

def format_4dec(result):
    return float("{:.4f}".format(result))

def seed_experiments(seed):
    """
    Seed the experiment, for reproducibility
    :param seed: the random seed to be used
    :return: nothing
    """
    # Set `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(seed)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

def plot_missing_ratios(df, model, missing_ratios = [0.2, 0.4, 0.6, 0.8, 0.95]):
    """Plots AUC performance for a given model.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    model (str): The model name.
    missing_ratios (list): List of missing ratios.
    base_dir (str): Base directory for saving plots.
    sub_dir (str): Sub-directory for saving plots.
    replace_origin (bool): Whether to replace origin data.
    origin_replacement_data (dict, optional): Data to replace origin values.
    """
    features = ['AUC for Region', 'AUC for Gender', 'AUC for Family Education', 'AUC for Family Education']
    colors = {'Origin': 'red', 'ComFair': 'lime', 'FairLISA': 'blue'}
    line_styles = {'Origin': 'dashed', 'ComFair': 'dashed', 'FairLISA': '-'}
    markers = {'Origin': None, 'ComFair': 'o', 'FairLISA': 'o'}
    alpha = 0.8

    df = df.drop_duplicates(subset=[id_columns], inplace=False)
    lambda_2 = {
        'FairLISA': 2.0,
        'Origin': 0,
        'ComFair': 2.0
    }
    lambda_3 = {
        'FairLISA': 1.0,
        'Origin': 0,
        'ComFair': 0
    }

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    title = f'Cognitive Diagnosis with user model: {model}'
    fig.suptitle(title)

    for i, feature in enumerate(features):
        feature_name = "".join(feature.split(" ")[2:])
        axs[i].set_title("Sensitive Attribute: " + feature_name)
        axs[i].grid(True)

        for method in colors.keys():
            y_values = []
            for ratio in missing_ratios:
                ratio_data = df[(df['MODEL'] == model) &
                                (df['MISSING_RATIO'] == ratio) &
                                (df['METHOD'] == method) &
                                (df['LAMBDA_2'] == lambda_2[method]) &
                                (df['LAMBDA_3'] == lambda_3[method])]
                if not ratio_data.empty:
                    y_values.append(ratio_data.iloc[0][feature])

            axs[i].plot(missing_ratios, y_values, label=method, color=colors[method], linestyle=line_styles[method], marker=markers[method], alpha=alpha)
            axs[i].set_xlabel('Missing Ratio (%)')
            axs[i].set_ylabel('AUC')
            axs[i].set_xticks(missing_ratios)
            axs[i].set_xticklabels([f'{int(ratio*100)}%' for ratio in missing_ratios])

        if i == 0:
            legend = axs[i].legend(loc='upper left')
            legend.get_frame().set_edgecolor('black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = f'{base_dir}ratios_{model}.png'
    plt.savefig(plot_path)
    plt.show()


def plot_lambdas(df, model, lambdas = [0, 0.5, 1.0, 1.5]):

    df = df.drop_duplicates(subset=[id_columns], inplace=False)
    # We only plot the lambdas for FairLISA
    method = 'FairLISA'
    # Filter where lambda 2 is 0 and lambda 3 is any other value
    lambda_2_zero = df[(df['MODEL'] == model) &
                       (df['LAMBDA_2'] == 0) &
                       (df['METHOD'] == method) &
                       df['LAMBDA_3'].isin(lambdas)]

    # Filter where lambda 3 is 0 and lambda 2 is any other value
    lambda_3_zero = df[(df['MODEL'] == model) &
                       (df['LAMBDA_3'] == 0) &
                       (df['LAMBDA_2'].isin(lambdas)) &
                       (df['METHOD'] == method)]

    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Effect of different lambdas on fairness')

    for i, sens_attr_col in enumerate(sensitive_columns):
        ax = plt.subplot(1, 4, i+1)
        ax.plot(lambda_3_zero['LAMBDA_2'], lambda_3_zero[sens_attr_col], 'ro--', label=r'$\lambda_2$ $(U_L)$')
        ax.plot(lambda_2_zero['LAMBDA_3'], lambda_2_zero[sens_attr_col], 'bo-.', label=r'$\lambda_3$ $(U_N)$')
        all_vals = sorted(lambda_3_zero[sens_attr_col].tolist() + lambda_2_zero[sens_attr_col].tolist(), reverse=True)
        print(all_vals)
        ax.set_xlabel(f'Cognitive Diagnosis: {model} ({sensitive_columns[i]})')
        ax.set_ylabel('AUC')
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_xticks(lambdas)
        ax.grid(True)
        if i == 0:
            legend = ax.legend(loc='lower right')
            legend.get_frame().set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(f'{base_dir}lambdas_plot_explicit_{model}.png')  # Save plot
    plt.show()