from datetime import datetime
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from recommenders.models.ncf.ncf_singlenode import NCF
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def calculate_auc_score_for_feature(feature_true, feature_pred):
    n_classes = len(np.unique(feature_true))

    # Binarize the true labels for OvR calculation
    true_binarized = label_binarize(feature_true, classes=np.arange(n_classes))

    # Predictions should be an array with shape [n_samples, n_classes]
    pred = np.array(feature_pred)

    if n_classes == 2:
        # Binary classification
        return roc_auc_score(true_binarized, pred[:, 1])
    else:
        # Multi-class classification
        return roc_auc_score(true_binarized, pred, multi_class='ovr')


def split_dataset(data, missing_ratio, seed):
    """
    Split the dataset based on a specified MISSING_RATIO with a seed for reproducibility.

    Args:
    data (DataFrame): The original dataset.
    missing_ratio (float): The ratio of data to be placed in train_data_nofeature.
    seed (int): The seed for the random number generator.

    Returns:
    tuple: Two DataFrames, train_data and train_data_nofeature.
    """
    # Set the seed
    np.random.seed(seed)

    # Calculate the number of records needed for the desired ratio
    target_record_count = int(len(data) * (1 - missing_ratio))

    # Get unique user IDs
    unique_user_ids = data['UserID'].unique()

    # Shuffle the user IDs
    np.random.shuffle(unique_user_ids)

    # Initialize variables to keep track of selected users and record count
    selected_users = []
    selected_record_count = 0

    # Select users until the target record count is reached
    for user_id in unique_user_ids:
        user_record_count = len(data[data['UserID'] == user_id])
        if selected_record_count + user_record_count > target_record_count:
            break
        selected_users.append(user_id)
        selected_record_count += user_record_count

    # Split the dataset
    train_data = data[data['UserID'].isin(selected_users)]
    train_data_nofeature = data[~data['UserID'].isin(selected_users)]

    return train_data, train_data_nofeature


def write_args_to_file(args, file_path):
    # Current timestamp to identify this run
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine the name based on the args
    if args.ORIGIN:
        name = "Origin"
    elif args.LAMBDA_3 == 0:
        name = "ComFair"
    else:
        name = "FairLISA"

    # Convert args to string
    args_str = str(args)

    # Append the args to the file (use 'a' for append mode)
    with open(file_path, "a") as file:
        file.write(name + "\n")
        file.write(f"Run Timestamp: {current_time}\n")
        file.write(args_str + "\n\n")
    return current_time


def load_ncf_model(model_dir, base_path, model_type='mlp', n_factors=8, layer_sizes=[64, 32, 16, 8], n_epochs=20,
                   batch_size=256, learning_rate=0.001, verbose=1):
    nonsplit_data = pd.read_csv(base_path + "merged_dataset.csv")

    model = NCF(
        n_users=nonsplit_data['UserID'].nunique(),
        n_items=nonsplit_data['MovieID'].nunique(),
        model_type=model_type,
        n_factors=n_factors,
        layer_sizes=layer_sizes,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose
    )

    if model_type == 'mlp':
        model.load(mlp_dir=model_dir)

    # Load the dictionaries from the file
    with open(model_dir + '/user2id.pkl', 'rb') as f:
        model.user2id = pickle.load(f)
    with open(model_dir + '/item2id.pkl', 'rb') as f:
        model.item2id = pickle.load(f)
    with open(model_dir + '/id2user.pkl', 'rb') as f:
        model.id2user = pickle.load(f)
    with open(model_dir + '/id2item.pkl', 'rb') as f:
        model.id2item = pickle.load(f)
    return model


def ncf_get_user_embeddings(model, user_ids):
    user_input = np.array([model.user2id[x] for x in user_ids])
    user_input_tensor = tf.convert_to_tensor(user_input, dtype=tf.int32)

    # Get user embeddings
    user_emb = tf.nn.embedding_lookup(params=model.embedding_mlp_P, ids=user_input_tensor)

    # Compute and retrieve the values of user embeddings
    with model.sess.as_default():
        user_emb_evaluated = user_emb.eval()

    return user_emb_evaluated


def ncf_get_movie_embeddings(model, movie_ids):
    item_input = np.array([model.item2id[x] for x in movie_ids])
    item_input_tensor = tf.convert_to_tensor(item_input, dtype=tf.int32)

    # Get user embeddings
    movie_embs = tf.nn.embedding_lookup(params=model.embedding_mlp_Q, ids=item_input_tensor)

    # Compute and retrieve the values of movie embeddings
    with model.sess.as_default():
        movie_embs = movie_embs.eval()

    return movie_embs


def save_table_latex(table, missing_ratio, base_dir, sub_dir):
    """Saves a pandas DataFrame as a LaTeX table in a specified directory.

    Args:
    table (pd.DataFrame): The DataFrame to be saved.
    missing_ratio (float): The missing ratio, used for naming the file.
    base_dir (str): The base directory where the file should be saved.
    sub_dir (str): The sub-directory under the base directory for the file.

    """
    filename = f'{sub_dir}_ratios_{missing_ratio}.tex'
    file_path = os.path.join(base_dir, sub_dir, filename)
    with open(file_path, 'w') as f:
        f.write(table.to_latex(index=False, caption=f'Table for Missing Ratio {missing_ratio}', label=f'tab:missing_{missing_ratio}'))
    print(f'Table for missing ratio {missing_ratio} saved as LaTeX file {filename}')


def plot_auc_performance(df, model, missing_ratios, base_dir, sub_dir, replace_origin=False,
                         origin_replacement_data=None, model_specific_addition=""):
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
    features = ['AUC for Gender', 'AUC for Age', 'AUC for Occupation']
    colors = {'Origin': 'red', 'ComFair': 'lime', 'FairLISA': 'blue'}
    line_styles = {'Origin': 'dashed', 'ComFair': 'dashed', 'FairLISA': 'dashed'}
    markers = {'Origin': None, 'ComFair': 'o', 'FairLISA': 'o'}
    alpha = 0.8

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    title = f'User Model: {model} {model_specific_addition}'
    fig.suptitle(title)

    for i, feature in enumerate(features):
        feature_name = feature.split(" ")[2]
        axs[i].set_title("Sensitive Attribute: " + feature_name)
        axs[i].grid(True)

        for method in colors.keys():
            y_values = []
            for ratio in missing_ratios:
                if replace_origin and method == 'Origin' and origin_replacement_data:
                    y_values.append(origin_replacement_data[model][feature])
                else:
                    ratio_data = df[(df['MODEL'] == model) &
                                    (df['MISSING_RATIO'] == ratio) &
                                    (df['Method'] == method)]
                    if not ratio_data.empty:
                        y_values.append(ratio_data.iloc[0][feature])
                    else:
                        y_values.append(None)

            axs[i].plot(missing_ratios, y_values, label=method, color=colors[method], linestyle=line_styles[method], marker=markers[method], alpha=alpha)
            axs[i].set_xlabel('Missing Ratio (%)')
            axs[i].set_ylabel('AUC')
            axs[i].set_xticks(missing_ratios)
            axs[i].set_xticklabels([f'{int(ratio*100)}%' for ratio in missing_ratios])

        if i == 0:
            legend = axs[i].legend()
            legend.get_frame().set_edgecolor('black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'{sub_dir}_ratios_{model}.png'
    plot_path = os.path.join(base_dir, sub_dir, plot_filename)
    plt.savefig(plot_path)
    plt.show()


def setup_file_path(base_dir, results_filename):
    """Sets up the file path for results, creating directories if they don't exist.

    Args:
    base_dir (str): The base directory where the results should be saved.
    results_filename (str): The filename of the results file.

    Returns:
    str: The full file path for the results file.
    """
    sub_dir = results_filename.split('.')[0]  # Directory name based on the file name

    # Create base_dir if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Create subdirectory inside base_dir
    sub_dir_path = os.path.join(base_dir, sub_dir)
    os.makedirs(sub_dir_path, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(sub_dir_path, results_filename)
    return file_path

