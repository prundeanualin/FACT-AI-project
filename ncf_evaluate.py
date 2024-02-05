import pandas as pd
import numpy as np
from utils_ml import load_ncf_model
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)

epoch = 20
model_type = 'mlp'
model_dir = "ncf_models/"
base_path = "data/ml-1m/"
binary = True
threshold = 3
model_path = f"{model_dir}thresh_{threshold}_epoch_{epoch}"

model = load_ncf_model(model_path, base_path)

# Load the validation set
validation_data = pd.read_csv(base_path + "validation.csv")

# Get user and item inputs for predictions
user_input = validation_data['UserID'].values
item_input = validation_data['MovieID'].values

# Predict ratings
y_pred = model.predict(user_input, item_input, is_list=True)

# Actual ratings
y_true = validation_data['Rating'].values
if binary:
    y_true = (y_true > threshold).astype(int)

y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

acc = accuracy_score(y_true, y_pred_binary)
prec = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
roc_auc = roc_auc_score(y_true, y_pred)

precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
pr_auc = auc(recall_curve, precision_curve)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

specificity = tn / (tn + fp)

print(f"Evaluation Results User Model (Filtered):")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print()
# Added these:
print(f"Precision: {prec:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print()
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
