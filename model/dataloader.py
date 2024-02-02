import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from utils import model_has_knowledge_dimension


def transform(model_name, batch_size, datasets, sensitive_features=None, item2knowledge=None):
    if model_has_knowledge_dimension(model_name):
        datasets_transformed = [
            transform_with_knowledge_dimension(
                batch_size=batch_size,
                user=data["user_id"],
                item=data["item_id"],
                item2knowledge=item2knowledge,
                score=data["score"],
                sensitive_features=data[sensitive_features] if sensitive_features is not None else None
            )
            for data in datasets
        ]
    else:
        datasets_transformed = [
            transform_without_knowledge_dimension(
                batch_size=batch_size,
                user=data["user_id"],
                item=data["item_id"],
                score=data["score"],
                sensitive_features=data[sensitive_features] if sensitive_features is not None else None
            )
            for data in datasets
        ]
    return datasets_transformed


def transform_without_knowledge_dimension(user, item, score, batch_size, sensitive_features=None):
    if sensitive_features is not None:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
            torch.tensor(np.array(sensitive_features), dtype=torch.float32),
        )
    else:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def transform_with_knowledge_dimension(user, item, item2knowledge, score, batch_size, sensitive_features=None):
    knowledge_emb = item2knowledge[list(item)]
    if sensitive_features is not None:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            torch.tensor(knowledge_emb, dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
            torch.tensor(np.array(sensitive_features), dtype=torch.float32),
        )
    else:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            torch.tensor(knowledge_emb, dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def attacker_transform(user, feature, batch_size):
    dataset = TensorDataset(
        torch.tensor(np.array(user), dtype=torch.int64),
        torch.tensor(np.array(feature), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)