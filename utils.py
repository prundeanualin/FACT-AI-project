import torch
import random
import numpy as np
import os


def model_has_knowledge_dimension(model_name):
    return model_name == "NCDM"


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

import statistics

datas = {
   'acc': [0.6566, 0.6567, 0.6552],
    'auc': [0.7160, 0.7158, 0.7108],
    'mae': [0.4235, 0.4234, 0.4229]
}