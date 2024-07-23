import os
import torch
import random
import numpy as np
from scipy.stats import pearsonr

def mk_pairs(task_list):
    """
    Generate all possible pairs of tasks from the given task list.

    Args:
    task_list (list): List of tasks.

    Returns:
    list: List of task pairs.
    """
    list_pairs = []
    for i in range(len(task_list)):
        for j in range(i+1, len(task_list)):
            pair = [task_list[i], task_list[j]]
            list_pairs.append(pair)
    return list_pairs

def train_valid_test_split(sub_info, train_size, valid_size, test_size):
    """
    Split subjects into training, validation, and test sets.

    Args:
    sub_info (numpy.ndarray or pandas.DataFrame): Subject information.
    train_size (float): Proportion of subjects for training.
    valid_size (float): Proportion of subjects for validation.
    test_size (float): Proportion of subjects for testing.

    Returns:
    tuple: Training, validation, and test subject information.
    """
    n_subs = len(sub_info)
    n_train = int(n_subs * train_size)
    n_valid = int(n_subs * valid_size)
    n_test = int(n_subs * test_size)

    rand_index = np.random.permutation(n_subs)
    id_train = rand_index[:n_train]
    id_valid = rand_index[n_train:n_train+n_valid]
    id_test = rand_index[n_train+n_valid:]
    
    if len(sub_info.shape) == 1:
        train_info = sub_info[id_train]
        valid_info = sub_info[id_valid]
        test_info = sub_info[id_test]
    else:
        train_info = sub_info.iloc[id_train,:]
        valid_info = sub_info.iloc[id_valid,:]
        test_info = sub_info.iloc[id_test,:]

    return train_info, valid_info, test_info

def extract_fc_matrix(signals):
    """
    Extract functional connectivity matrices from signals.

    Args:
    signals (numpy.ndarray or torch.Tensor): Input signals.

    Returns:
    numpy.ndarray: Functional connectivity matrices.
    """
    if torch.is_tensor(signals):
        signals = signals.detach().to('cpu').numpy()

    n_subs, _, n_rois = signals.shape
    features = np.zeros((n_subs, n_rois, n_rois))

    for i in range(n_subs):
        signal = signals[i,:,:]
        fc = np.corrcoef(signal, rowvar=False)
        features[i,:,:] = fc

    return features

def extract_fc(signals):
    """
    Extract functional connectivity features from signals.

    Args:
    signals (numpy.ndarray or torch.Tensor): Input signals.

    Returns:
    numpy.ndarray: Functional connectivity features.
    """
    if torch.is_tensor(signals):
        signals = signals.detach().to('cpu').numpy()

    n_subs, _, n_rois = signals.shape
    features = np.zeros((n_subs, int((n_rois*n_rois-n_rois)/2)))

    for i in range(n_subs):
        signal = signals[i,:,:]
        fc = np.corrcoef(signal, rowvar=False)
        fc = fc[np.triu_indices(n_rois, k=1)]
        fc = np.arctanh(fc)  # Fisher z-transform
        features[i,:] = fc

    return features

def compute_fc(signal):
    """
    Compute functional connectivity matrix for a single subject.

    Args:
    signal (numpy.ndarray or torch.Tensor): Input signal.

    Returns:
    numpy.ndarray: Functional connectivity matrix.
    """
    if torch.is_tensor(signal):
        signal = signal.detach().to('cpu').numpy()

    n_time, n_rois = signal.shape
    fc = np.zeros((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(i, n_rois):
            fc[i,j] = pearsonr(signal[:,i], signal[:,j])[0]
            fc[j,i] = fc[i,j]

    return fc   

def id_metric(corr):
    """
    Compute identification metrics from a correlation matrix.

    Args:
    corr (numpy.ndarray): Correlation matrix.

    Returns:
    tuple: Accuracy and various similarity metrics.
    """
    n = corr.shape[0]
    list_sim_intra, list_sim_inter, list_diff = [], [], []

    for i in range(n):
        sim_intra = corr[i,i]
        sim_inter = (np.sum(corr[i,:]) - corr[i,i]) / (n-1)
        diff = sim_intra - sim_inter

        list_sim_intra.append(sim_intra)
        list_sim_inter.append(sim_inter)
        list_diff.append(diff)

    avg_sim_intra = np.mean(list_sim_intra)
    std_sim_intra = np.std(list_sim_intra)
    avg_sim_inter = np.mean(list_sim_inter)
    std_sim_inter = np.std(list_sim_inter)
    avg_diff = np.mean(list_diff)
    std_diff = np.std(list_diff)

    # Compute accuracy
    max_in_row = np.argmax(corr, axis=1)
    max_on_diag = np.sum(max_in_row == np.arange(corr.shape[0]))
    acc = max_on_diag / n

    return acc, avg_diff, std_diff, avg_sim_intra, std_sim_intra, avg_sim_inter, std_sim_inter

def trans_1d_to_2d(vector, r):
    """
    Transform a 1D vector to a 2D symmetric matrix.

    Args:
    vector (numpy.ndarray): 1D input vector.
    r (int): Dimension of the output matrix.

    Returns:
    numpy.ndarray: 2D symmetric matrix.
    """
    matrix = np.zeros((r,r))
    k = 0
    for i in range(r):
        for j in range(i+1, r):
            matrix[i,j] = vector[k]
            matrix[j,i] = vector[k]
            k += 1
    return matrix