import os
import sys
import torch
import numpy as np
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler

sys.path.append('.')
sys.path.append('..')

def load_data_list(path, sub_info, n_length, device):
    """
    Load and preprocess data for multiple tasks and subjects.

    Args:
    path (str): Path to the data directory.
    sub_info (list): List of subject IDs.
    n_length (int): Number of time points to include.
    device (torch.device): Device to load the data onto.

    Returns:
    list: Preprocessed data for all tasks and subjects.
    """
    task_info = os.listdir(path)
    data = []

    for task in task_info:
        data_tsk = []

        for sub in sub_info:
            file = os.path.join(path, task, f'{str(sub)}_{task}.npy')
            signal = np.load(file, allow_pickle=True)

            # Normalize the signal
            scaler = StandardScaler()
            signal = scaler.fit_transform(signal)

            # Transpose and truncate the signal
            signal = np.transpose(signal)
            signal = signal[:n_length, :]

            # Convert to tensor and move to specified device
            signal = torch.from_numpy(signal).to(torch.float32).to(device)

            data_tsk.append(signal)

        data.append(data_tsk)

    return data

class MyDataset(Data.Dataset):
    """
    Custom Dataset class for paired task data.
    """
    def __init__(self, data_list, tsk_pair=None):
        self.data_list = data_list
        self.tsk_pair = tsk_pair
    
    def __len__(self):
        return len(self.data_list[0])
    
    def __getitem__(self, index):
        tsk0 = self.tsk_pair[0]
        tsk1 = self.tsk_pair[1]

        data0 = self.data_list[tsk0]
        data1 = self.data_list[tsk1]

        return data0[index], data1[index]

def load_data(path, tsk, sub_info, n_time, n_rois):
    """
    Load and preprocess data for a specific task.

    Args:
    path (str): Path to the data directory.
    tsk (str): Task name.
    sub_info (list): List of subject IDs.
    n_time (int): Number of time points to include.
    n_rois (int): Number of regions of interest.

    Returns:
    numpy.ndarray: Preprocessed signals for all subjects.
    """
    n_subs = len(sub_info)
    signals = np.zeros((n_subs, n_time, n_rois))

    file_type = 'rfMRI' if tsk in ['REST1', 'REST2'] else 'tfMRI'
    run = 'LR' if 'train' in path or 'generalization' in path else 'RL'

    for i, sub in enumerate(sub_info):
        file_path = os.path.join(path, f'{file_type}_{tsk}_{run}', f'{sub}_{file_type}_{tsk}_{run}.npy')
        signal = np.load(file_path)

        # Truncate the signal to the specified time points
        signal = signal[:, :n_time]

        # Normalize the signal
        scaler = StandardScaler()
        signal = scaler.fit_transform(signal)

        # Transpose the signal
        signal = np.transpose(signal)

        signals[i, :, :] = signal
    
    return signals