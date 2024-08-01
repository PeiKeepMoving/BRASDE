import os
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

sys.path.append('.')
sys.path.append('..')

from utils.utils import mk_pairs, train_valid_test_split
from utils.load_data import load_data_list, MyDataset
from model.brasde import Encoder, Decoder
from train.train_brasde import train_brasde

# Set up command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--path_sub', type=str, default='data/hcp_s1200_subjects_with_all_session_3T.csv')
parser.add_argument('--path_data', type=str, default='data/train')
parser.add_argument('--folder_model', type=str, default='results/models/brasde')
parser.add_argument('--folder_log', type=str, default='log/brasde')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--n_length', type=int, default=176)
parser.add_argument('--n_rois', type=int, default=360)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_hiddens', type=int, default=512)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr_ind', type=float, default=0.0001)  # Learning rate for individual encoder
parser.add_argument('--lr_grp', type=float, default=0.0001)  # Learning rate for group encoder
parser.add_argument('--lr_dec', type=float, default=0.0001)  # Learning rate for decoder
parser.add_argument('--wd', type=float, default=0.0001)  # Weight decay
args = parser.parse_args()

def main():
    # the length of time series
    time_list = [176]

    # Generate all possible task pairs
    tsk_pairs = [(i,j) for i in range(7) for j in range(i+1,7)]

    # Load subject information
    sub_info = pd.read_csv(args.path_sub)['Subject']

    # Split data into training, validation, and test sets
    train_info, valid_info, test_info = train_valid_test_split(sub_info, 0.7, 0.1, 0.2)

    args.test_info = test_info

    for n_time in time_list:
        args.n_length = n_time

        # Set up logging and model paths
        args.path_log = os.path.join(args.folder_log, str(n_time))
        args.path_model = os.path.join(args.folder_model, str(n_time))
        if not os.path.exists(args.path_model):
            os.mkdir(args.path_model)

        # Load training data
        train_data = load_data_list(args.path_data, train_info, args.n_length, args.device)
        train_set = MyDataset(train_data)
        train_loader = Data.DataLoader(train_set, batch_size=args.batchsize, shuffle=True, drop_last=True)

        # Load validation data
        valid_data = load_data_list(args.path_data, valid_info, args.n_length, args.device)
        valid_set = MyDataset(valid_data)
        valid_loader = Data.DataLoader(valid_set, batch_size=int(0.1*len(sub_info)), shuffle=True, drop_last=True)

        # Initialize models
        enc_ind = Encoder(args.n_rois, args.n_hiddens, args.n_layers, args.dropout).to(args.device)
        enc_grp = Encoder(args.n_rois, args.n_hiddens, args.n_layers, args.dropout).to(args.device)
        decoder = Decoder(args.n_rois, args.n_hiddens, args.n_layers, args.dropout).to(args.device)

        # Set up optimizers
        optim_ind = optim.Adam(enc_ind.parameters(), lr=args.lr_ind, weight_decay=args.wd)
        optim_grp = optim.Adam(enc_grp.parameters(), lr=args.lr_grp, weight_decay=args.wd)
        optim_dec = optim.Adam(decoder.parameters(), lr=args.lr_dec, weight_decay=args.wd)

        # Train the model
        train_brasde(args, tsk_pairs, train_set, train_loader, valid_set, valid_loader, enc_ind, enc_grp, decoder, optim_ind, optim_grp, optim_dec)

if __name__ == '__main__':
    # Set random seed for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
