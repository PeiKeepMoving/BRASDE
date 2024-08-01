import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append('.')
sys.path.append('..')

def iteration(x1, x2, enc_ind, enc_grp, decoder):
    """
    Perform a single iteration of the BRASDE model.

    Args:
    x1, x2 (torch.Tensor): Input data for two tasks.
    enc_ind, enc_grp (torch.nn.Module): Individual and group encoders.
    decoder (torch.nn.Module): Decoder.

    Returns:
    tuple: Total loss, individual loss, group loss, and reconstruction loss.
    """
    half = int(len(x1)/2)
    
    # Encode and decode inputs
    x1_ind = decoder(enc_ind(x1))
    x1_grp = decoder(enc_grp(x1))
    x2_ind = decoder(enc_ind(x2))
    x2_grp = decoder(enc_grp(x2))

    # Reconstruction
    y1 = x1_ind + x1_grp
    y2 = x2_ind + x2_grp
    l_rec = F.mse_loss(x1, y1) + F.mse_loss(x2, y2)

    # Individual decoupling strategy
    x1_crs_ind = x1_grp + x2_ind
    x2_crs_ind = x2_grp + x1_ind
    l_ind = F.mse_loss(x1, x1_crs_ind) + F.mse_loss(x2, x2_crs_ind) \
        + (F.mse_loss(x1_ind, x2_ind) / (F.mse_loss(x1_ind[half:,:,:], x1_ind[:half,:,:]) + F.mse_loss(x2_ind[half:,:,:], x2_ind[:half,:,:])))

    # Group decoupling strategy
    x1_crs_grp = x1_ind + torch.concatenate((x2_grp[half:,:,:], x2_grp[:half,:,:]), axis=0)
    x2_crs_grp = x2_ind + torch.concatenate((x1_grp[half:,:,:], x1_grp[:half,:,:]), axis=0)
    l_grp = F.mse_loss(x1, x1_crs_grp) + F.mse_loss(x2, x2_crs_grp)

    loss = l_ind + 2*l_grp

    return loss, l_ind, l_grp, l_rec

def train_brasde(args, tsk_pairs, train_set, train_loader, valid_set, valid_loader, enc_ind, enc_grp, decoder, optim_ind, optim_grp, optim_dec):
    """
    Train the BRASDE model.

    Args:
    args: Training arguments.
    tsk_pairs: Task pairs for training.
    train_set, valid_set: Training and validation datasets.
    train_loader, valid_loader: Data loaders for training and validation.
    enc_ind, enc_grp, decoder: Model components.
    optim_ind, optim_grp, optim_dec: Optimizers for model components.
    """
    current_time = time.time()
    formatted_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(current_time))
    writer = SummaryWriter(os.path.join(args.path_log, formatted_time))

    min_valid_loss = float('inf')
    for epoch in range(args.n_epochs):
        train_list_loss, train_list_ind, train_list_grp, train_list_rec = [], [], [], []

        # Training loop
        for tsk_pair in tsk_pairs:
            train_set.tsk_pair = tsk_pair
            for x1, x2 in train_loader:
                # Zero gradients
                enc_ind.zero_grad()
                enc_grp.zero_grad()
                decoder.zero_grad()
                optim_ind.zero_grad()
                optim_grp.zero_grad()
                optim_dec.zero_grad()

                # Forward pass
                l, l_ind, l_grp, l_rec = iteration(x1, x2, enc_ind, enc_grp, decoder)

                # Backward pass and optimization
                l.backward()
                optim_ind.step()
                optim_grp.step()
                optim_dec.step()

                # Record losses
                train_list_loss.append(l.item())
                train_list_ind.append(l_ind.item())
                train_list_grp.append(l_grp.item())
                train_list_rec.append(l_rec.item())

        # Validation loop
        enc_ind.eval()
        enc_grp.eval()
        decoder.eval()
        valid_list_loss, valid_list_ind, valid_list_grp, valid_list_rec = [], [], [], []

        for tsk_pair in tsk_pairs:
            valid_set.tsk_pair = tsk_pair
            for x1, x2 in valid_loader:
                with torch.no_grad():
                    l, l_ind, l_grp, l_rec = iteration(x1, x2, enc_ind, enc_grp, decoder)
                valid_list_loss.append(l.item())
                valid_list_ind.append(l_ind.item())
                valid_list_grp.append(l_grp.item())
                valid_list_rec.append(l_rec.item())

        enc_ind.train()
        enc_grp.train()
        decoder.train()

        # Calculate average losses
        l_train = np.mean(train_list_loss)
        l_train_ind = np.mean(train_list_ind)
        l_train_grp = np.mean(train_list_grp)
        l_train_rec = np.mean(train_list_rec)

        l_valid = np.mean(valid_list_loss)
        l_valid_ind = np.mean(valid_list_ind)
        l_valid_grp = np.mean(valid_list_grp)
        l_valid_rec = np.mean(valid_list_rec)

        # Log losses
        writer.add_scalar('Loss/train', l_train, epoch)
        writer.add_scalar('Loss_ind/train', l_train_ind, epoch)
        writer.add_scalar('Loss_grp/train', l_train_grp, epoch)
        writer.add_scalar('Loss_rec/train', l_train_rec, epoch)

        writer.add_scalar('Loss/valid', l_valid, epoch)
        writer.add_scalar('Loss_ind/valid', l_valid_ind, epoch)
        writer.add_scalar('Loss_grp/valid', l_valid_grp, epoch)
        writer.add_scalar('Loss_rec/valid', l_valid_rec, epoch)

        print(f'{epoch+1}/{args.n_epochs}: loss_train:{l_train:.4f}\tloss_valid:{l_valid:.4f}\t\n')

        # Save best model
        if min_valid_loss > l_valid:
            min_valid_loss = l_valid
            model = {
                'enc_ind': enc_ind, 
                'enc_grp': enc_grp, 
                'decoder': decoder, 
                'args': args
            }
            np.save(os.path.join(args.path_model, 'brasde.npy'), model)

        # Save last model
        model = {
            'enc_ind': enc_ind, 
            'enc_grp': enc_grp, 
            'decoder': decoder, 
            'args': args
        }
        np.save(os.path.join(args.path_model, 'brasde_last.npy'), model)
