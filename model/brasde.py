import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append('.')
sys.path.append('..')

class Encoder(nn.Module):
    """
    Encoder module for the BRASDE model.
    
    This module consists of a linear layer followed by a GRU (Gated Recurrent Unit).
    
    Args:
    n_rois (int): Number of regions of interest (input/output features).
    n_hiddens (int): Number of hidden units in the GRU.
    n_layers (int): Number of GRU layers (default: 1).
    dropout (float): Dropout rate (default: 0.5).
    """

    def __init__(self, n_rois, n_hiddens, n_layers=1, dropout=0.5):
        super().__init__()
        
        self.linear = nn.Linear(in_features=n_rois, out_features=n_rois)
        self.encoder = nn.GRU(n_rois, n_hiddens, num_layers=n_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        """
        Forward pass of the encoder.
        
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_rois).
        
        Returns:
        torch.Tensor: Encoded output of shape (batch_size, sequence_length, n_hiddens).
        """
        output = self.linear(x)
        output, _ = self.encoder(output)
        return output


class Decoder(nn.Module):
    """
    Decoder module for the BRASDE model.
    
    This module consists of a GRU (Gated Recurrent Unit) followed by a linear layer.
    
    Args:
    n_rois (int): Number of regions of interest (output features).
    n_hiddens (int): Number of hidden units in the GRU (input features).
    n_layers (int): Number of GRU layers (default: 1).
    dropout (float): Dropout rate (default: 0.5).
    """

    def __init__(self, n_rois, n_hiddens, n_layers=1, dropout=0.5):
        super().__init__()

        self.decoder = nn.GRU(n_hiddens, n_rois, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(in_features=n_rois, out_features=n_rois)

    def forward(self, x):
        """
        Forward pass of the decoder.
        
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_hiddens).
        
        Returns:
        torch.Tensor: Decoded output of shape (batch_size, sequence_length, n_rois).
        """
        output, _ = self.decoder(x)
        output = self.linear(output)
        return output