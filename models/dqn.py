from models.causal_cnn import CausalCNNEncoder
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#############
# DQN Agent #
#############
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(depth=3,
                                            kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=hidden_dim,
                                            reduced_size=hidden_dim)
        self.fc_layer = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, obs):
        x = F.relu(self.cnn_encoder(obs))
        x = self.fc_layer(x)
        return x

