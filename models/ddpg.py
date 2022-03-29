from models.causal_cnn import CausalCNNEncoder

import copy
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##############
# DDPG Agent #
##############
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(depth=3,
                                            kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=hidden_dim,
                                            reduced_size=hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
    
    def forward(self, obs):
        x = F.relu(self.cnn_encoder(obs))
        x = self.net(x)
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(depth=3,
                                            kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=hidden_dim,
                                            reduced_size=hidden_dim)
        self.act_layer = nn.Linear(act_dim, hidden_dim)
        self.fc_layer = nn.Linear(hidden_dim, 1)
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, obs, act):
        x = F.relu(self.cnn_encoder(obs) + self.act_layer(act))
        x = self.net(x)
        return x.squeeze()

