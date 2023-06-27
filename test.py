

import os 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym
import torch
import torch.nn as nn

import numpy as np


temp = torch.from_numpy(np.ones((7,45*2+1,45*2+1,45*2))).float()

class CustomCNN(BaseFeaturesExtractor):


    def __init__(self, observation_space: np.array, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn1 = nn.Sequential(
            nn.Conv3d(n_input_channels, 14, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv3d(14, 28, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )


        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten1 = self.cnn1(
                observation_space).float()
            
            n_flatten = self.cnn2(
                n_flatten1).double().shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn1 = self.cnn1(observations)
        print(cnn1.shape)
        cnn2 = self.cnn2(cnn1)
        print(cnn2.shape)
        last = self.linear(cnn2)
        print(last.shape)
        return last
    
model = CustomCNN(temp)

model(temp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))
