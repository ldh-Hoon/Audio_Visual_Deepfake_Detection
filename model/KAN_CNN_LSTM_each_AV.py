"""
This code ReLUKAN is modified based on
https://github.com/quiqi/relu_kan/blob/main/torch_relu_kan.py

"""

import numpy as np
import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from var import *
from model.encoder import *

class KAN_CNN_LSTM_each_AV(nn.Module):
    def __init__(self, num_classes=2, num_mfcc=60, grid_size=5, k=3):
        super(KAN_CNN_LSTM_each_AV, self).__init__()

        self.video_cnn = VideoCNN()

        self.video_lstm = nn.LSTM(input_size=(COMBINE_SIZE-60), hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(input_size=(60), hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)

        self.k1 = ReLUKAN([LSTM_DIM * 4, 128, num_classes], grid_size, k)
        self.dropout = nn.Dropout(0.3)

    def forward(self, video_input, audio_input):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        x_video = self.video_cnn(x_video)
        x_video = x_video.reshape(batch_size, frames, -1)

        audio_input = audio_input.view(batch_size, -1)  # (Batch, Features)

        num_mfcc_features = audio_input.size(1) // N_MFCC
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)  # (Batch, Frames, N_MFCC)

        x_video_lstm, _ = self.video_lstm(x_video)
        x_audio_lstm, _ = self.audio_lstm(x_audio)

        x_video_last = x_video_lstm[:, -1, :]  # (batch_size, LSTM_DIM * 2)
        x_audio_last = x_audio_lstm[:, -1, :]  # (batch_size, LSTM_DIM * 2)

        combined_features = torch.cat((x_video_last, x_audio_last), dim=1)  # (batch_size, LSTM_DIM * 4)

        x = self.dropout(combined_features)
        x = self.k1(x)
        x = x.squeeze(-1)
        return x

class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))
        
    def forward(self, x):
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_height - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size, 1))
        return x


class ReLUKAN(nn.Module):
    def __init__(self, width, grid, k):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], grid, k, width[i+1]))
            # if len(width) - i > 2:
            #     self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        x = x.unsqueeze(-1)
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        # x = x.reshape((len(x), self.width[-1]))
        return x