import torch
import torch.nn as nn
import torch.nn.functional as F

from var import *
from model.encoder import *


class CNN_LSTM_V(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_LSTM_V, self).__init__()

        self.cnn = VideoCNN()
        self.lstm = nn.LSTM(input_size=COMBINE_SIZE-N_MFCC, hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(LSTM_DIM*2, 128)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, video_input, audio):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        x_video = self.cnn(x_video)
        x_video = x_video.reshape(batch_size, frames, -1)

        x_lstm, _ = self.lstm(x_video)
        x = x_lstm[:, -1, :]

        x = self.dropout(x)
        x = self.fc(x)
        return x
