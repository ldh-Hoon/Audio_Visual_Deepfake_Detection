import torch
import torch.nn as nn
import torch.nn.functional as F

from var import *
from model.encoder import *

class CNN_LSTM_AV(nn.Module):
    def __init__(self, num_classes=2, num_mfcc=60):
        super(CNN_LSTM_AV, self).__init__()

        self.video_cnn = VideoCNN()

        self.video_lstm = nn.LSTM(input_size=(COMBINE_SIZE), hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(LSTM_DIM * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, video_input, audio_input):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        x_video = self.video_cnn(x_video)
        x_video = x_video.reshape(batch_size, frames, -1)

        audio_input = audio_input.view(batch_size, -1)  # (Batch, Features)

        num_mfcc_features = audio_input.size(1) // N_MFCC
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)  # (Batch, Frames, N_MFCC)

        x = torch.cat((x_video, x_audio), dim=-1)
        x_video_lstm, _ = self.video_lstm(x)

        x_video_last = x_video_lstm[:, -1, :]  # (batch_size, LSTM_DIM * 2)

        x = self.dropout(x_video_last)
        x = self.fc1(x)

        x = self.dropout(x)
        x = self.fc2(x)

        return x
