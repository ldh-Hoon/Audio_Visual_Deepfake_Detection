import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101

from var import *
from model.encoder import *

class Resnet_LSTM_AV(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet_LSTM_AV, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, (COMBINE_SIZE-60)))
        self.lstm = nn.LSTM(input_size=(COMBINE_SIZE-60), hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(input_size=(60), hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(LSTM_DIM*4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, video, audio):
        batch_size, frames, channels, height, width = video.size()
        hidden = None
        for t in range(video.size(1)):
            with torch.no_grad():
                x = self.resnet(video[:, t, :, :, :])

                out, hidden = self.lstm(x.unsqueeze(1), hidden)

        audio_input = audio.view(batch_size, -1)  # (Batch, Features)
        num_mfcc_features = audio_input.size(1) // N_MFCC
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)  # (Batch, Frames, N_MFCC)
        x_audio_lstm, _ = self.audio_lstm(x_audio)

        x_video_last = out[:, -1, :] 
        x_audio_last = x_audio_lstm[:, -1, :]

        combined_features = torch.cat((x_video_last, x_audio_last), dim=1)
        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.fc2(x)
        return x