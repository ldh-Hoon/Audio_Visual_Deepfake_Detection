import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig

from var import *
from model.encoder import *

class CNN_BiMAMBA_each_AV(nn.Module):
    def __init__(self, num_classes=2, num_mfcc=60):
        super(CNN_BiMAMBA_each_AV, self).__init__()

        self.video_cnn = VideoCNN()

        self.video_forward_config = MambaConfig(d_model=COMBINE_SIZE-num_mfcc, n_layers=2)
        self.video_backward_config = MambaConfig(d_model=COMBINE_SIZE-num_mfcc, n_layers=2)

        self.audio_forward_config = MambaConfig(d_model=num_mfcc, n_layers=2)
        self.audio_backward_config = MambaConfig(d_model=num_mfcc, n_layers=2)

        self.video_forward_mamba = Mamba(self.video_forward_config)
        self.video_backward_mamba = Mamba(self.video_backward_config)

        self.audio_forward_mamba = Mamba(self.audio_forward_config)
        self.audio_backward_mamba = Mamba(self.audio_backward_config)

        self.fc = nn.Linear(COMBINE_SIZE * 2, COMBINE_SIZE)
        self.fc2 = nn.Linear(COMBINE_SIZE, num_classes) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, video_input, audio_input):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        x_video = self.video_cnn(x_video)
        x_video = x_video.reshape(batch_size, frames, -1)

        audio_input = audio_input.view(batch_size, -1)  # (Batch, Features)

        num_mfcc_features = audio_input.size(1) // N_MFCC  # N_MFCC는 MFCC의 개수
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)  # (Batch, Frames, N_MFCC)

        x_video_forward = self.video_forward_mamba(x_video)
        x_video_backward = self.video_backward_mamba(x_video.flip(dims=[1]))
        x_video_backward = x_video_backward.flip(dims=[1])
        x_video_combined = torch.cat((x_video_forward, x_video_backward), dim=-1)

        x_audio_forward = self.audio_forward_mamba(x_audio)
        x_audio_backward = self.audio_backward_mamba(x_audio.flip(dims=[1]))
        x_audio_backward = x_audio_backward.flip(dims=[1])
        x_audio_combined = torch.cat((x_audio_forward, x_audio_backward), dim=-1)

        x_combined = torch.cat((x_video_combined, x_audio_combined), dim=-1)

        x_last = x_combined[:, -1, :]

        x = self.dropout(x_last)
        x = self.fc(x)
        x = self.fc2(x)
        return x
