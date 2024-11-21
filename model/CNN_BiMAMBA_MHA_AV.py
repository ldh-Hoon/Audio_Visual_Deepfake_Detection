import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig

from var import *
from model.encoder import *

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        attn_output, _ = self.attention(x, x, x)  # self-attention
        return attn_output.permute(1, 0, 2)  # (batch_size, seq_len, d_model)

class CNN_BiMAMBA_MHA_AV(nn.Module):
    def __init__(self, num_classes=2, num_mfcc=60):
        super(CNN_BiMAMBA_MHA_AV, self).__init__()

        self.video_cnn = VideoCNN()

        self.v_forward_config = MambaConfig(d_model=COMBINE_SIZE-num_mfcc, n_layers=2)
        self.v_backward_config = MambaConfig(d_model=COMBINE_SIZE-num_mfcc, n_layers=2)

        self.a_forward_config = MambaConfig(d_model=num_mfcc, n_layers=2)
        self.a_backward_config = MambaConfig(d_model=num_mfcc, n_layers=2)

        self.v_forward_mamba = Mamba(self.v_forward_config)
        self.v_backward_mamba = Mamba(self.v_backward_config)

        self.a_forward_mamba = Mamba(self.a_forward_config)
        self.a_backward_mamba = Mamba(self.a_backward_config)

        self.v_mha = MultiHeadAttention(d_model=(COMBINE_SIZE-num_mfcc)*2, nhead=8)  # video MHA
        self.a_mha = MultiHeadAttention(d_model=num_mfcc*2, nhead=8)  # audio MHA

        self.fc = nn.Linear(COMBINE_SIZE*2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, video_input, audio_input):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        x_video = self.video_cnn(x_video)
        x_video = x_video.reshape(batch_size, frames, -1)

        x_v_forward = self.v_forward_mamba(x_video)
        x_v_backward = self.v_backward_mamba(x_video.flip(dims=[1])).flip(dims=[1])
        x_video_combined = torch.cat((x_v_forward, x_v_backward), dim=-1)
        x_video_mha = self.v_mha(x_video_combined)

        audio_input = audio_input.view(batch_size, -1)  # (Batch, Features)

        num_mfcc_features = audio_input.size(1) // N_MFCC
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)  # (Batch, Frames, N_MFCC)

        x_a_forward = self.a_forward_mamba(x_audio)
        x_a_backward = self.a_backward_mamba(x_audio.flip(dims=[1])).flip(dims=[1])
        x_audio_combined = torch.cat((x_a_forward, x_a_backward), dim=-1)
        x_audio_mha = self.a_mha(x_audio_combined)

        x_last_video = x_video_mha[:, -1, :]
        x_last_audio = x_audio_mha[:, -1, :]
        x_combined = torch.cat((x_last_video, x_last_audio), dim=-1)

        x = self.dropout(x_combined)
        x = self.fc(x)
        return x
