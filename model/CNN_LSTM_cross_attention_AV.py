import torch
import torch.nn as nn
import torch.nn.functional as F

from var import *
from model.encoder import *

class CrossAttention(nn.Module):
    def __init__(self, dim, dropout=0.):
        super(CrossAttention, self).__init__()
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x_video, x_audio):
        q = self.to_q(x_video)
        k = self.to_k(x_audio)
        v = self.to_v(x_audio)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        return self.to_out(out)

class CNN_LSTM_cross_attention_AV(nn.Module):
    def __init__(self, num_classes=2, num_mfcc=60):
        super(CNN_LSTM_cross_attention_AV, self).__init__()

        self.video_cnn = VideoCNN()

        self.video_lstm = nn.LSTM(input_size=(COMBINE_SIZE - num_mfcc), hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(input_size=(num_mfcc), hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)

        self.cross_attention = CrossAttention(dim=LSTM_DIM * 2)

        self.final_lstm = nn.LSTM(input_size=LSTM_DIM * 2, hidden_size=LSTM_DIM, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(LSTM_DIM * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, video_input, audio_input):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        x_video = self.video_cnn(x_video)
        x_video = x_video.reshape(batch_size, frames, -1)

        audio_input = audio_input.view(batch_size, -1)

        num_mfcc_features = audio_input.size(1) // N_MFCC
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)

        x_video_lstm, _ = self.video_lstm(x_video)
        x_audio_lstm, _ = self.audio_lstm(x_audio)

        x_video_last = x_video_lstm[:, -1, :]
        x_audio_last = x_audio_lstm[:, -1, :]

        attention_output = self.cross_attention(x_video_lstm, x_audio_lstm)
        final_lstm_output, _ = self.final_lstm(attention_output)

        final_lstm_last = final_lstm_output[:, -1, :]

        x = self.dropout(final_lstm_last)
        x = self.fc1(x)

        x = self.dropout(x)
        x = self.fc2(x)

        return x
