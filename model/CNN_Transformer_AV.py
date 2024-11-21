import torch
import torch.nn as nn
import torch.nn.functional as F

from var import *
from model.encoder import *

class CNN_Transformer_AV(nn.Module):
    def __init__(self, num_classes=2, num_mfcc=60, combine_size=512, num_heads=4, num_layers=2):
        super(CNN_Transformer_AV, self).__init__()

        self.video_cnn = VideoCNN() 

        self.d_model = combine_size - num_mfcc

        # Transformer layers for video and audio
        self.video_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads),
            num_layers=num_layers
        )
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_mfcc, nhead=num_heads),
            num_layers=num_layers
        )

        self.fc = nn.Linear(self.d_model + num_mfcc, self.d_model)
        self.fc2 = nn.Linear(self.d_model, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, video_input, audio_input):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        x_video = self.video_cnn(x_video)  # (batch_size * frames, features)
        x_video = x_video.reshape(batch_size, frames, -1)  # (batch_size, frames, features)
        
        audio_input = audio_input.view(batch_size, -1)  # (Batch, Features)
        num_mfcc_features = audio_input.size(1) // N_MFCC
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)  # (Batch, Frames, N_MFCC)

        x_video = x_video.permute(1, 0, 2)  # Change to (frames, batch_size, features)
        x_audio = x_audio.permute(1, 0, 2)  # Change to (frames, batch_size, N_MFCC)

        x_video_transformed = self.video_transformer(x_video)  # (frames, batch_size, d_model)
        x_audio_transformed = self.audio_transformer(x_audio)  # (frames, batch_size, num_mfcc)

        x_video_last = x_video_transformed[-1, :, :]  # (batch_size, d_model)
        x_audio_last = x_audio_transformed[-1, :, :]  # (batch_size, num_mfcc)

        combined_features = torch.cat((x_video_last, x_audio_last), dim=1)  # (batch_size, d_model + num_mfcc)

        x = self.dropout(combined_features)
        x = self.fc(x)
        x = self.fc2(x)
        return x
