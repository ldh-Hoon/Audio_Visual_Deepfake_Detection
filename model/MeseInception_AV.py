'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is mainly modified from the below link:
https://github.com/HongguLiu/MesoNet-Pytorch
'''
import torch
import torch.nn as nn

from var import *
from model.encoder import *

class MeseInception_AV(nn.Module):
    def __init__(self, num_classes=2):
        super(MeseInception_AV, self).__init__()
        self.num_classes = num_classes

        # Inception Layer 1
        self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # Inception Layer 2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(input_size=60, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(272, 16)
        self.fc2 = nn.Linear(16, self.num_classes)

    # Inception Layer 1
    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)

        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)

        return y


    def features(self, input):
        x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
        x = self.InceptionLayer2(x)      # (Batch, 12, 64, 64)

        x = self.conv1(x)                 # (Batch, 16, 64, 64)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)           # (Batch, 16, 32, 32)

        x = self.conv2(x)                 # (Batch, 16, 32, 32)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)           # (Batch, 16, 8, 8)

        x = x.reshape(x.size(0), -1)         # (Batch, 16 * 8 * 8)

        return x

    def classifier(self, feature, audio_features):
        combined = torch.cat((feature, audio_features), dim=1)
        out = self.dropout(combined)
        out = self.fc1(out) 
        out = self.leakyrelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def forward(self, video_input, audio_input):
        batch_size, num_frames, c, h, w = video_input.size() 

        video_outputs = []
        for i in range(num_frames):
            frame = video_input[:, i, :, :, :] 
            video_feature = self.features(frame)
            video_outputs.append(video_feature)

        video_outputs = torch.mean(torch.stack(video_outputs), dim=0) 

        audio_input = audio_input.view(batch_size, -1)
        frames = audio_input.size(1) // 60
        x_audio = audio_input.reshape(batch_size, frames, -1)

        x_audio_lstm, _ = self.lstm(x_audio)

        x_audio_last = x_audio_lstm[:, -1, :] 


        return self.classifier(video_outputs, x_audio_last)