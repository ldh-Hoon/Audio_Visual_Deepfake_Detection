import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from var import *
from model.encoder import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KAN_CNN_Transformer_AV(nn.Module):
    def __init__(self, num_classes=2, num_mfcc=60, combine_size=512, num_heads=4, num_layers=2):
        super(KAN_CNN_Transformer_AV, self).__init__()

        self.video_cnn = VideoCNN()

        self.d_model = combine_size - num_mfcc

        self.video_transformer = TransformerEncoder(2, emb_size=self.d_model)
        self.audio_transformer = TransformerEncoder(2, emb_size=num_mfcc)

        self.fc = nn.Linear(self.d_model + num_mfcc, self.d_model)
        self.fc2 = nn.Linear(self.d_model, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, video_input, audio_input):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        
        x_video = self.video_cnn(x_video)  # (batch_size * frames, features)
        
        x_video = x_video.reshape(batch_size, frames, -1)

        audio_input = audio_input.view(batch_size, -1)  # (Batch, Features)

        num_mfcc_features = audio_input.size(1) // N_MFCC
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)  # (Batch, Frames, N_MFCC)

        # Prepare inputs for Transformer
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

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 16, num_heads: int = 4, dropout: float = 0,g=5, k=3):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = ReLUKAN([emb_size, emb_size * 3], g, k)
        self.att_drop = nn.Dropout(dropout)
        self.projection = ReLUKAN([emb_size, emb_size], g, k)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        b, E, _ = x.shape
        x = rearrange(x, 'b e n -> (b e) n')
        x = self.qkv(x)
        x = rearrange(x, '(b e) n 1 -> b e n', b=b)
        
        qkv = rearrange(x, "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)

        out = rearrange(out, "b h n d -> (b n) (h d)")
        out = self.projection(out)
        out = rearrange(out, '(b e) n 1 -> b e n', b=b)
        return out


# img_size // patch = patch_size  
# ex) 28 // 7 = 4
# 
class PatchEmbedding(nn.Module): 
    def __init__(self, in_channels: int = 1, patch: int = 7, 
                 emb_size: int = 16, img_size: int = 28):
        self.patch_size = patch
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch, stride=patch),
            Rearrange('b e (h) (w) -> b (h w) e')).to(device)
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size)).to(device)
        self.positions = nn.Parameter(torch.randn((img_size // patch) **2 + 1, emb_size)).to(device)
        
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        x = torch.cat([cls_tokens, x], dim=1)

        x += self.positions

        return x
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0., g=5, k=3):
        super().__init__()
        self.relukan1 = ReLUKAN([emb_size, expansion * emb_size], g, k)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_p)
        self.relukan2 = ReLUKAN([expansion * emb_size, emb_size], g, k)

    def forward(self, x):
        batch_size, E, o = x.size()
        x = rearrange(x, 'b e o -> (b e) o')
        x = self.relukan1(x)

        x = self.gelu(x)
        x = self.dropout(x)

        x = x.squeeze(-1)
        x = self.relukan2(x)

        x = x.reshape(batch_size, -1, o)
        return x

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 16,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads=4),
                nn.Dropout(drop_p)
            )).to(device),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ).to(device))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 16, n_classes: int = 10, grid_size=5, k=3):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            ReLUKAN([emb_size, 10], grid_size, k))
        
class ReLU_KAN_Transformer(nn.Module):
    def __init__(self, d_model=16, num_heads=4, depth=6, n_classes: int = 10, grid_size: int = 5, k: int = 3):
        super().__init__()
        self.PE = PatchEmbedding(1, 7, d_model, 28)
        self.TE = TransformerEncoder(depth, emb_size=d_model)
        self.CH = ClassificationHead(emb_size=d_model, n_classes = 10, grid_size=grid_size, k=k)

    def forward(self, x):
        x = self.PE(x)
        x = self.TE(x)
        x = self.CH(x)
        x = F.log_softmax(x, dim=1)
        x = x.squeeze(-1)
        return x

