import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

from var import *

class DwfFusion(nn.Module):
    def __init__(self, num_hidden_layers=2, hidden_size=1024, num_attention_heads=16, intermediate_size=400):
        super(DwfFusion, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.mhca_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
            for _ in range(num_hidden_layers)
        ])

        self.linear_WF = nn.Linear(hidden_size, hidden_size)  # Linear transformation for video
        self.linear_WA = nn.Linear(hidden_size, hidden_size)  # Linear transformation for audio

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
        )

        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_hidden_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_hidden_layers)])

    def calculate_attention_weights(self, Q, K):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        return F.softmax(scores, dim=-1)

    def forward(self, Aclass, Fclass):
        Q_f = self.linear_WF(Fclass)  # Query for video
        K_a = self.linear_WA(Aclass)  # Key for audio
        beta_fa = self.calculate_attention_weights(Q_f, K_a)  # Attention weights for video

        Q_a = self.linear_WA(Aclass)  # Query for audio
        K_f = self.linear_WF(Fclass)  # Key for video
        beta_af = self.calculate_attention_weights(Q_a, K_f)  # Attention weights for audio

        W_F = torch.mean(beta_fa, dim=1)  # Average across heads for video
        W_A = torch.mean(beta_af, dim=1)  # Average across heads for audio

        for i in range(self.num_hidden_layers):
            F_attention, _ = self.mhca_layers[i](Fclass, Fclass, Fclass)  # Video attention
            A_attention, _ = self.mhca_layers[i](Aclass, Aclass, Aclass)  # Audio attention

            Fclass = self.layer_norms1[i](F_attention + Fclass)
            Aclass = self.layer_norms2[i](A_attention + Aclass)

            Fclass = self.ffn(Fclass)
            Aclass = self.ffn(Aclass)

        V = W_F.unsqueeze(1) * Fclass + W_A.unsqueeze(1) * Aclass  # Combine modalities

        return V, Fclass, Aclass


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            x = self.dropout(x)

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size=112, patch_size=112, num_patches=200,num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x
        # return self.mlp_head(x)


class ViT_audio(nn.Module):
    def __init__(self, *, image_size=30, patch_size=20, num_patches=30,num_classes=2,  dim=1024, depth=6, heads=16, mlp_dim=2048, pool = 'cls', channels = 1, dim_head = 64, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        patch_height, patch_width = pair(patch_size)

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x
        # return self.mlp_head(x)
        


class AVT_DWF_AV(nn.Module):
    def __init__(self, num_classes=2):
        super(AVT_DWF_AV, self).__init__()
        self.vit_image = ViT()
        self.vit_audio = ViT_audio()
        self.fusion = DwfFusion(num_hidden_layers=2, hidden_size=1024, intermediate_size=400)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, img, audio):
        img_features = self.vit_image(self.pad_frames(img))
        audio_features = self.vit_audio(self.pad_frames(audio))

        img_features = img_features.float()
        audio_features = audio_features.float()

        joint_emb, hidden_states, weight_norm = self.fusion(img_features, audio_features)

        output = self.fc(joint_emb)
        return output
    
    def pad_frames(self, tensor, target_frames=200):
        if tensor.dim() == 3:  # shape: (b, f, n)
            b, f, n = tensor.shape
            if f > target_frames:
                out = tensor[:, :target_frames, :]
            else:
                padding_size = target_frames - f
                out = F.pad(tensor, (0, 0, 0, padding_size))  # (n, f) -> (n, f + padding_size)
            
        elif tensor.dim() == 5:  # shape: (b, f, c, h, w)
            b, f, c, h, w = tensor.shape
            if f > target_frames:
                out = tensor[:, :target_frames, :, :, :]
            else:
                padding_size = target_frames - f
                out = F.pad(tensor, (0, 0, 0, 0, 0, 0, 0, padding_size))  # (w, h, c, f + padding_size)
        
        return out