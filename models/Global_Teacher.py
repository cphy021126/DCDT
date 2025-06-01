import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange

# ViT Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (224 // patch_size) * (224 // patch_size) + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        patches = rearrange(patches, 'b e h w -> b (h w) e')
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x = x + self.pos_embedding
        return x


# Transformer Encoder
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, ff_dim=3072):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.layer_norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        return x


# Decoder Head
class ViTDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels):
        super(ViTDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x, output_size):
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        x = self.decoder(x)
        return torch.sigmoid(x)


# 主模型
class ViTTeacher(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072,
                 patch_size=16):
        super(ViTTeacher, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.decoder = ViTDecoder(embed_dim, num_classes)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        for layer in self.transformer_encoders:
            x = layer(x)
        x = x[:, 1:, :]  # remove cls token
        H_out = W_out = int(H // self.patch_size)
        x = x.permute(0, 2, 1).reshape(B, self.embed_dim, H_out, W_out)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        out = self.decoder(feat, output_size=x.shape[2:])
        return out


class ViTTeacherMC(ViTTeacher):
    def __init__(self, *args, **kwargs):
        super(ViTTeacherMC, self).__init__(*args, **kwargs)
        self.dropout_rate = 0.1

    def set_dropout(self, rate):
        self.dropout_rate = rate
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate

    def forward(self, x):
        self.set_dropout(self.dropout_rate)
        return super(ViTTeacherMC, self).forward(x)

    def predict_with_uncertainty(self, x, n_iter=30):
        predictions = []
        for _ in range(n_iter):
            out = self(x)
            predictions.append(out.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        return mean_pred, uncertainty