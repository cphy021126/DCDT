import torch
import torch.nn as nn
import os
from timm.models.vision_transformer import vit_base_patch16_224

class ViTAdapter(nn.Module):
    def __init__(self, img_size, in_channels=1, pretrained_path=None):
        super().__init__()
        self.vit = vit_base_patch16_224(pretrained=False)
        if in_channels != 3:
            self.vit.patch_embed.proj = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16)
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'patch_embed.proj.weight' in state_dict and state_dict['patch_embed.proj.weight'].shape[1] != in_channels:
                state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].sum(dim=1, keepdim=True)
            self.vit.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)  # [B, C, H', W']
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        #x = x.flatten(2).transpose(1, 2)
        x = x.flatten(2)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        x = x[:, 1:, :]  # 去掉 cls token
        return x

class TransUNetV2(nn.Module):
    def __init__(self, img_size=224, in_channels=1, num_classes=1, vit_weights=None):
        super().__init__()
        base_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(base_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.vit = ViTAdapter(img_size, in_channels, vit_weights)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        if vit_weights and os.path.exists(vit_weights):
            state_dict = torch.load(vit_weights, map_location='cpu')
            if 'patch_embed.proj.weight' in state_dict and state_dict['patch_embed.proj.weight'].shape[1] != in_channels:
                state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].sum(dim=1, keepdim=True)
            for key in ["conv1.0.weight", "conv1.0.bias"]:
                if key in state_dict:
                    del state_dict[key]
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.encoder(x0)
        B, C, H, W = x1.shape
        vit_out = self.vit(x)
        vit_out = vit_out.permute(0, 2, 1).reshape(B, 768, H // 4, W // 4)
        out = self.decoder(vit_out)
        return out
