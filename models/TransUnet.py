import torch
import torch.nn as nn
import timm
from einops import rearrange


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        width = int(out_channels * (base_width / 64))
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return self.relu(out + identity)


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_skip=None):
        x = self.upsample(x)
        if x_skip is not None:
            x = torch.cat([x_skip, x], dim=1)
        return self.layer(x)


class ViTAdapter(nn.Module):
    def __init__(self, img_size=128, in_channels=1):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        # 修改 patch embedding 以接受灰度图输入
        self.vit.patch_embed.proj = nn.Conv2d(in_channels, self.vit.embed_dim,
                                              kernel_size=16, stride=16)
        self.img_size = img_size
        self.grid_size = img_size // 16
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)  # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.vit.pos_drop(x + self.vit.pos_embed[:, 1:, :])
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        x = x.transpose(1, 2).view(B, self.embed_dim, self.grid_size, self.grid_size)
        return x  # shape: [B, 768, H/16, W/16]


class TransUNetV2(nn.Module):
    def __init__(self, img_size=128, in_channels=1, base_channels=128, num_classes=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.enc1 = EncoderBottleneck(base_channels, base_channels * 2)
        self.enc2 = EncoderBottleneck(base_channels * 2, base_channels * 4)
        self.enc3 = EncoderBottleneck(base_channels * 4, base_channels * 8)

        self.vit = ViTAdapter(img_size=img_size, in_channels=in_channels)
        self.bridge = nn.Conv2d(768, base_channels * 8, kernel_size=1)

        self.dec1 = DecoderBottleneck(base_channels * 16, base_channels * 2)
        self.dec2 = DecoderBottleneck(base_channels * 4, base_channels)
        self.dec3 = DecoderBottleneck(base_channels * 2, base_channels // 2)
        self.dec4 = DecoderBottleneck(base_channels // 2, base_channels // 8)

        self.final = nn.Conv2d(base_channels // 8, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.conv1(x)       # [B, 128, 64, 64]
        x1 = self.enc1(x0)       # [B, 256, 32, 32]
        x2 = self.enc2(x1)       # [B, 512, 16, 16]
        x3 = self.enc3(x2)       # [B, 1024, 8, 8]

        vit_out = self.vit(x)    # [B, 768, 8, 8]
        bridge = self.bridge(vit_out)

        x = self.dec1(torch.cat([bridge, x3], dim=1), x2)
        x = self.dec2(x, x1)
        x = self.dec3(x, x0)
        x = self.dec4(x)
        out = self.final(x)
        return out
