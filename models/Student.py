# # models/Student.py
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.Global_Teacher import ViTTeacher
# from models.Boundary_Teacher import DeepLabTeacher
#
# # CNN 分支复用 DeepLab encoder 输出 2048 通道
# class CNNBranch(nn.Module):
#     def __init__(self):
#         super(CNNBranch, self).__init__()
#         self.encoder = DeepLabTeacher().model.backbone
#
#     def forward(self, x):
#         return self.encoder(x)['out']  # shape: [B, 2048, H/8, W/8]
#
#
# # ViT 分支复用 ViTTeacher encoder 输出 768 通道
# class ViTBranch(nn.Module):
#     def __init__(self):
#         super(ViTBranch, self).__init__()
#         self.vit = ViTTeacher()
#
#     def forward(self, x):
#         return self.vit.forward_features(x)  # shape: [B, 768, H/16, W/16]
#
#
# # 解码器，不包含 sigmoid
# class Decoder(nn.Module):
#     def __init__(self, in_channels, mid_channels=512, out_channels=1):
#         super(Decoder, self).__init__()
#         self.fusion = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=1)
#         )
#
#     def forward(self, x):
#         return self.fusion(x)  # 输出 raw logits
#
#
# # 主学生模型
# class DualBranchStudent(nn.Module):
#     def __init__(self):
#         super(DualBranchStudent, self).__init__()
#         self.vit_branch = ViTBranch()
#         self.cnn_branch = CNNBranch()
#
#         vit_out = 768
#         cnn_out = 2048
#         self.decoder = Decoder(in_channels=vit_out + cnn_out, out_channels=1)
#
#     def forward(self, x):
#         feat_vit = self.vit_branch(x)  # [B, 768, H/16, W/16]
#         feat_cnn = self.cnn_branch(x)  # [B, 2048, H/8, W/8]
#
#         # 对齐特征图尺寸（以 CNN 输出为目标）
#         if feat_vit.shape[2:] != feat_cnn.shape[2:]:
#             feat_vit = F.interpolate(feat_vit, size=feat_cnn.shape[2:], mode='bilinear', align_corners=False)
#
#         feat_fused = torch.cat([feat_vit, feat_cnn], dim=1)  # [B, vit+cnn, H, W]
#         out = self.decoder(feat_fused)  # [B, 1, H, W]，未经过 sigmoid
#         return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Global_Teacher import ViTTeacher
from models.Boundary_Teacher import DeepLabTeacher


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(64 + 64, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4, f5 = features

        d4 = self.up4(f5)
        d4 = self.conv4(torch.cat([d4, f4], dim=1))

        d3 = self.up3(d4)
        d3 = self.conv3(torch.cat([d3, f3], dim=1))

        d2 = self.up2(d3)
        d2 = self.conv2(torch.cat([d2, f2], dim=1))

        d1 = self.up1(d2)
        d1 = self.conv1(torch.cat([d1, f1], dim=1))

        out = self.out_conv(d1)
        return out


class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = DeepLabTeacher().model.backbone
        self.stage1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)  # 64
        self.stage2 = backbone.layer1  # 256
        self.stage3 = backbone.layer2  # 512
        self.stage4 = backbone.layer3  # 1024
        self.stage5 = backbone.layer4  # 2048

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return [f1, f2, f3, f4, f5]


class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTTeacher()

    def forward(self, x):
        feat = self.vit.forward_features(x)  # [B, C, H/16, W/16]
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)  # up to H/8
        return [None, None, None, None, feat]  # match with 5-layer list


class DualBranchStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_branch = CNNBranch()
        self.vit_branch = ViTBranch()
        self.decoder = Decoder()

    def forward(self, x):
        cnn_feats = self.cnn_branch(x)
        vit_feats = self.vit_branch(x)

        fused_feats = []
        for c, v in zip(cnn_feats, vit_feats):
            if c is None and v is not None:
                fused_feats.append(v)
            elif c is not None and v is not None:
                if v.shape[2:] != c.shape[2:]:
                    v = F.interpolate(v, size=c.shape[2:], mode='bilinear', align_corners=False)
                fused_feats.append(torch.cat([c, v], dim=1))
            else:
                fused_feats.append(c)

        out = self.decoder(fused_feats)
        return out
