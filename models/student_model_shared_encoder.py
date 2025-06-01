import torch
import torch.nn as nn
from models.attention_unet import AttU_Net
from models.transunet_v2_localvit_final_fixed_v6 import TransUNetV2

# ========== Shared Encoder ==========
class SharedEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(SharedEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.out_channels = base_channels * 4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# ========== Feature Fusion ==========
class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2):
        fused = torch.cat([feat1, feat2], dim=1)
        return self.conv(fused)

# ========== 学生网络主结构 ==========
class StudentNet(nn.Module):
    def __init__(self, img_size=224, in_channels=1, num_classes=1):
        super(StudentNet, self).__init__()
        self.encoder = SharedEncoder(in_channels=in_channels)

        # 创建两个分支：使用教师网络结构，但输入为 encoder 的输出特征
        #self.branch_global = TransUNetV2(img_size=img_size, in_channels=self.encoder.out_channels, num_classes=num_classes)
        self.branch_global = AttU_Net(img_ch=self.encoder.out_channels, output_ch=num_classes)

        self.branch_local = AttU_Net(img_ch=self.encoder.out_channels, output_ch=num_classes)

        self.fusion = FeatureFusion(in_channels=num_classes, out_channels=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shared_feat = self.encoder(x)
        out_global = self.branch_global(shared_feat)
        out_local = self.branch_local(shared_feat)
        fused = self.fusion(out_global, out_local)
        return self.sigmoid(fused), out_global, out_local
