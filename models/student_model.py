import torch
import torch.nn as nn
from attention_unet import AttU_Net
from transunet_v2_localvit_final_fixed_v6_fixed import TransUNetV2

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

class StudentNet(nn.Module):
    def __init__(self, img_size=224, in_channels=1, num_classes=1, base_channels=128):
        super(StudentNet, self).__init__()
        self.branch_global = TransUNetV2(img_size=img_size, in_channels=in_channels, num_classes=num_classes)
        self.branch_local = AttU_Net(img_ch=in_channels, output_ch=num_classes)
        self.fusion = FeatureFusion(in_channels=num_classes, out_channels=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_global = self.branch_global(x)
        out_local = self.branch_local(x)
        fused = self.fusion(out_global, out_local)
        return self.sigmoid(fused), out_global, out_local  # 输出融合结果与两个分支单独输出
