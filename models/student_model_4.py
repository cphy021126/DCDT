import torch
import torch.nn as nn
from models.transunet_v2_localvit_final_fixed_v6_fixed import TransUNetV2
from models.attention_unet import AttU_Net

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 1x1 conv to generate attention map

    def forward(self, out_glb, out_loc):
        # 将两个分支的输出拼接在一起
        out_cat = torch.cat([out_glb, out_loc], dim=1)

        # 计算注意力图
        attention_map = torch.sigmoid(self.attn_conv(out_cat))  # Sigmoid to get weights between 0 and 1

        # 计算加权后的融合结果
        out_fused = attention_map * out_glb + (1 - attention_map) * out_loc  # 通过注意力加权融合

        return out_fused, attention_map

class StudentNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, vit_weights=None):
        super().__init__()

        # 完整结构的 TransUNet，结构与教师A完全一致，适用于 EMA
        self.branch_global = TransUNetV2(
            img_size=img_size,
            in_channels=1,
            num_classes=num_classes,
            vit_weights=vit_weights
        )

        # 完整结构的 AttentionUNet，结构与教师B完全一致，适用于 EMA
        self.branch_local = AttU_Net(img_ch=1, output_ch=num_classes)

        # # 输出融合：可替换为注意力加权等方式
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)
        # )
        # 使用 Attention Fusion 进行加权融合
        self.attention_fusion = AttentionFusion(in_channels=num_classes * 2)

    def forward(self, x):
        out_glb = self.branch_global(x)
        out_loc = self.branch_local(x)

        if out_glb.shape != out_loc.shape:
            raise ValueError(f"Shape mismatch: out_glb {out_glb.shape}, out_loc {out_loc.shape}")

        # out_cat = torch.cat([out_glb, out_loc], dim=1)
        # out_fused = self.fusion(out_cat)
        # 使用 Attention Fusion 融合两个分支
        out_fused, attention_map = self.attention_fusion(out_glb, out_loc)

        return out_fused, out_glb, out_loc, attention_map
