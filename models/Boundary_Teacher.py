import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabTeacher(nn.Module):
    """
    局部边界教师模型，使用 DeepLabv3+（以 ResNet-50 为 backbone）。
    支持 MC Dropout 模式，用于不确定性估计。
    """
    def __init__(self, num_classes=1, pretrained_backbone=True, dropout_rate=0.1):
        super(DeepLabTeacher, self).__init__()
        self.model = deeplabv3_resnet50(pretrained=pretrained_backbone)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # 替换分类头
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        features = self.model.backbone(x)['out']   # 特征提取
        features = self.dropout(features)          # MC Dropout
        output = self.model.classifier(features)   # 解码器输出
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(output)               # 输出概率图（适用于二分类）

    def forward_features(self, x):
        """输出 encoder 特征图"""
        features = self.model.backbone(x)['out']
        return features

    def predict_with_uncertainty(self, x, n_iter=10):
        """
        使用 MC Dropout 多次推理，计算均值和方差作为不确定性估计
        """
        self.train()
        preds = []

        for _ in range(n_iter):
            with torch.no_grad():
                pred = self.forward(x)  # [B, 1, H, W]
                preds.append(pred)

        preds = torch.stack(preds, dim=0)  # [T, B, 1, H, W]
        mean_pred = preds.mean(dim=0)
        uncertainty = preds.var(dim=0)
        return mean_pred, uncertainty
