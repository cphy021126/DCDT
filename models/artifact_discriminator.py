import torch
import torch.nn as nn

class ArtifactDiscriminator(nn.Module):
    """
    判别器网络：用于判别伪影图像与原始图像的分布差异。
    输入为学生模型对伪影图像的预测图（概率图），输出为是否为"真实分布"。
    """
    def __init__(self, in_channels=1):
        super(ArtifactDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 输出为概率值
        )

    def forward(self, x):
        return self.net(x)
