import torch
from models.transunet_v2_localvit_final_fixed_v6_fixed import TransUNetV2

# 设置路径
vit_weights = "/data16t/sihan/DDT/weights/vit_base_patch16_224.pth"
pretrained_ckpt = "/data16t/sihan/DDT/outputs/checkpoints/transunet_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = TransUNetV2(
    img_size=224,
    in_channels=1,
    num_classes=1,
    vit_weights=vit_weights,
    pretrained_ckpt=pretrained_ckpt
).to(device)

# 打印 conv1 权重结构确认
print("✅ conv1 权重 shape:", model.conv1[0].weight.shape)  # 应为 [128, 1, 7, 7]

# 构造一张假图像输入
dummy_input = torch.randn(1, 1, 224, 224).to(device)

# 前向推理
with torch.no_grad():
    output = model(dummy_input)
    print("✅ 输出 shape:", output.shape)  # 应为 [1, 1, H, W]
