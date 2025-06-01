import torch
state = torch.load("/data16t/sihan/DDT/outputs/checkpoints/transunetv2_pretrained_full2500_with_val.pth")
print(state['conv1.0.weight'].shape)  # 应该是 [128, 1, 7, 7]，如果是 [128, 256, 7, 7] 就说明模型错了
