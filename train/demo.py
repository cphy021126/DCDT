# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os
#
# # Step 1: 读取真实灰度图像（替换为你自己的路径）
# img_path = r'/data16t/sihan/DDT/data/image/train/labeled/train_img_0000.png'  # 替换成你的图像路径
# output_path = r'/data16t/sihan/DDT/data/image'     # 输出保存路径
#
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# assert img is not None, f"图像路径无效：{img_path}"
# img = cv2.resize(img, (256, 256))  # 可根据实际需求调整分辨率
# H, W = img.shape
#
# # Step 2: 模拟中间特征图（模拟教师响应）
# img_tensor = torch.tensor(img / 255.0).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
# feature_map_g = torch.rand(1, 1, H, W) * 0.5 + img_tensor * 0.5
# feature_map_b = torch.rand(1, 1, H, W) * 0.5 + img_tensor * 0.2
#
# # Step 3: 生成注意力权重图（归一化）
# attn_concat = torch.cat([feature_map_g, feature_map_b], dim=1)  # [1, 2, H, W]
# attn_weights = torch.softmax(attn_concat, dim=1)  # [1, 2, H, W]
# attn_g = attn_weights[0, 0].detach().numpy()
# attn_b = attn_weights[0, 1].detach().numpy()
#
# # Step 4: 可视化并保存为 PNG 文件
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
# axs[0].imshow(img, cmap='gray')
# axs[0].set_title('Input Image')
# axs[0].axis('off')
#
# axs[1].imshow(attn_g, cmap='gray')
# axs[1].set_title('Attention Map $A_g$ (Global)')
# axs[1].axis('off')
#
# axs[2].imshow(attn_b, cmap='gray')
# axs[2].set_title('Attention Map $A_b$ (Boundary)')
# axs[2].axis('off')
#
# plt.tight_layout()
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"注意力图已保存至：{os.path.abspath(output_path)}")


import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Step 1: 读取真实灰度图像（替换为你自己的图像路径）
img_path = r'/data16t/sihan/DDT/data/image/test/images_resize/test_img_0161.png'  # 替换为你自己的图像路径
output_ag_path = r'/data16t/sihan/DDT/data/image/attention_ag.png'
output_ab_path = r'/data16t/sihan/DDT/data/image/attention_ab.png'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
assert img is not None, f"图像路径无效：{img_path}"
img = cv2.resize(img, (256, 256))  # 如有需要可调整
H, W = img.shape

# Step 2: 模拟两个教师的中间特征图
img_tensor = torch.tensor(img / 255.0).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
feature_map_g = torch.rand(1, 1, H, W) * 0.5 + img_tensor * 0.5
feature_map_b = torch.rand(1, 1, H, W) * 0.5 + img_tensor * 0.2

# Step 3: 计算 softmax 注意力图
attn_concat = torch.cat([feature_map_g, feature_map_b], dim=1)
attn_weights = torch.softmax(attn_concat, dim=1)
attn_g = attn_weights[0, 0].detach().numpy()
attn_b = attn_weights[0, 1].detach().numpy()

# Step 4: 将注意力图转换为 0~255 的 uint8 灰度图
attn_g_img = (attn_g * 255).astype(np.uint8)
attn_b_img = (attn_b * 255).astype(np.uint8)

# Step 5: 保存两个注意力图为 PNG
cv2.imwrite(output_ag_path, attn_g_img)
cv2.imwrite(output_ab_path, attn_b_img)
print(f"已保存 attention_ag.png 和 attention_ab.png 到目录：{os.getcwd()}")
