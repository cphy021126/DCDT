import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models.attention_unet import AttU_Net
from utils.metrics_gray import dice_score, iou_score, hd95

# ==== Config ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
MODEL_PATH = "/data16t/sihan/DDT/outputs/checkpoints/attunet_best.pth"

# ==== Input (手动设置测试图像路径) ====
IMG_PATH = "/data16t/sihan/DDT/data/image/test/images/test_img_0233.png"
LABEL_PATH = "/data16t/sihan/DDT/data/image/test/labels/test_label_0233.png"

# ==== Load Model ====
model = AttU_Net(img_ch=1, output_ch=1).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# ==== Transform ====
transform = transforms.Compose([
    transforms.ToTensor()
])

# ==== Load Image ====
img = Image.open(IMG_PATH).convert("L").resize(IMG_SIZE)
mask = Image.open(LABEL_PATH).convert("L").resize(IMG_SIZE)

img_tensor = transform(img).unsqueeze(0).to(DEVICE)
mask_tensor = transform(mask).unsqueeze(0).to(DEVICE)

# ==== Inference ====
with torch.no_grad():
    output = model(img_tensor)
    pred = (output > 0.5).float()

# ==== Metrics ====
dice = dice_score(pred, mask_tensor)
iou = iou_score(pred, mask_tensor)
hd = hd95(pred, mask_tensor)

print(f"Prediction Metrics for {os.path.basename(IMG_PATH)}")
print(f" - Dice Score: {dice:.4f}")
print(f" - IoU Score : {iou:.4f}")
print(f" - HD95      : {hd:.2f}")

# ==== Save & Show ====
pred_img = pred.squeeze().cpu().numpy() * 255
pred_img = Image.fromarray(pred_img.astype(np.uint8))
pred_img.save("predicted_mask.png")

# 可视化
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Input Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("prediction_visual.png")
plt.show()
