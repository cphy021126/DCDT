import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.transunet_v2_localvit_fixed_v7 import TransUNetV2
from utils.metrics_gray import dice_score, iou_score, hd95
from tqdm import tqdm

# === 配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 50
VIT_PATH = "/data16t/sihan/DDT/weights/vit_base_patch16_224.pth"
CKPT_PATH = "/data16t/sihan/DDT/outputs/checkpoints/transunet_v7_best.pth"
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

# === 数据集定义 ===
class GraySegDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L").resize(IMG_SIZE)
        mask = Image.open(self.label_paths[idx]).convert("L").resize(IMG_SIZE)
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = GraySegDataset("/data16t/sihan/DDT/data/image/train/labeled", "/data16t/sihan/DDT/data/image/train/labels", transform)
val_dataset = GraySegDataset("/data16t/sihan/DDT/data/image/val/images", "/data16t/sihan/DDT/data/image/val/labels", transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# === 模型与优化器 ===
model = TransUNetV2(img_size=224, in_channels=1, num_classes=1, vit_weights=VIT_PATH).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

best_dice = 0

# === 训练循环 ===
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]")
    total_loss = 0

    for imgs, masks in pbar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs).squeeze(1)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})

    # === 验证阶段 ===
    model.eval()
    dice_scores, iou_scores, hd_scores = [], [], []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
            pred = model(x_val).squeeze(1)
            pred_bin = (pred > 0.5).float()
            dice_scores.append(dice_score(pred_bin, y_val))
            iou_scores.append(iou_score(pred_bin, y_val))
            hd_scores.append(hd95(pred_bin, y_val))

    dice_avg = sum(dice_scores) / len(dice_scores)
    iou_avg = sum(iou_scores) / len(iou_scores)
    hd_avg = sum(hd_scores) / len(hd_scores)

    print(f"Validation Dice: {dice_avg:.4f}, IoU: {iou_avg:.4f}, HD95: {hd_avg:.2f}")

    # 保存最优模型
    if dice_avg > best_dice:
        best_dice = dice_avg
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"✅ Saved new best model with Dice {best_dice:.4f}")
