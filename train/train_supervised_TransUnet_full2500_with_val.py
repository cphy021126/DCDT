import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
#from models.TransUnet import TransUNetV2
from models.transunet_v2_localvit_final_fixed_v6_fixed import TransUNetV2
from utils.metrics_gray import dice_score, iou_score, hd95

# 数据集定义
class ThyroidDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        label = Image.open(self.label_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

def get_file_list(img_dir, label_dir, total):
    images = []
    labels = []
    for i in range(total):
        img_name = f"train_img_{i:04d}.png"
        label_name = f"train_label_{i:04d}.png"
        if i < 500:
            images.append(os.path.join(img_dir, "labeled", img_name))
        else:
            images.append(os.path.join(img_dir, "unlabeled", img_name))
        labels.append(os.path.join(img_dir, "labels", label_name))
    return images, labels

# 验证集
def get_val_dataset(val_dir, transform):
    val_img_dir = os.path.join(val_dir, "images")
    val_label_dir = os.path.join(val_dir, "labels")
    val_images = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir)])
    val_labels = sorted([os.path.join(val_label_dir, f) for f in os.listdir(val_label_dir)])
    return ThyroidDataset(val_images, val_labels, transform)

# 参数设置
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_EPOCHS = 50
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径设置
train_root = "/data16t/sihan/DDT/data/image/train"
#val_root = "data/image/val"
val_root = "/data16t/sihan/DDT/data/image/val"

# 数据增强
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# 加载训练集
train_imgs, train_labels = get_file_list(train_root, train_root, total=2500)
train_dataset = ThyroidDataset(train_imgs, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 加载验证集
val_dataset = get_val_dataset(val_root, transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 模型、损失、优化器
#model = TransUNetV2(img_size=IMG_SIZE, in_channels=1, num_classes=1).to(DEVICE)
model = TransUNetV2(
    img_size=224,
    in_channels=1,
    num_classes=1,
    vit_weights="/data16t/sihan/DDT/weights/vit_base_patch16_224.pth"  # 替换为你的实际路径
    #vit_weights=None
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{NUM_EPOCHS}] Training")
    for imgs, masks in pbar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        print("imgs shape:", imgs.shape)
        outputs = model(imgs).squeeze(1)
        masks = masks.squeeze(1)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(f"✅ Epoch {epoch+1} Training Loss: {total_loss/len(train_loader):.4f}")

    # 验证阶段
    model.eval()
    dices, ious, hd95s = [], [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs).squeeze(1)
            masks = masks.squeeze(1)

            preds = torch.sigmoid(outputs)
            dices.append(dice_score(preds, masks))
            ious.append(iou_score(preds, masks))
            hd95s.append(hd95(preds.unsqueeze(1), masks.unsqueeze(1)))

    print(f"🔍 Val Dice: {sum(dices)/len(dices):.4f}, IoU: {sum(ious)/len(ious):.4f}, HD95: {sum(hd95s)/len(hd95s):.2f}")

# 保存模型
# os.makedirs("checkpoints", exist_ok=True)
# torch.save(model.state_dict(), "checkpoints/transunetv2_pretrained_full2500_with_val.pth")
# print("✅ 微调完成，模型已保存到 checkpoints/transunetv2_pretrained_full2500_with_val.pth")
os.makedirs("/data16t/sihan/DDT/outputs/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "/data16t/sihan/DDT/outputs/checkpoints/transunetv2_pretrained_full2500_with_val.pth")
print("✅ 微调完成，模型已保存到 /data16t/sihan/DDT/outputs/checkpoints/transunetv2_pretrained_full2500_with_val.pth")
