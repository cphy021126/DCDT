import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
from models.attention_unet import AttU_Net
from utils.metrics_gray import dice_score, iou_score, hd95

# ==== Config ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 50
LR = 1e-4
IMG_SIZE = (224, 224)
DATA_ROOT = "/data16t/sihan/DDT/data/image"
MODEL_SAVE_PATH = "/data16t/sihan/DDT/outputs/checkpoints/attunet_best.pth"

# ==== Dataset ====
class FullSupervisedDataset(Dataset):
    def __init__(self, labeled_dir, unlabeled_dir, label_dir, transform=None):
        self.image_paths = sorted([
            os.path.join(labeled_dir, f) for f in os.listdir(labeled_dir)
        ] + [
            os.path.join(unlabeled_dir, f) for f in os.listdir(unlabeled_dir)
        ])
        self.label_paths = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
        ])
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

# ==== Transforms ====
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ==== Load Data ====
train_dataset = FullSupervisedDataset(
    os.path.join(DATA_ROOT, "train/labeled"),
    os.path.join(DATA_ROOT, "train/unlabeled"),
    os.path.join(DATA_ROOT, "train/labels"),
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_image_dir = os.path.join(DATA_ROOT, "val/images")
val_label_dir = os.path.join(DATA_ROOT, "val/labels")

class ValDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
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

val_dataset = ValDataset(val_image_dir, val_label_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ==== Model ====
model = AttU_Net(img_ch=1, output_ch=1).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_dice = 0.0

# ==== Train ====
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{NUM_EPOCHS}] Training")
    for imgs, masks in pbar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    # ==== Validation ====
    model.eval()
    dice_scores, iou_scores, hd95_scores = [], [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            preds = (outputs > 0.5).float()
            dice_scores.append(dice_score(preds, masks))
            iou_scores.append(iou_score(preds, masks))
            hd95_scores.append(hd95(preds, masks))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_hd95 = np.mean(hd95_scores)

    print(f"[Validation] Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, HD95: {avg_hd95:.2f}")

    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved.")
