import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from models.attention_unet import AttU_Net
from utils.metrics_gray import dice_score, iou_score, hd95

# ==== Config ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
BATCH_SIZE = 1
MODEL_PATH = "/data16t/sihan/DDT/outputs/checkpoints/attunet_best.pth"
TEST_IMG_DIR = "/data16t/sihan/DDT/data/image/test/images"
TEST_LABEL_DIR = "/data16t/sihan/DDT/data/image/test/labels"

# ==== Dataset ====
class TestDataset(Dataset):
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

# ==== Transforms ====
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ==== Load Data ====
test_dataset = TestDataset(TEST_IMG_DIR, TEST_LABEL_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Load Model ====
model = AttU_Net(img_ch=1, output_ch=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== Evaluation ====
dice_scores, iou_scores, hd95_scores = [], [], []

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        outputs = model(imgs)
        preds = (outputs > 0.5).float()
        dice_scores.append(dice_score(preds, masks))
        iou_scores.append(iou_score(preds, masks))
        hd95_scores.append(hd95(preds, masks))

# ==== Report ====
print(f"Test Results on {len(test_dataset)} samples:")
print(f" - Dice Score: {np.mean(dice_scores):.4f}")
print(f" - IoU Score : {np.mean(iou_scores):.4f}")
print(f" - HD95      : {np.mean(hd95_scores):.2f}")
