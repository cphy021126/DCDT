# train/trainer_supervised.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import csv
import torchvision.utils as vutils
from utils.metrics import calculate_metrics

def upsample_like(pred, target):
    return F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

def save_visual_debug(pred, label, epoch, index, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pred = torch.sigmoid(pred)
    grid = torch.cat([pred, label], dim=0)  # [2, 1, H, W]
    vutils.save_image(grid, os.path.join(save_dir, f'epoch{epoch}_sample{index}.png'))

def train_supervised(student_model, train_loader, val_loader, optimizer, device, args):
    best_dice = 0.0
    student_model.to(device)

    log_path = os.path.join(args.save_dir, "supervised_metrics.csv")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Loss", "DICE", "IoU", "HD95"])

    pos_weight = torch.tensor([10.0], device=device)  # 正类稀疏时推荐加权
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(args.num_epochs):
        student_model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Supervised] Epoch {epoch+1}")

        for batch_idx, (x_l, y_l, _, _) in enumerate(pbar):
            x_l, y_l = x_l.to(device), y_l.to(device)

            pred_l = student_model(x_l)
            pred_l = upsample_like(pred_l, y_l)

            loss = criterion(pred_l, y_l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

            if batch_idx < 2 and epoch == 0:
                print(f"🔍 pred_l: {pred_l.min():.4f} ~ {pred_l.max():.4f}")
                print(f"🔍 y_l unique: {torch.unique(y_l)}")
                save_visual_debug(pred_l, y_l, epoch, batch_idx, os.path.join(args.save_dir, "debug_vis"))

        # === 验证 ===
        student_model.eval()
        dice_all, iou_all, hd95_all = [], [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)

                pred = torch.sigmoid(student_model(x_val))
                pred = upsample_like(pred, y_val)

                dice, iou, hd95 = calculate_metrics(pred, y_val)
                dice_all.append(dice)
                iou_all.append(iou)
                hd95_all.append(hd95)

        dice_mean = torch.tensor(dice_all).mean().item()
        iou_mean = torch.tensor(iou_all).mean().item()
        hd95_mean = torch.tensor(hd95_all).mean().item()

        print(f"📊 Epoch {epoch+1}: DICE={dice_mean:.4f}, IoU={iou_mean:.4f}, HD95={hd95_mean:.2f}")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, round(total_loss, 4), round(dice_mean, 4), round(iou_mean, 4), round(hd95_mean, 2)])

        if dice_mean > best_dice:
            best_dice = dice_mean
            torch.save(student_model.state_dict(), os.path.join(args.save_dir, "best_supervised.pth"))
            print(f"✅ Saved best model at epoch {epoch+1} with DICE={best_dice:.4f}")
