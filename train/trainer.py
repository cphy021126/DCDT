import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils

from utils.metrics import calculate_metrics
from utils.pseudo_label_utils import generate_pseudo_label
from models.ema import update_dual_ema


def upsample_like(pred, target):
    return F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)


def save_visual_debug(pred, label, epoch, index, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pred = torch.sigmoid(pred)  # logits → [0, 1]
    pred_bin = (pred > 0.5).float()
    grid = torch.cat([pred, pred_bin, label], dim=0)  # 可视化 3 图：预测、二值、标签
    vutils.save_image(grid, os.path.join(save_dir, f'epoch{epoch}_sample{index}.png'))


def train(student_model, teacher_vit, teacher_cnn, artifact_discriminator,
          train_loader, val_loader, optimizer, device, args):

    best_dice = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型部署到设备
    student_model = student_model.to(device)
    teacher_vit = teacher_vit.to(device)
    teacher_cnn = teacher_cnn.to(device)
    artifact_discriminator = artifact_discriminator.to(device)

    # loss 加前景加权
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    log_path = os.path.join(args.save_dir, "train_metrics.csv")
    with open(log_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Loss", "DICE", "IoU", "HD95", "Dominant"])

    for epoch in range(args.num_epochs):
        student_model.train()
        teacher_vit.eval()
        teacher_cnn.eval()

        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)

        for batch_idx, (x_l, y_l, x_u, x_art) in enumerate(progress_bar):
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u, x_art = x_u.to(device), x_art.to(device)

            # === 教师模型伪标签 ===
            with torch.no_grad():
                pred_vit, unc_vit = teacher_vit.predict_with_uncertainty(x_u, n_iter=3)
                pred_cnn, unc_cnn = teacher_cnn.predict_with_uncertainty(x_u, n_iter=3)
                pseudo_label, dominant = generate_pseudo_label(pred_vit, unc_vit, pred_cnn, unc_cnn)

            # === 学生预测 ===
            pred_l = student_model(x_l)
            pred_u = student_model(x_u)
            pred_art = student_model(x_art)

            pred_l = upsample_like(pred_l, y_l)
            pred_u = upsample_like(pred_u, pseudo_label)
            pred_art = upsample_like(pred_art, pseudo_label)

            # === 损失 ===
            sup_loss = criterion(pred_l, y_l)

            if epoch < args.warmup_epochs:
                loss = args.lambda_sup * sup_loss
            else:
                pseudo_loss = criterion(pred_u, pseudo_label)
                consis_loss = F.mse_loss(
                    artifact_discriminator(pred_u),
                    artifact_discriminator(pred_art)
                )
                loss = (args.lambda_sup * sup_loss +
                        args.lambda_pseudo * pseudo_loss +
                        args.lambda_art * consis_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # === EMA 更新（非 warmup）===
            if epoch >= args.warmup_epochs:
                if dominant == "vit":
                    update_dual_ema(student_model.vit_branch, teacher_vit, args.ema_alpha, args.ema_beta)
                else:
                    update_dual_ema(student_model.cnn_branch, teacher_cnn, args.ema_alpha, args.ema_beta)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), dominant=dominant)

            if batch_idx < 2 and epoch < 2:
                save_visual_debug(pred_l, y_l, epoch, batch_idx, os.path.join(args.save_dir, "debug_vis"))

        # === 验证 ===
        student_model.eval()
        dice_all, iou_all, hd95_all = [], [], []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred = student_model(x_val)
                pred = torch.sigmoid(upsample_like(pred, y_val))
                dice, iou, hd95 = calculate_metrics(pred, y_val)
                dice_all.append(dice)
                iou_all.append(iou)
                hd95_all.append(hd95)

        dice_score = torch.tensor(dice_all).mean().item()
        iou_score = torch.tensor(iou_all).mean().item()
        hd95_score = torch.tensor(hd95_all).mean().item()

        print(f"[Supervised] Epoch {epoch+1}: DICE={dice_score:.4f}, IoU={iou_score:.4f}, HD95={hd95_score:.2f}, Dominant={dominant}")

        with open(log_path, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch+1, round(total_loss, 4), round(dice_score, 4), round(iou_score, 4), round(hd95_score, 2), dominant])

        if dice_score > best_dice:
            best_dice = dice_score
            torch.save(student_model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"💾 Best model saved at epoch {epoch+1} with DICE: {best_dice:.4f}")
