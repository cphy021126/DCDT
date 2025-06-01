import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from skimage import measure


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = ((pred_bin + target_bin) > 0).sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def hd95_score(pred, target, threshold=0.5):
    pred_np = (pred.squeeze(1).detach().cpu().numpy() > threshold).astype(np.uint8)
    target_np = (target.squeeze(1).detach().cpu().numpy() > threshold).astype(np.uint8)

    hd_list = []

    for p, t in zip(pred_np, target_np):
        # 提取边界点坐标
        p_contour = np.argwhere(measure.find_contours(p, 0.5)[0]) if np.any(p) else None
        t_contour = np.argwhere(measure.find_contours(t, 0.5)[0]) if np.any(t) else None

        if p_contour is None or t_contour is None:
            hd_list.append(999.0)  # 默认超大值
            continue

        # 计算 directed hausdorff 双向距离
        hd1 = directed_hausdorff(p_contour, t_contour)[0]
        hd2 = directed_hausdorff(t_contour, p_contour)[0]
        hd = max(hd1, hd2)
        hd_list.append(hd)

    return np.mean(hd_list)

import torch
import numpy as np
from medpy.metric import binary

def calculate_metrics(pred, target):
    """
    计算 DICE, IoU 和 HD95 指标（用于医学图像分割）
    输入：
        pred: [B, 1, H, W] sigmoid 概率图
        target: [B, 1, H, W] 真实标签
    返回：
        dice, iou, hd95（单个 batch 的平均）
    """
    pred = (pred > 0.5).float()
    dice_all, iou_all, hd95_all = [], [], []

    for i in range(pred.shape[0]):
        pred_np = pred[i, 0].cpu().numpy().astype(np.uint8)
        target_np = target[i, 0].cpu().numpy().astype(np.uint8)

        try:
            dice = binary.dc(pred_np, target_np)
            iou = binary.voe(pred_np, target_np)
            hd95 = binary.hd95(pred_np, target_np)
        except:
            dice, iou, hd95 = 0.0, 1.0, 999.0  # 防止空白图报错

        dice_all.append(dice)
        iou_all.append(1 - iou)
        hd95_all.append(hd95)

    return np.mean(dice_all), np.mean(iou_all), np.mean(hd95_all)
