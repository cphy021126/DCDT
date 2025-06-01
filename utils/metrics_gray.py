import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from skimage import measure

def dice_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred + target).clamp(0, 1).sum(dim=(1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def hd95(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(np.uint8)
    target = (target > 0.5).cpu().numpy().astype(np.uint8)
    batch_size = pred.shape[0]
    hd95_batch = []

    for i in range(batch_size):
        if np.sum(pred[i]) == 0 or np.sum(target[i]) == 0:
            hd95_batch.append(0)
            continue

        pred_contour = measure.find_contours(pred[i, 0], 0.5)
        target_contour = measure.find_contours(target[i, 0], 0.5)

        if not pred_contour or not target_contour:
            hd95_batch.append(0)
            continue

        pred_pts = np.vstack(pred_contour)
        target_pts = np.vstack(target_contour)

        d1 = directed_hausdorff(pred_pts, target_pts)[0]
        d2 = directed_hausdorff(target_pts, pred_pts)[0]
        hd95_batch.append(max(d1, d2))

    return np.mean(hd95_batch)
