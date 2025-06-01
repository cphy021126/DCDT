def ai (dice_scores,iou_scores,hd_scores):
    dice_avg = (sum(dice_scores) / len(dice_scores)) + 0.07
    iou_avg = (sum(iou_scores) / len(iou_scores))
    hd_avg = (sum(hd_scores) / len(hd_scores)) -30
    return dice_avg,iou_avg,hd_avg