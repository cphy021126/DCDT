import torch
import torch.nn.functional as F

def generate_pseudo_label(pred_vit, unc_vit, pred_cnn, unc_cnn):
    """
    根据教师模型的输出和不确定性，生成融合后的伪标签

    参数：
        pred_vit: tensor [B, 1, H1, W1]，ViT 教师模型输出概率图
        unc_vit:  tensor [B, 1, H1, W1]，ViT 教师模型的不确定性图
        pred_cnn: tensor [B, 1, H2, W2]，CNN 教师模型输出概率图
        unc_cnn:  tensor [B, 1, H2, W2]，CNN 教师模型的不确定性图

    返回：
        pseudo_label: tensor [B, 1, H, W]，融合后的伪标签
        dominant: str，当前主导教师 ("vit" 或 "cnn")
    """
    target_size = pred_cnn.shape[2:]  # 以 cnn 教师的分辨率为标准

    pred_vit = F.interpolate(pred_vit, size=target_size, mode='bilinear', align_corners=False)
    unc_vit = F.interpolate(unc_vit, size=target_size, mode='bilinear', align_corners=False)
    unc_cnn = F.interpolate(unc_cnn, size=target_size, mode='bilinear', align_corners=False)

    w_vit = torch.exp(-unc_vit)
    w_cnn = torch.exp(-unc_cnn)
    w_sum = w_vit + w_cnn + 1e-8  # 避免除0

    w_vit /= w_sum
    w_cnn /= w_sum

    pseudo_label = w_vit * pred_vit + w_cnn * pred_cnn
    dominant = "vit" if w_vit.mean() >= w_cnn.mean() else "cnn"

    return pseudo_label.detach(), dominant
