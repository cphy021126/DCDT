import torch
import torch.nn.functional as F

def compute_entropy(pred, eps=1e-6):
    return -pred * torch.log(pred + eps) - (1 - pred) * torch.log(1 - pred + eps)

def compute_consistency(pred_A, pred_B):
    return 1.0 - F.l1_loss(pred_A, pred_B, reduction='mean')  # 越接近 1，越一致

@torch.no_grad()
def select_dominant_teacher(pred_A, pred_B, epoch, total_epochs, alpha=0.4, beta=0.4, gamma=0.2):
    """
    综合评估主导教师。
    - alpha：entropy 权重（越小越自信）
    - beta：一致性评分权重
    - gamma：时间调节因子权重（前期偏向 Teacher A）
    """

    # === 熵值越小越可信 ===
    entropy_A = compute_entropy(pred_A).mean()
    entropy_B = compute_entropy(pred_B).mean()
    score_entropy_A = 1.0 - entropy_A
    score_entropy_B = 1.0 - entropy_B

    # === 一致性评分 ===
    consistency_score = compute_consistency(pred_A, pred_B)

    # === 时间因子 ===
    time_factor = epoch / total_epochs  # 越往后越趋于1

    # === 综合评分 ===
    final_score_A = alpha * score_entropy_A + beta * consistency_score + gamma * (1 - time_factor)
    final_score_B = alpha * score_entropy_B + beta * (1 - consistency_score) + gamma * time_factor

    # === 权重归一化 ===
    total = final_score_A + final_score_B + 1e-6
    wA = final_score_A / total
    wB = final_score_B / total

    dominant = "A" if wA >= wB else "B"

    return dominant, wA, wB, score_entropy_A.item(), score_entropy_B.item()
