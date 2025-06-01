import torch

def select_dominant_teacher(pred_A, pred_B, uncertainty_A=None, uncertainty_B=None,
                             epoch=0, total_epochs=100, alpha_consistency=0.3, alpha_uncertainty=0.4, alpha_time=0.3):
    """
    综合三个因素判断主导教师：
    1）MC Dropout不确定度（低者更可信）
    2）教师一致性（输出接近表示都可信）
    3）训练进度偏好（前期偏全局A，后期偏局部B）

    返回:
    - dominant: "A" or "B"
    - weight_A: 作为融合权重
    - weight_B: 作为融合权重
    - confidence_score_A/B: 不确定度部分供 EMA 使用
    """

    # ----- 不确定度转置信度 -----
    if uncertainty_A is not None:
        conf_A = 1.0 - torch.clamp(uncertainty_A.mean(), 0.0, 1.0).item()
    else:
        conf_A = 0.5  # 若无信息默认中性

    if uncertainty_B is not None:
        conf_B = 1.0 - torch.clamp(uncertainty_B.mean(), 0.0, 1.0).item()
    else:
        conf_B = 0.5

    # ----- 一致性得分（越接近越高）-----
    consistency_score = 1.0 - torch.mean(torch.abs(pred_A - pred_B)).item()

    # ----- 时间偏好因子 -----
    progress = epoch / total_epochs
    time_weight_A = 1.0 - progress
    time_weight_B = progress

    # ----- 总融合评分 -----
    score_A = alpha_uncertainty * conf_A + alpha_consistency * consistency_score + alpha_time * time_weight_A
    score_B = alpha_uncertainty * conf_B + alpha_consistency * consistency_score + alpha_time * time_weight_B

    # ----- Normalize 为权重 -----
    total = score_A + score_B + 1e-8
    weight_A = score_A / total
    weight_B = score_B / total

    dominant = "A" if weight_A >= weight_B else "B"

    return dominant, weight_A, weight_B, conf_A, conf_B
