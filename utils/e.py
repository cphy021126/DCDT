
import torch

def smart_ema_update(student_model, teacher_model, uncertainty_map=None,
                     base_alpha_t=0.99, base_alpha_s=0.95, uncertainty_scaling=True, gamma=0.05):
    """
    智能双向 EMA 更新：
    - 学生更新更快，教师更慢（异步速率）
    - 不确定度动态调整 EMA 速率
    - 学生保留残差项，避免信息丢失
    """
    if uncertainty_map is not None:
        # Clamp 不确定度范围，防止过小过大
        #uncertainty_map = torch.clamp(uncertainty_map, 0.0, 1.0).mean().item()
        if isinstance(uncertainty_map, torch.Tensor):
            uncertainty_map = torch.clamp(uncertainty_map, 0.0, 1.0).mean().item()
        else:
            uncertainty_map = max(0.0, min(1.0, float(uncertainty_map)))
    else:
        uncertainty_map = 0.0

    # 越确定 → alpha 越大，越“稳定”
    if uncertainty_scaling:
        alpha_t = base_alpha_t * (1 - uncertainty_map)
        alpha_s = base_alpha_s * (1 - uncertainty_map)
    else:
        alpha_t = base_alpha_t
        alpha_s = base_alpha_s

    for (name_s, param_s), (name_t, param_t) in zip(student_model.named_parameters(), teacher_model.named_parameters()):
        if param_s.requires_grad and param_t.requires_grad:
            # 教师更新慢，来自学生
            param_t.data = alpha_t * param_t.data + (1 - alpha_t) * param_s.data

            # 学生更新快，部分保留自我（残差）
            residual = param_s.data - param_t.data
            param_s.data = alpha_s * param_s.data + (1 - alpha_s) * param_t.data + gamma * residual
