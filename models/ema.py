# import torch
# #
# # @torch.no_grad()
# # def update_dual_ema(student_branch, teacher_model, alpha=0.99, beta=0.05):
# #     """
# #     动态双向 EMA 更新机制。
# #
# #     参数:
# #     - student_branch: 学生模型中的某一分支 (如 student_model.vit_branch)
# #     - teacher_model: 对应的教师模型 (如 teacher_vit)
# #     - alpha: 教师模型更新速率（通常 0.99）
# #     - beta: 学生模型残差反馈比例（通常很小，如 0.05）
# #
# #     作用:
# #     - 教师模型: teacher_param = α * teacher_param + (1 - α) * student_param
# #     - 学生模型: student_param += β * (teacher_param - student_param)
# #     """
# #
# #     for s_param, t_param in zip(student_branch.parameters(), teacher_model.parameters()):
# #         # Teacher ← Student (滑动平均)
# #         t_param.data.mul_(alpha).add_((1 - alpha) * s_param.data)
# #
# #     for s_param, t_param in zip(student_branch.parameters(), teacher_model.parameters()):
# #         # Student ← Teacher (残差反馈)
# #         s_param.data.add_(beta * (t_param.data - s_param.data))
#
# @torch.no_grad()
# def update_dual_ema(student_branch, teacher_model, alpha=0.99, beta=0.05):
#     """
#     动态双向 EMA，自动跳过 shape 不匹配的参数。
#     """
#     for (s_name, s_param), (t_name, t_param) in zip(student_branch.named_parameters(), teacher_model.named_parameters()):
#         if s_param.shape != t_param.shape:
#             continue
#         t_param.data.mul_(alpha).add_((1 - alpha) * s_param.data)
#
#     for (s_name, s_param), (t_name, t_param) in zip(student_branch.named_parameters(), teacher_model.named_parameters()):
#         if s_param.shape != t_param.shape:
#             continue
#         s_param.data.add_(beta * (t_param.data - s_param.data))

import torch

@torch.no_grad()
def update_dual_ema(student_branch, teacher_model, alpha=0.99, beta=0.05):
    """
    使用滑动平均更新教师模型的参数
    - alpha: 控制教师接收学生新知识的速度
    - beta: 适配不同维度参数（如ViT与CNN）
    """
    # === 检查两侧参数数目一致性 ===
    student_state = student_branch.state_dict()
    teacher_state = teacher_model.state_dict()

    updated_state = {}

    for name, param_s in student_state.items():
        if name in teacher_state:
            param_t = teacher_state[name]

            if param_s.shape == param_t.shape:
                # 经典 EMA 更新：t = α * t + (1 - α) * s
                updated_param = alpha * param_t + (1.0 - alpha) * param_s
                updated_state[name] = updated_param.clone()
            else:
                print(f"[EMA] ⚠️ Skip param: {name}, shape mismatch: {param_s.shape} vs {param_t.shape}")
        else:
            print(f"[EMA] ❌ Param missing in teacher: {name}")

    teacher_model.load_state_dict(updated_state, strict=False)
