import torch
import copy
import torch.nn.functional as F

def enable_dropout(model):
    """
    启用模型中的 Dropout 层（即使在 eval 模式下）用于 MC Dropout。
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# @torch.no_grad()
# def compute_uncertainty(model, input_tensor, T=4, sigmoid=True):
#     """
#     使用 MC Dropout 重复预测 T 次，计算输出的不确定度。
#     - model: 教师模型（需含 Dropout）
#     - input_tensor: shape [B, C, H, W]
#     - T: 重复次数
#     - sigmoid: 是否对输出做 sigmoid
#     """
#     model.eval()
#     enable_dropout(model)
#
#     preds = []
#     for _ in range(T):
#         output = model(input_tensor)
#         if isinstance(output, (tuple, list)):
#             output = output[0]  # 学生结构返回多个分支，取主分支
#         if sigmoid:
#             output = torch.sigmoid(output)
#         preds.append(output)
#
#     preds = torch.stack(preds, dim=0)  # [T, B, 1, H, W]
#     mean_pred = preds.mean(dim=0)
#     var_pred = preds.var(dim=0)
#     return var_pred  # 越大越不确定
# @torch.no_grad()
# def compute_uncertainty(model, input_tensor, T=4, sigmoid=True):
#     """
#     使用 MC Dropout 重复预测 T 次，计算输出的不确定度。
#     """
#     enable_dropout(model)  # ✅ 保持 Dropout 开启
#     # ❌ 不再 model.eval()，以避免 BN 出错
#
#     preds = []
#     for _ in range(T):
#         output = model(input_tensor)
#         if isinstance(output, (tuple, list)):
#             output = output[0]
#         if sigmoid:
#             output = torch.sigmoid(output)
#         preds.append(output)
#
#     preds = torch.stack(preds, dim=0)
#     var_pred = preds.var(dim=0)
#     return var_pred


@torch.no_grad()
def compute_uncertainty(model, input_tensor, T=4, sigmoid=True):
    model_copy = copy.deepcopy(model)  # ✅ 使用完全分离副本
    model_copy.to(input_tensor.device)
    enable_dropout(model_copy)

    preds = []
    for _ in range(T):
        output = model_copy(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if sigmoid:
            output = torch.sigmoid(output)
        preds.append(output)

    preds = torch.stack(preds, dim=0)
    var_pred = preds.var(dim=0)
    return var_pred
