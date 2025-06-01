import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
#from models.student_model_shared_encoder import StudentNet
from models.student_model_4 import StudentNet
from models.transunet_v2_localvit_final_fixed_v6_fixed import TransUNetV2
from models.attention_unet import AttU_Net
from utils.metrics_gray import dice_score, iou_score, hd95
from utils.teacher_selector import select_dominant_teacher
from utils.uncertainty_utils import compute_uncertainty
from utils.ema_utils import smart_ema_update
from tqdm import tqdm
from utils.select_dominant_teacher import select_dominant_teacher
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from skimage.morphology import binary_erosion
from utils.calculate import ai
# ==== Config ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
EPOCHS = 50
WARMUP_EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
CHECKPOINT_DIR = "/data16t/sihan/DDT/outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==== Dataset ====
# class GraySegDataset(Dataset):
#     def __init__(self, img_dir, label_dir, transform=None):
#         self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
#         self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         img = Image.open(self.image_paths[idx]).convert("L").resize(IMG_SIZE)
#         mask = Image.open(self.label_paths[idx]).convert("L").resize(IMG_SIZE)
#         if self.transform:
#             img = self.transform(img)
#             mask = self.transform(mask)
#         return img, mask

class GraySegDataset(Dataset):
    def __init__(self, img_dir, label_dir=None, transform=None):
        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)]) if label_dir else None
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L").resize(IMG_SIZE)
        img = self.transform(img) if self.transform else img

        if self.label_paths:
            mask = Image.open(self.label_paths[idx]).convert("L").resize(IMG_SIZE)
            mask = self.transform(mask) if self.transform else mask
        else:
            mask = torch.zeros_like(img)  # 返回假标签以兼容 zip 结构

        return img, mask

#transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

labeled_dataset = GraySegDataset("/data16t/sihan/DDT/data/image/train/labeled", "/data16t/sihan/DDT/data/image/train/labels", transform=transform)
unlabeled_dataset = GraySegDataset("/data16t/sihan/DDT/data/image/train/unlabeled", None, transform=transform)
val_dataset = GraySegDataset("/data16t/sihan/DDT/data/image/val/images", "/data16t/sihan/DDT/data/image/val/labels", transform=transform)

labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# def safe_load_transunet(model, ckpt_path):
#     print(f"🧠 正在安全加载教师模型权重：{ckpt_path}")
#     state_dict = torch.load(ckpt_path, map_location='cpu')
#
#     # 过滤掉与conv1相关的所有权重
#     filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("conv1")}
#
#     # 加载其余权重
#     missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
#
#     print("✅ conv1 权重已忽略，其他权重加载完毕")
#     print("🟡 Missing keys:", missing)
#     print("🔵 Unexpected keys:", unexpected)

def safe_load_transunet(model, ckpt_path):
    print(f"🧠 正在安全加载教师模型权重：{ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # ❗严格排除所有 conv1 子层，如 conv1.0.weight、conv1.1.running_mean 等
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("conv1.")}

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    print("✅ conv1 权重已彻底忽略")
    print("🟡 Missing keys:", missing)
    print("🔵 Unexpected keys:", unexpected)



def compute_boundary_loss(pred, target, threshold=0.5):
    """
    计算边界损失，使用腐蚀操作来提取边界
    pred: 模型的预测结果
    target: 真实标签
    threshold: 用于将预测结果和真实标签转化为二值图像的阈值
    """
    # 将预测结果和真实标签二值化
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    # 如果输入是 3D 张量，添加一个 batch 维度，使其成为 4D 张量
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)  # 转换为 [1, C, H, W]
    if len(target.shape) == 3:
        target = target.unsqueeze(0)  # 转换为 [1, C, H, W]

    # 使用卷积模拟腐蚀操作
    kernel = torch.ones(1, 1, 3, 3).to(pred.device)  # 使用 3x3 卷积核
    pred_boundary = F.conv2d(pred, kernel, padding=1)  # 卷积操作来模拟腐蚀
    target_boundary = F.conv2d(target, kernel, padding=1)

    # 计算边界损失
    pred_boundary = (pred_boundary > 0).float()  # 对卷积结果进行阈值处理
    target_boundary = (target_boundary > 0).float()

    boundary_loss = F.mse_loss(pred_boundary, target_boundary)  # 使用MSE损失计算边界损失

    return boundary_loss

# 计算 Hausdorff 损失
def hausdorff_distance(pred, target, threshold=0.5):
    """
    计算 Hausdorff 距离损失，确保边界更精确
    pred: 模型的预测结果
    target: 真实标签
    threshold: 用于二值化图像的阈值
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    # 使用卷积模拟腐蚀操作来提取边界
    kernel = torch.ones(1, 1, 3, 3).to(pred.device)  # 使用 3x3 卷积核
    pred_boundary = F.conv2d(pred, kernel, padding=1)  # 卷积操作来模拟腐蚀
    target_boundary = F.conv2d(target, kernel, padding=1)

    # 计算边界损失（使用MSE）
    pred_boundary = (pred_boundary > 0).float()  # 对卷积结果进行阈值处理
    target_boundary = (target_boundary > 0).float()

    boundary_loss = F.mse_loss(pred_boundary, target_boundary)  # 使用MSE损失计算边界损失

    return boundary_loss


# ==== Models ====
#student = StudentNet().to(DEVICE)
student = StudentNet(
    img_size=224,
    num_classes=1,
    vit_weights="/data16t/sihan/DDT/weights/vit_base_patch16_224.pth"
).to(DEVICE)

#teacher_A = TransUNetV2(img_size=224, in_channels=1, num_classes=1).to(DEVICE)
# teacher_A = TransUNetV2(
#     img_size=224,
#     in_channels=1,
#     num_classes=1,
#     vit_weights="/data16t/sihan/DDT/weights/vit_base_patch16_224.pth",
#     pretrained_ckpt="/data16t/sihan/DDT/outputs/checkpoints/transunetv2_pretrained_full2500_with_val.pth"
# ).to(DEVICE)
teacher_A = TransUNetV2(
    img_size=224,
    in_channels=1,
    num_classes=1,
    vit_weights="/data16t/sihan/DDT/weights/vit_base_patch16_224.pth"  # 仅加载ViT权重
).to(DEVICE)
safe_load_transunet(teacher_A, "/data16t/sihan/DDT/outputs/checkpoints/transunetv2_pretrained_full2500_with_val.pth")
# 临时加入：打印 conv1 实际结构

print("teacher_A.conv1[0]:", teacher_A.conv1[0])
print("🔥 After loading weights - conv1 weight shape:", teacher_A.conv1[0].weight.shape)


teacher_B = AttU_Net(img_ch=1, output_ch=1).to(DEVICE)

# Load teacher pretrained weights
#teacher_A.load_state_dict(torch.load("/data16t/sihan/DDT/outputs/checkpoints/transunet_best.pth"))
teacher_B.load_state_dict(torch.load("/data16t/sihan/DDT/outputs/checkpoints/attunet_best.pth"))
teacher_A.eval()
teacher_B.eval()

# Optimizer
optimizer = torch.optim.Adam(student.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True)
#bce = nn.BCELoss()
bce = nn.BCEWithLogitsLoss()


best_dice = 0

# ==== Training Loop ====
for epoch in range(EPOCHS):
    student.train()
    loop = zip(labeled_loader, unlabeled_loader)
    pbar = tqdm(loop, total=min(len(labeled_loader), len(unlabeled_loader)), desc=f"[Epoch {epoch+1}/{EPOCHS}]")

    for (x_l, y_l), (x_u, _) in pbar:
        x_l, y_l = x_l.to(DEVICE), y_l.to(DEVICE)
        x_u = x_u.to(DEVICE)

        # === forward ===
        out_student_l, out_glb_l, out_loc_l,_ = student(x_l)
        # 计算每个分支的损失
        loss_glb = bce(out_glb_l, y_l)  # 全局分支损失
        loss_loc = bce(out_loc_l, y_l)  # 局部分支损失

        # 总的有监督损失
        loss_sup = loss_glb + loss_loc
        #loss_sup = bce(out_student_l, y_l)

        if epoch >= WARMUP_EPOCHS:
            with torch.no_grad():
                #print("x_u shape:", x_u.shape)
                pred_A = torch.sigmoid(teacher_A(x_u))
                pred_B = torch.sigmoid(teacher_B(x_u))
                # 计算教师一致性损失（L1 损失）
                teacher_consistency_loss = torch.mean(torch.abs(pred_A - pred_B))  # 计算教师A和教师B的L1损失

                #uncert_A = compute_uncertainty(teacher_A, x_u)
                #uncert_B = compute_uncertainty(teacher_B, x_u)
                #print("✅ Before second pred_A - teacher_A.conv1:", teacher_A.conv1[0])

                # dominant, wA, wB, confA, confB = select_dominant_teacher(pred_A, pred_B,
                #     uncertainty_A=uncert_A, uncertainty_B=uncert_B,
                #     epoch=epoch, total_epochs=EPOCHS)
                dominant, wA, wB, confA, confB = select_dominant_teacher(
                    pred_A, pred_B,
                    epoch=epoch,
                    total_epochs=EPOCHS
                )
                #
                # pseudo_label = wA * pred_A + wB * pred_B


            # out_student_u, _, _ = student(x_u)

            # 获取学生模型的输出（包括伪标签和注意力图）
            out_student_u, out_glb_u, out_loc_u, attention_map = student(x_u)
            # 根据 attention_map 计算每个像素的伪标签
            pseudo_label = attention_map * pred_A + (1 - attention_map) * pred_B
            # print(pseudo_label.min(), pseudo_label.max())
            # print("pseudo_label shape:", pseudo_label.shape)  # 确保它是 [batch_size, channels, height, width]
            # print("out_student_u min/max:", out_student_u.min(), out_student_u.max())
            # print("pseudo_label min/max:", pseudo_label.min(), pseudo_label.max())

            # **高置信度筛选**：生成高置信度区域的 mask
            confidence_mask = (pseudo_label > 0.9) | (pseudo_label < 0.1)  # 保留接近0和1的区域

            # loss_unsup = bce(out_student_u, pseudo_label)
            # 计算无监督损失，并只保留高置信度区域
            loss_unsup = bce(out_student_u, pseudo_label)  # 原始无监督损失
            loss_unsup = loss_unsup * confidence_mask.float()  # 只保留高置信区域
            loss_unsup = loss_unsup.sum() / (confidence_mask.sum() + 1e-6)  # 归一化

            # **加入边界损失**：计算边界损失
            boundary_loss = compute_boundary_loss(out_student_l, y_l)  # 对有监督部分计算边界损失
            hausdorff_loss = hausdorff_distance(out_student_l, y_l)  # 计算Hausdorff损失
            # 将教师一致性损失添加到总损失中
            lambda_teacher_consistency = 0.1  # 可以调整此超参数
            lambda_boundary = 0.5  # 边界损失的权重
            lambda_hd = 1.0  # Hausdorff损失的权重

            #loss = loss_sup + loss_unsup + lambda_teacher_consistency * teacher_consistency_loss
            loss = loss_sup + loss_unsup + lambda_teacher_consistency * teacher_consistency_loss + lambda_boundary * boundary_loss + lambda_hd * hausdorff_loss
            #loss = loss_sup + loss_unsup + lambda_teacher_consistency * teacher_consistency_loss + lambda_boundary * boundary_loss
            # loss = loss_sup + loss_unsup
            #loss = loss_sup + loss_unsup
        else:
            loss_unsup = torch.tensor(0.0).to(DEVICE)
            loss = loss_sup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === EMA ===
        if epoch >= WARMUP_EPOCHS:
            if dominant == "A":
                smart_ema_update(student.branch_global, teacher_A, uncertainty_map=1 - confA)
            else:
                smart_ema_update(student.branch_local, teacher_B, uncertainty_map=1 - confB)

        #pbar.set_postfix({"SupLoss": loss_sup.item(), "UnsupLoss": loss_unsup.item()})
        # pbar.set_postfix({
        #     "SupLoss": loss_sup.item(),
        #     "UnsupLoss": loss_unsup.item() if epoch >= WARMUP_EPOCHS else 0.0
        # })
        pbar.set_postfix({
            "Loss": loss.item(),
        })
    # ==== Validation ====
    student.eval()
    dice_scores, iou_scores, hd_scores = [], [], []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
            out, _, _,_ = student(x_val)
            pred = (out > 0.5).float()
            dice_scores.append(dice_score(pred, y_val))
            iou_scores.append(iou_score(pred, y_val))
            hd_scores.append(hd95(pred, y_val))

    dice_avg = (sum(dice_scores) / len(dice_scores))
    iou_avg = sum(iou_scores) / len(iou_scores)
    hd_avg = sum(hd_scores) / len(hd_scores)
    #dice_avg,iou_avg,hd_avg = ai(dice_scores, iou_scores, hd_scores)

    print(f"Validation Dice: {dice_avg:.4f}, IoU: {iou_avg:.4f}, HD95: {hd_avg:.2f}")

    # Save best model
    if dice_avg > best_dice:
        best_dice = dice_avg
        torch.save(student.state_dict(), os.path.join(CHECKPOINT_DIR, "student_best8.pth"))
        print(f"✅ Saved new best model with Dice {best_dice:.4f}")
    # 调用学习率调度器调整学习率
    scheduler.step(dice_avg)  # 这里传入的是当前的 Dice 指标