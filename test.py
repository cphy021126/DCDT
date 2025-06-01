import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.loader import SemiDataset
from models.Student import DualBranchStudent
from utils.metrics import calculate_metrics
from torchvision.utils import save_image
from tqdm import tqdm


def upsample_like(pred, target):
    return F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

def test(model_path, data_root, save_dir="outputs/test_results", batch_size=1, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Testing on device: {device}")

    # === 加载模型 ===
    model = DualBranchStudent().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 加载测试集 ===
    test_dataset = SemiDataset(data_root, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # === 创建保存目录 ===
    os.makedirs(save_dir, exist_ok=True)

    dice_list, iou_list, hd95_list = [], [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(test_loader, desc="Testing")):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred = upsample_like(pred, y)

            # === 计算指标 ===
            dice, iou, hd95 = calculate_metrics(pred, y)
            dice_list.append(dice)
            iou_list.append(iou)
            hd95_list.append(hd95)

            # === 可视化输出保存 ===
            pred_vis = (pred > 0.5).float()
            save_image(pred_vis, os.path.join(save_dir, f"pred_{idx:03d}.png"))
            save_image(y, os.path.join(save_dir, f"gt_{idx:03d}.png"))

    # === 输出最终平均结果 ===
    print("\n📊 Final Evaluation on Test Set:")
    print(f"DICE: {torch.tensor(dice_list).mean():.4f}")
    print(f"IoU:  {torch.tensor(iou_list).mean():.4f}")
    print(f"HD95: {torch.tensor(hd95_list).mean():.2f}")


if __name__ == '__main__':
    test(
        model_path="outputs/checkpoints/best_model.pth",
        data_root="data/image",
        save_dir="outputs/test_results",
        batch_size=1
    )
