import os
import shutil
import random
from tqdm import tqdm

random.seed(42)

def make_dirs(base):
    structure = [
        "train/labeled", "train/unlabeled", "train/labels",
        "val/images", "val/labels",
        "test/images", "test/labels"
    ]
    for path in structure:
        os.makedirs(os.path.join(base, path), exist_ok=True)

def rename_and_copy(src_img, src_label, dst_img, dst_label, prefix, idx):
    new_img_name = f"{prefix}_img_{idx:04d}.png"
    new_lbl_name = f"{prefix}_label_{idx:04d}.png"
    shutil.copy(src_img, os.path.join(dst_img, new_img_name))
    shutil.copy(src_label, os.path.join(dst_label, new_lbl_name))

def collect_images(img_dir, label_dir, prefix):
    img_list = sorted(os.listdir(img_dir))
    return [(os.path.join(img_dir, f), os.path.join(label_dir, f), f"{prefix}_{i:04d}") for i, f in enumerate(img_list)]

def split_dataset(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir, target_root,
                  train_num=2500, labeled_num=500, val_num=493, test_num=500):

    # 加载所有图像路径，并添加前缀防止命名冲突
    train_data = collect_images(train_img_dir, train_mask_dir, "train")
    test_data = collect_images(test_img_dir, test_mask_dir, "test")

    all_data = train_data + test_data
    random.shuffle(all_data)

    # 划分数据
    selected_train = all_data[:train_num]
    selected_val = all_data[train_num:train_num + val_num]
    selected_test = all_data[train_num + val_num:train_num + val_num + test_num]

    labeled = selected_train[:labeled_num]
    unlabeled = selected_train[labeled_num:]

    print(f"✅ 总训练图像: {len(selected_train)}，有标签: {len(labeled)}，无标签: {len(unlabeled)}")
    print(f"✅ 验证集: {len(selected_val)}，测试集: {len(selected_test)}")

    # 拷贝
    for i, (img_path, label_path, name_id) in enumerate(tqdm(labeled, desc="Copying labeled")):
        rename_and_copy(img_path, label_path,
                        os.path.join(target_root, "train/labeled"),
                        os.path.join(target_root, "train/labels"),
                        prefix="train", idx=i)

    for i, (img_path, label_path, name_id) in enumerate(tqdm(unlabeled, desc="Copying unlabeled")):
        rename_and_copy(img_path, label_path,
                        os.path.join(target_root, "train/unlabeled"),
                        os.path.join(target_root, "train/labels"),
                        prefix="train", idx=i + len(labeled))

    for i, (img_path, label_path, name_id) in enumerate(tqdm(selected_val, desc="Copying val")):
        rename_and_copy(img_path, label_path,
                        os.path.join(target_root, "val/images"),
                        os.path.join(target_root, "val/labels"),
                        prefix="val", idx=i)

    for i, (img_path, label_path, name_id) in enumerate(tqdm(selected_test, desc="Copying test")):
        rename_and_copy(img_path, label_path,
                        os.path.join(target_root, "test/images"),
                        os.path.join(target_root, "test/labels"),
                        prefix="test", idx=i)


if __name__ == '__main__':
    train_img_dir = r"C:\Users\Sihan Chen\Desktop\data/train/images"
    train_mask_dir = r"C:\Users\Sihan Chen\Desktop\data/train/masks"
    test_img_dir = r"C:\Users\Sihan Chen\Desktop\data/test/images"
    test_mask_dir = r"C:\Users\Sihan Chen\Desktop\data/test/masks"
    target_root = r"C:\Users\Sihan Chen\Desktop\data/image"

    make_dirs(target_root)
    split_dataset(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir, target_root)
