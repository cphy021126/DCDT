import os
import torch
from PIL import Image
from torchvision import transforms

# 设置路径
input_dir = r"/data16t/sihan/DDT/data/image/test/images"
output_dir = r"/data16t/sihan/DDT/data/image/test/images_resize"
os.makedirs(output_dir, exist_ok=True)

# 定义与训练相同的 transforms
IMG_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# 处理每个标签文件
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 打开标签图像并转换为灰度
        img = Image.open(input_path).convert('L')

        # 应用 transforms
        img_transformed = transform(img)

        # 转回 PIL Image 并保存
        img_resized = transforms.ToPILImage()(img_transformed)
        img_resized.save(output_path)

        print(f"已处理: {filename}")

print("所有标签处理完成！")
