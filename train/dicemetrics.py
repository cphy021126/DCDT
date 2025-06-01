def dice_to_iou():
    try:
        # 获取用户输入的Dice值（支持百分比或小数形式）
        dice_input = input("请输入Dice系数（0-100或0-1格式，例如：80.17 或 0.8017）：")
        dice = float(dice_input.strip())

        # 判断输入格式（百分比或小数）
        if dice > 1.0:
            if dice > 100.0:
                raise ValueError("输入值不能超过100%")
            dice /= 100.0  # 将百分比转换为小数

        # 检查输入范围
        if not (0.0 <= dice <= 1.0):
            raise ValueError("Dice值应在0到1之间或0到100之间")

        # 计算IoU
        iou = dice / (2 - dice)

        # 输出结果（百分比形式，保留2位小数）
        print(f"Dice = {dice * 100:.2f}%  →  IoU = {iou * 100:.2f}%")

    except ValueError as e:
        print(f"错误：{e}")


# 运行函数
dice_to_iou()