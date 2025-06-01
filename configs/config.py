class Args:
    num_epochs = 100
    lambda_sup = 1.0
    lambda_pseudo = 1.0
    lambda_art = 0.2
    lambda_adv = 0.1

    ema_alpha = 0.99   # 教师的平滑因子
    ema_beta = 0.05    # 学生的反馈吸收比例
    warmup_epochs = 5